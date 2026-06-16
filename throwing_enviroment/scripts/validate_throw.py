#!/usr/bin/env python3
"""Validate a trained SAC throw-primitive model against test configurations.

Runs the agent deterministically on predefined target positions and measures
success rate (distance < threshold) across multiple attempts per target.

Usage:
    source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
    cd throwing_enviroment
    python scripts/validate_throw.py \
        --checkpoint logs/skrl/throwing_primitive/2026-06-14_.../checkpoints/agent_6000.pt \
        --num_tests 10 --attempts 3 --headless
    python scripts/validate_throw.py \
        --checkpoint logs/skrl/throwing_primitive/.../checkpoints/agent_6000.pt \
        --success_threshold 0.20
"""

import argparse
import os
import sys

import torch
import torch._dynamo  # noqa: F401
import torch._C  # noqa: F401
import torch.optim  # noqa: F401

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "Throwing"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate trained SAC throw model")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to agent checkpoint (.pt for skrl, .zip for SB3)")
parser.add_argument("--model_type", type=str, default="auto",
                    choices=["auto", "skrl", "sb3"],
                    help="Model type: auto-detect, skrl, or sb3")
parser.add_argument("--num_tests", type=int, default=10, help="Number of test configs to run (max 10)")
parser.add_argument("--attempts", type=int, default=3, help="Throws per target")
parser.add_argument("--success_threshold", type=float, default=0.15, help="Distance threshold for success (m)")
parser.add_argument("--no_plot", action="store_true", help="Skip plot generation")
parser.add_argument("--fast", action="store_true",
                    help="Use DirectRLEnv (faster, no IK overhead)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import List

from tasks.throwing_env_cfg import ThrowingEnvCfg, TABLE_Z
from tasks.throwing_env import ThrowingEnv
from tasks.events import _set_gripper_state
from tasks.throw_primitive import (
    ThrowPrimitiveExecutor,
    ThrowPrimitiveParams,
    map_action_to_params,
    DRINK_WORLD_X,
    DRINK_WORLD_Y,
    DRINK_WORLD_Z,
)
from tasks.throw_validation_configs import (
    THROW_TESTS,
    get_test_config,
    get_test_count,
    ThrowTestConfig,
)

PLAYING_SIDE = "right"
EE_BODY = "right_wrist_3_link"
ARM_JOINT_PATTERNS = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]
OBS_MAX_NORM = 3.0


@dataclass
class ThrowResult:
    test_id: int
    test_name: str
    target_x: float
    target_y: float
    distances: List[float] = field(default_factory=list)
    actions: List[List[float]] = field(default_factory=list)
    landings: List[List[float]] = field(default_factory=list)

    @property
    def best_distance(self):
        return min(self.distances) if self.distances else float("inf")

    @property
    def success_count(self):
        return sum(1 for d in self.distances if d < args_cli.success_threshold)

    @property
    def passed(self):
        return self.success_count > 0


def compute_obs(env, device):
    """Compute the 8D observation matching ThrowingPrimitiveEnv._get_obs()."""
    milk = env.scene["milk"]
    target = env.scene["target"]
    origins = env.scene.env_origins.to(device)

    milk_pos = milk.data.root_pos_w[:, :3] - origins
    target_pos = target.data.root_pos_w[:, :3] - origins

    dist_vec = milk_pos - target_pos
    dist = torch.norm(dist_vec, dim=-1, keepdim=True)
    dist_x = torch.abs(dist_vec[:, 0:1])
    dist_y = torch.abs(dist_vec[:, 1:2])

    robot_indicator = torch.full((1, 1), 1.0, device=device)

    obs = torch.cat([
        robot_indicator,
        target_pos[:, 0:1] / OBS_MAX_NORM,
        target_pos[:, 1:2] / OBS_MAX_NORM,
        milk_pos[:, 0:1] / OBS_MAX_NORM,
        milk_pos[:, 1:2] / OBS_MAX_NORM,
        dist / OBS_MAX_NORM,
        dist_x / OBS_MAX_NORM,
        dist_y / OBS_MAX_NORM,
    ], dim=-1)

    return obs


def set_target_position(env, target_x, target_y, device):
    """Set the basket (target) at a fixed position."""
    target_obj = env.scene["target"]
    origin = env.scene.env_origins[0].to(device)
    target_z = TABLE_Z + 0.001
    pos = torch.tensor([[target_x, target_y, target_z]], device=device) + origin.unsqueeze(0)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    target_obj.write_root_pose_to_sim(
        torch.cat([pos, quat], dim=-1),
        env_ids=torch.tensor([0], device=device),
    )


def load_sac_agent(env_wrapped, checkpoint_path, device):
    """Load SAC agent using skrl Runner (Option A)."""
    from skrl.utils.runner.torch import Runner

    agent_cfg_path = os.path.join(
        _PROJECT_ROOT, "source", "Throwing", "Throwing", "tasks",
        "throwing", "agents", "skrl_sac_cfg.yaml",
    )
    with open(agent_cfg_path, "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["seed"] = 42
    agent_cfg["trainer"]["timesteps"] = 1
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    runner = Runner(env_wrapped, agent_cfg)
    runner.agent.load(checkpoint_path)
    runner.agent.policy.eval()
    print(f"[INFO] Loaded SAC agent from: {checkpoint_path}")
    return runner.agent


def plot_results(results: List[ThrowResult], success_threshold: float, save_path: str, show: bool = True):
    """Generate validation plots: bird's-eye scatter + distance bar chart."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Plot 1: Bird's-eye scatter (top-down view) ─────────────────────
    ax1.set_title("Throw Validation — Bird's Eye View", fontsize=12)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")

    table_x_range = (-1.0, 1.0)
    table_y_range = (0.4, 1.85)
    table_rect = plt.Rectangle(
        (table_x_range[0], table_y_range[0]),
        table_x_range[1] - table_x_range[0],
        table_y_range[1] - table_y_range[0],
        linewidth=1, edgecolor="grey", facecolor="lightgrey", alpha=0.3,
    )
    ax1.add_patch(table_rect)

    ax1.plot(0, 0, "ks", markersize=10, label="Robot")

    for r in results:
        ax1.plot(r.target_x, r.target_y, "o", color="blue", markersize=10, alpha=0.7)
        ax1.annotate(str(r.test_id), (r.target_x, r.target_y),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

        for i, (landing, dist) in enumerate(zip(r.landings, r.distances)):
            color = "green" if dist < success_threshold else "red"
            ax1.plot(landing[0], landing[1], "x", color=color, markersize=6, alpha=0.7)
            ax1.plot(
                [r.target_x, landing[0]], [r.target_y, landing[1]],
                "--", color=color, alpha=0.3, linewidth=0.8,
            )

    ax1.set_xlim(-0.6, 0.8)
    ax1.set_ylim(-0.2, 1.8)

    legend_elements = [
        mpatches.Patch(color="blue", alpha=0.7, label="Target"),
        plt.Line2D([0], [0], marker="x", color="green", linestyle="None", markersize=8, label="Success"),
        plt.Line2D([0], [0], marker="x", color="red", linestyle="None", markersize=8, label="Fail"),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=8, label="Robot"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Distance bar chart ─────────────────────────────────────
    ax2.set_title("Best Distance per Test", fontsize=12)
    ax2.set_xlabel("Test ID")
    ax2.set_ylabel("Distance to Target (m)")

    test_ids = [r.test_id for r in results]
    best_dists = [r.best_distance for r in results]
    colors = ["green" if r.passed else "red" for r in results]

    bars = ax2.bar(test_ids, best_dists, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=success_threshold, color="blue", linestyle="--", linewidth=1.5,
                label=f"Threshold ({success_threshold}m)")
    ax2.set_xticks(test_ids)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, max(best_dists) * 1.2 if best_dists else 1.0)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Validation plot saved to: {save_path}")

    if show:
        plt.show()


def _run_fast_validation():
    """Fast validation using DirectRLEnv (no IK, no ManagerBased overhead)."""
    from tasks.throwing_direct_env_cfg import ThrowingDirectEnvCfg, TABLE_Z as DTABLE_Z
    from tasks.throwing_direct_env import ThrowingDirectEnv
    from tasks.sb3_vec_env import DirectRLVecEnv

    headless = args_cli.headless

    cfg = ThrowingDirectEnvCfg()
    cfg.scene.num_envs = 1
    cfg.playing_arm_side = PLAYING_SIDE

    env = ThrowingDirectEnv(cfg=cfg)
    device = env.device
    env.reset()

    ckpt_path = args_cli.checkpoint
    is_zip = ckpt_path.endswith(".zip")
    model_type = args_cli.model_type
    if model_type == "auto":
        model_type = "sb3" if is_zip else "skrl"

    if model_type == "sb3":
        # ── SB3 model loading ─────────────────────────────────────────
        from stable_baselines3 import SAC

        env_wrapped = DirectRLVecEnv(env)
        model = SAC.load(ckpt_path, env=env_wrapped, seed=42)
        model_obs_dim = env.cfg.observation_space
        print(f"[INFO] Loaded SB3 SAC from: {ckpt_path}")
    else:
        # ── skrl model loading (legacy) ───────────────────────────────
        import gymnasium as gym
        from skrl.envs.wrappers.torch import wrap_env

        model_obs_dim = cfg.observation_space
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            policy_sd = ckpt["policy"]
            if isinstance(policy_sd, dict) and "net_container.0.weight" in policy_sd:
                model_obs_dim = policy_sd["net_container.0.weight"].shape[1]

        dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(model_obs_dim,), dtype=np.float32)
        dummy_act_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 0.05, 0.1], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        class _DummyEnv:
            def __init__(self):
                self.observation_space = dummy_obs_space
                self.action_space = dummy_act_space
                self.state_space = dummy_obs_space
                self.num_envs = 1
                self.num_agents = 1
                self.device = device
            def reset(self): return torch.zeros(1, model_obs_dim, device=device), {}
            def step(self, a): return torch.zeros(1, model_obs_dim, device=device), torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}
            def close(self): pass
            def render(self): pass

        dummy_wrapped = wrap_env(_DummyEnv(), wrapper="gymnasium")
        agent = load_sac_agent(dummy_wrapped, ckpt_path, device)
        print(f"[INFO] Model obs dim: {model_obs_dim}, Env obs dim: {cfg.observation_space}")

    _OBS_8D_FROM_10D = [0, 1, 2, 4, 5, 7, 8, 9]

    n_tests = min(args_cli.num_tests, get_test_count())
    results: List[ThrowResult] = []

    for test_idx in range(1, n_tests + 1):
        test_cfg = get_test_config(test_idx)
        if test_cfg is None:
            continue

        result = ThrowResult(
            test_id=test_cfg.test_id, test_name=test_cfg.name,
            target_x=test_cfg.target_x, target_y=test_cfg.target_y,
        )
        print(f"\n[Test {test_idx}/{n_tests}] {test_cfg.name} — target=({test_cfg.target_x:.2f}, {test_cfg.target_y:.2f})")

        for attempt in range(args_cli.attempts):
            target_obj = env.scene["target"]
            origin = env.scene.env_origins[0]
            tgt_z = DTABLE_Z + 0.001
            pos = torch.tensor([[test_cfg.target_x + origin[0].item(),
                                 test_cfg.target_y + origin[1].item(),
                                 tgt_z + origin[2].item()]], device=device)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
            target_obj.write_root_pose_to_sim(
                torch.cat([pos, quat], dim=-1), env_ids=torch.tensor([0], device=device)
            )

            obs_raw = env._get_observations()["policy"]
            if model_obs_dim < obs_raw.shape[-1]:
                obs_raw = obs_raw[:, _OBS_8D_FROM_10D]

            if model_type == "sb3":
                obs_np = obs_raw.cpu().numpy()
                action_np, _ = model.predict(obs_np, deterministic=True)
                action_np = action_np.flatten()
            else:
                with torch.no_grad():
                    act_out = agent.policy.act({"states": obs_raw}, role="policy")
                    action = act_out[0]
                action_np = action.cpu().numpy().flatten()

            action_clamped = torch.tensor(action_np, device=device).unsqueeze(0)
            action_clamped[:, 0].clamp_(-1.0, 1.0)
            action_clamped[:, 1].clamp_(-1.0, 1.0)
            action_clamped[:, 2].clamp_(0.05, 1.0)
            action_clamped[:, 3].clamp_(0.1, 1.0)

            env.step(action_clamped)

            distance = env._last_distances[0].item()
            landing = env._last_milk_pos[0].cpu().tolist()

            params_t = map_action_to_params(action_clamped, side=PLAYING_SIDE)
            result.distances.append(distance)
            result.actions.append(action_np.tolist())
            result.landings.append(landing[:2])

            status = "HIT" if distance < args_cli.success_threshold else "miss"
            print(
                f"  Attempt {attempt+1}/{args_cli.attempts}: "
                f"action=[{action_np[0]:+.2f},{action_np[1]:+.2f},{action_np[2]:.2f},{action_np[3]:.2f}] "
                f"→ ijv={params_t[0,0]:.2f} fjv={params_t[0,1]:.2f} "
                f"dist={distance:.3f}m [{status}]"
            )

        results.append(result)
        pass_str = "PASS" if result.passed else "FAIL"
        print(f"  Result: {pass_str} | best={result.best_distance:.3f}m | success={result.success_count}/{args_cli.attempts}")

    n_passed = sum(1 for r in results if r.passed)
    sr = n_passed / len(results) * 100 if results else 0
    avg_best = np.mean([r.best_distance for r in results]) if results else 0

    print(f"\n{'='*60}")
    print(f"  THROW VALIDATION RESULTS (DirectRLEnv — fast)")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  Test {r.test_id:2d} | {r.test_name:22s} | best={r.best_distance:.3f}m | "
              f"{r.success_count}/{args_cli.attempts} | {status}")
    print(f"{'='*60}")
    print(f"  Passed: {n_passed}/{len(results)} ({sr:.1f}%) | Avg best: {avg_best:.3f}m")
    print(f"{'='*60}")

    if not args_cli.no_plot:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_path = os.path.join(_PROJECT_ROOT, "logs", f"validation_results_fast_{ts}.png")
        plot_results(results, args_cli.success_threshold, plot_path, show=not headless)

    env.close()


def main():
    headless = args_cli.headless
    device_str = "cuda:0"

    print(f"\n{'='*60}")
    print(f"  THROW VALIDATION {'(DirectRLEnv — fast)' if args_cli.fast else ''}")
    print(f"  Checkpoint        : {args_cli.checkpoint}")
    print(f"  Tests             : {args_cli.num_tests}")
    print(f"  Attempts/test     : {args_cli.attempts}")
    print(f"  Success threshold : {args_cli.success_threshold}m")
    print(f"{'='*60}\n")

    if args_cli.fast:
        _run_fast_validation()
        return

    # ── Create environment ────────────────────────────────────────────────
    cfg = ThrowingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.ik_solver = "diffik"
    cfg.playing_arm_side = PLAYING_SIDE
    cfg.release_at_step = 0
    cfg.release_vel_threshold = float("inf")
    cfg.disable_attachment = True
    cfg.randomize_target = False
    cfg.target_x_range = (0.0, 0.0)
    cfg.target_y_range = (1.0, 1.0)
    cfg.__post_init__()

    if hasattr(cfg.actions.arm, "scale"):
        cfg.actions.arm.scale = 0.8
    if hasattr(cfg.actions.arm, "position_scale"):
        cfg.actions.arm.position_scale = 0.8
        cfg.actions.arm.orientation_scale = 0.8

    env = ThrowingEnv(cfg=cfg)
    device = env.device
    env.reset()
    env._holding[:] = False
    env._released[:] = False

    robot = env.scene["robot"]
    milk = env.scene["milk"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_PATTERNS)

    executor = ThrowPrimitiveExecutor(
        robot=robot,
        milk=milk,
        arm_joint_ids=arm_ids,
        gripper_set_fn=_set_gripper_state,
        ee_body_name=EE_BODY,
        side=PLAYING_SIDE,
        sim_dt=1.0 / 120.0,
        device=device,
    )

    # ── Load SAC agent via skrl Runner ────────────────────────────────────
    from tasks.throwing_primitive_env import ThrowingPrimitiveEnv
    from tasks.throwing_primitive_env_cfg import ThrowingPrimitiveEnvCfg
    from skrl.envs.wrappers.torch import wrap_env

    dummy_cfg = ThrowingPrimitiveEnvCfg()
    dummy_cfg.num_envs = 1
    dummy_cfg.playing_arm_side = PLAYING_SIDE

    import gymnasium as gym
    dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    dummy_act_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0, 0.05, 0.1], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )

    class DummyEnvForModel:
        """Minimal env stub for skrl Runner to read spaces."""
        def __init__(self):
            self.observation_space = dummy_obs_space
            self.action_space = dummy_act_space
            self.state_space = dummy_obs_space
            self.num_envs = 1
            self.num_agents = 1
            self.device = device
        def reset(self): return torch.zeros(1, 8, device=device), {}
        def step(self, a): return torch.zeros(1, 8, device=device), torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}
        def close(self): pass
        def render(self): pass

    dummy_env = DummyEnvForModel()
    dummy_wrapped = wrap_env(dummy_env, wrapper="gymnasium")
    agent = load_sac_agent(dummy_wrapped, args_cli.checkpoint, device)

    # ── Run validation ────────────────────────────────────────────────────
    n_tests = min(args_cli.num_tests, get_test_count())
    results: List[ThrowResult] = []

    for test_idx in range(1, n_tests + 1):
        test_cfg = get_test_config(test_idx)
        if test_cfg is None:
            continue

        result = ThrowResult(
            test_id=test_cfg.test_id,
            test_name=test_cfg.name,
            target_x=test_cfg.target_x,
            target_y=test_cfg.target_y,
        )

        print(f"\n[Test {test_idx}/{n_tests}] {test_cfg.name} — target=({test_cfg.target_x:.2f}, {test_cfg.target_y:.2f})")

        for attempt in range(args_cli.attempts):
            env.reset()
            env._holding[:] = False
            env._released[:] = False

            set_target_position(env, test_cfg.target_x, test_cfg.target_y, device)

            for _ in range(5):
                env.step(torch.zeros(1, 6, device=device))

            obs = compute_obs(env, device)

            with torch.no_grad():
                act_out = agent.policy.act({"states": obs}, role="policy")
                action = act_out[0]

            action_np = action.cpu().numpy().flatten()
            action_clamped = torch.tensor(action_np, device=device).unsqueeze(0)
            action_clamped[:, 0].clamp_(-1.0, 1.0)
            action_clamped[:, 1].clamp_(-1.0, 1.0)
            action_clamped[:, 2].clamp_(0.05, 1.0)
            action_clamped[:, 3].clamp_(0.1, 1.0)

            params_t = map_action_to_params(action_clamped, side=PLAYING_SIDE)
            params = ThrowPrimitiveParams(
                initial_joint_value=params_t[0, 0].item(),
                final_joint_value=params_t[0, 1].item(),
                releasing_time=params_t[0, 2].item(),
                duration=params_t[0, 3].item(),
            )

            distance = executor.execute_single(
                env, params, env_id=0, headless=headless, verbose=False,
            )

            origin = env.scene.env_origins[0].to(device)
            landing_pos = (milk.data.root_pos_w[0, :3] - origin).cpu().tolist()

            result.distances.append(distance)
            result.actions.append(action_np.tolist())
            result.landings.append(landing_pos[:2])

            status = "HIT" if distance < args_cli.success_threshold else "miss"
            print(
                f"  Attempt {attempt+1}/{args_cli.attempts}: "
                f"action=[{action_np[0]:+.2f},{action_np[1]:+.2f},{action_np[2]:.2f},{action_np[3]:.2f}] "
                f"→ params=[ijv={params.initial_joint_value:.2f} fjv={params.final_joint_value:.2f} "
                f"rel={params.releasing_time:.2f} dur={params.duration:.2f}] "
                f"dist={distance:.3f}m [{status}]"
            )

        results.append(result)
        pass_str = "PASS" if result.passed else "FAIL"
        print(f"  Result: {pass_str} | best={result.best_distance:.3f}m | success={result.success_count}/{args_cli.attempts}")

    # ── Summary ───────────────────────────────────────────────────────────
    n_passed = sum(1 for r in results if r.passed)
    sr = n_passed / len(results) * 100 if results else 0
    avg_best = np.mean([r.best_distance for r in results]) if results else 0
    all_dists = [d for r in results for d in r.distances]
    avg_all = np.mean(all_dists) if all_dists else 0

    print(f"\n{'='*60}")
    print(f"  THROW VALIDATION RESULTS")
    print(f"  Threshold: {args_cli.success_threshold}m | Attempts/test: {args_cli.attempts}")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  Test {r.test_id:2d} | {r.test_name:22s} | "
            f"target=({r.target_x:.2f},{r.target_y:.2f}) | "
            f"best={r.best_distance:.3f}m | {r.success_count}/{args_cli.attempts} | {status}"
        )
    print(f"{'='*60}")
    print(f"  Total tests  : {len(results)}")
    print(f"  Passed       : {n_passed}/{len(results)} ({sr:.1f}%)")
    print(f"  Avg best dist: {avg_best:.3f}m")
    print(f"  Avg all dist : {avg_all:.3f}m")
    print(f"{'='*60}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args_cli.no_plot:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_path = os.path.join(_PROJECT_ROOT, "logs", f"validation_results_{ts}.png")
        plot_results(results, args_cli.success_threshold, plot_path, show=not headless)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
