# DifferentialIK variant of train.py.
# Identical to train.py except line 216 below imports AsyncDualPlayDiffIKEnvCfg
# instead of AsyncDualPlayEnvCfg.  Run via hpc/train_small_diffik.slurm.
# To update shared training logic: edit train.py, then re-copy and re-apply this diff.
import isaaclab.app
from isaaclab.app import AppLauncher

import gc
import math
import os
import signal
import sys
import threading
import yaml
import argparse
from datetime import datetime
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SuppressAllOutput:
    """Context manager that silences both C-level (stdout/stderr fd) and Python-level output."""

    def __enter__(self):
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout = os.dup(self.stdout_fd)
        self.saved_stderr = os.dup(self.stderr_fd)
        self.devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(self.devnull, self.stdout_fd)
        os.dup2(self.devnull, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout, self.stdout_fd)
        os.dup2(self.saved_stderr, self.stderr_fd)
        os.close(self.saved_stdout)
        os.close(self.saved_stderr)
        os.close(self.devnull)
        if exc_type:
            print(f"Error occurred while suppressed: {exc_val}", file=sys.stderr)


def install_noise_filter():
    """
    Permanently redirect C-level stderr (fd 2) through a pipe and drop lines
    that match known Lula/URDF/carb noise patterns.  Everything else is passed
    through to the original stderr unchanged.  Call once at process start.
    """
    _DROP = (
        b"[Lula] Joint",
        b"Warning: link",
        b"urdf_parser",
        b"flat_black",
        b"IMemoryBudgetManager",
    )
    orig_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _worker():
        orig = os.fdopen(orig_fd, "wb", buffering=0)
        buf = b""
        with os.fdopen(read_fd, "rb", buffering=0) as src:
            while True:
                chunk = src.read(256)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_nl = line + b"\n"
                    if not any(p in line_nl for p in _DROP):
                        orig.write(line_nl)
                        orig.flush()
        if buf and not any(p in buf for p in _DROP):
            orig.write(buf)
            orig.flush()

    threading.Thread(target=_worker, daemon=True).start()


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Async Dual Play PPO")
    parser.add_argument("--exp_name", type=str, default="async_dual_play_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help="Override rollout steps per env (to prevent OOM on high num_envs)",
    )
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument(
        "--alice_decay_alpha",
        type=float,
        default=3.33,
        help=(
            "Steepness of Alice entropy decay. Uses normalized progress p=iter/max_iterations. "
            "alpha=3.33 → ~Test-2 shape (drops to ~0.10 by 75%% of run). "
            "alpha=1.5 → slower, suits long production runs."
        ),
    )
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument(
        "--max_alice_bob_ratio",
        type=int,
        default=None,
        help="Max consecutive Bob-only updates before forcing an Alice update. "
        "Auto-computed as ceil(bob_timesteps / alice_timesteps) * max(1, 64 // num_envs) "
        "if not specified.",
    )
    parser.add_argument(
        "--chkpt_alice",
        type=str,
        default=None,
        help="Path to Alice checkpoint (.pt) for resuming training",
    )
    parser.add_argument(
        "--chkpt_bob",
        type=str,
        default=None,
        help="Path to Bob checkpoint (.pt) for resuming training",
    )
    parser.add_argument(
        "--resume_iteration",
        type=int,
        default=0,
        help="Starting iteration count when resuming from checkpoint",
    )
    parser.add_argument(
        "--arm_config",
        type=str,
        default="default",
        choices=["default", "rotated"],
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of manipulated objects: 1=target only, 2=target+cube. Default=1.",
    )
    parser.add_argument(
        "--dummy_alice", action="store_true", help="Use dummy Alice wrapper"
    )
    parser.add_argument(
        "--diag_alice_exploration",
        action="store_true",
        help="Test 2: use DiagnosticAliceWrapper — real Alice policy with relaxed "
        "thresholds (alice_pos_req=0.02m, MIN_XY_DISP=0.03m) and verbose "
        "[AliceDisp] logging. Do NOT use for full training.",
    )
    parser.add_argument(
        "--test_bob_reward",
        action="store_true",
        help="Test: use DummyBobWrapper (teleports target→goal at step 50). "
        "Expected: Bob Rew > 0, SR > 0 from iter 1. "
        "Verifies sparse reward threshold and +5 completion bonus.",
    )
    parser.add_argument(
        "--test_abc_verbose",
        action="store_true",
        help="Test: print ABC demo content (goal shape, obs range) each time "
        "a trajectory is added. Verifies goal is appended to obs and "
        "filtering only stores Bob-failure episodes.",
    )
    parser.add_argument(
        "--test_hparams",
        action="store_true",
        help="Test: print hyperparameter audit comparing loaded config "
        "against paper Table 2 values and exit.",
    )
    parser.add_argument(
        "--dummy_goal_distance",
        action="store_true",
        help="Test: use DummyGoalDistanceWrapper (snaps both objects to stored "
        "goals at Bob step 30, then measures goal_distance()). "
        "Expected: pos_dist < 0.01 and rot_dist < 0.01 (printed as ✓). "
        "A distance ≈ env_origin magnitude (~2m) means double-subtraction bug.",
    )
    parser.add_argument(
        "--test_movement",
        action="store_true",
        help="Test: use DummyMovementWrapper (teleports target to [0.15,0.5,0.05] "
        "local during Alice's phase). "
        "Expected: measured_dist ≈ 0.362 and validate_goal says 'Valid Goal'. "
        "measured_dist ≈ 0.000 means stale initial_states.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-iteration timing profiler. Prints a breakdown table each "
        "iteration showing which section (env_step, alice_act, abc_buffer, …) "
        "dominates wall-clock time. Use with --max_iterations 3 for a quick audit.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    install_noise_filter()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    from isaaclab.envs import ManagerBasedRLEnv
    from asyncDualPlayPPO.tasks.async_dual_play_diffik import AsyncDualPlayDiffIKEnvCfg as AsyncDualPlayEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
    from asyncDualPlayPPO.tasks.utils.dummy_alice_wrapper import (
        DummyAliceWrapper,
        DummyBobWrapper,
        DummyGoalDistanceWrapper,
        DummyMovementWrapper,
        DiagnosticAliceWrapper,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer
    from asyncDualPlayPPO.utils.historical_pool import HistoricalPolicyPool
    from asyncDualPlayPPO.utils.profiler import TrainingProfiler

    # Import reward constants so train.py stays in sync with rewards.py.
    # Do not hardcode reward values here — change them in rewards.py instead.
    from asyncDualPlayPPO.tasks.utils.rewards import (
        ALICE_BOB_FAIL_REWARD,
        ALICE_BOB_SUCCESS_REWARD,
        ALICE_VALID_GOAL_BONUS,
        ALICE_INVALID_GOAL_PENALTY,
    )

    # --- Environment ---
    task_cfg_path = os.path.join(
        os.path.dirname(__file__), "cfg/task/AsyncDualPlay.yaml"
    )
    ppo_cfg_path = os.path.join(
        os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml"
    )
    ppo_cfg = load_cfg(ppo_cfg_path)
    task_cfg = load_cfg(task_cfg_path)

    alice_timesteps = task_cfg.get("alice_timesteps", 150)
    bob_timesteps = task_cfg.get("bob_timesteps", 200)
    max_goals_per_episode = task_cfg.get("max_goals_per_episode", 5)
    print(
        f"[Config] Episode structure: alice_timesteps={alice_timesteps}, "
        f"bob_timesteps={bob_timesteps}, max_goals={max_goals_per_episode} "
        f"(from AsyncDualPlay.yaml)"
    )

    if args.test_hparams:
        # Paper Table 2 reference values
        PAPER = {
            "gamma": 0.998,
            "lam": 0.95,
            "e_clip": 0.2,
            "entropy_coef": 0.01,
            "learning_rate": 3e-4,
            "mini_epochs": 3,  # "sample reuse (experience replay)"
            "critic_coef": 1.0,  # "value loss weight"
            "abc_coef": 0.5,  # "ABC loss weight β"
        }
        learn = ppo_cfg["params"].get("learn", {})
        actual = {
            "gamma": learn.get("gamma", "?"),
            "lam": learn.get("lam", "?"),
            "e_clip": learn.get("cliprange", "?"),
            "entropy_coef": learn.get("ent_coef", "?"),
            "learning_rate": learn.get("optim_stepsize", "?"),
            "mini_epochs": learn.get("noptepochs", "?"),
            "critic_coef": learn.get("value_loss_coef", "?"),
            "abc_coef": learn.get("abc_coef", "?"),
        }
        print("\n" + "=" * 60)
        print("HYPERPARAMETER AUDIT  (paper Table 2 vs loaded config)")
        print(f"{'Key':<18} {'Paper':>10} {'Loaded':>10}  {'OK?':>6}")
        print("-" * 60)
        all_ok = True
        for k, pv in PAPER.items():
            av = actual[k]
            ok = "✓" if av == pv else "✗ MISMATCH"
            if av != pv:
                all_ok = False
            print(f"  {k:<16} {str(pv):>10} {str(av):>10}  {ok}")
        print("=" * 60)
        print("All OK!" if all_ok else "Fix mismatches before large-scale training.")
        print("=" * 60 + "\n")
        import sys as _sys

        _sys.exit(0)

    if args.nsteps is not None:
        print(
            f"[Config] Overriding nsteps: {ppo_cfg['params']['learn']['nsteps']} -> {args.nsteps}"
        )
        ppo_cfg["params"]["learn"]["nsteps"] = args.nsteps

    # --- Multi-categorical action space config ---
    _pol_cfg = ppo_cfg["params"]["policy"]
    use_mc = _pol_cfg.get("use_multicategorical", False)
    num_cat_dims = _pol_cfg.get("num_cat_dims", 6)
    num_bins = _pol_cfg.get("num_bins", 11)
    if use_mc:
        print(
            f"[Config] Multi-categorical action space: {num_cat_dims} dims × {num_bins} bins "
            f"(physical scale set by env RMPFlow scale factor)"
        )

    def bins_to_env_action(
        bin_indices: "torch.Tensor", gripper_state: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Convert policy bin indices (N, 6) → 7D RMPFlow+gripper env action.

        XYZ: normalized = (bin - center) / center → [-1, 1]; env scale=0.05 → ±5cm/step.
        Rx, Ry: delta = (bin - center) / center * max_delta_rot (0.5 rad)

        Gripper (sticky): only the outer bins trigger a state change.
          Dead zone = center 3 bins (4/5/6) → keep previous gripper_state.
          This prevents random-policy spassing at the start of training.
          Threshold: bins 0-3 → close (-1), bins 4-6 → hold, bins 7-10 → open (+1)
        """
        center = (num_bins - 1) / 2.0  # 5.0 for 11 bins
        threshold = 2.0  # ±2 bins from center triggers change
        normalized = (bin_indices.float() - center) / center
        xyz = normalized[:, :3]  # [-1, 1]; RMPFlow scale=0.05 gives ±5cm/step

        # 2D wrist rotation (Rx, Ry) - scale 0.5 rad (~28 deg) at max bin
        max_delta_rot = 0.5
        rot_xy = normalized[:, 3:5] * max_delta_rot

        g_bin = bin_indices[:, 5].float()
        new_gs = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0  # bins 0-2  → close
        new_gs[g_bin > center + threshold - 1] = 1.0  # bins 8-10 → open
        # bins 3-7 → keep previous state

        # RMPFlow expects 6D (xyz, rpy) + 1D gripper. We zero out Rz.
        zeros1 = torch.zeros(bin_indices.shape[0], 1, device=bin_indices.device)
        return (
            torch.cat([xyz, rot_xy, zeros1, new_gs], dim=-1),
            new_gs,
        )

    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    if args.num_objects == 1:
        print("[Config] num_objects=1: removing cube from scene and observations.")
        env_cfg.scene.cube = None
        env_cfg.observations.alice_policy.cube_state = None
        env_cfg.observations.bob_policy.cube_state = None
        env_cfg.observations.bob_policy.cube_goal_state = None
        env_cfg.observations.bob_policy.cube_goal_distance = None

    if args.arm_config == "rotated":
        print("[Config] Rotated arm configuration: shoulder −90°")
        env_cfg.scene.robot.init_state.joint_pos["shoulder_pan_joint"] = 1.57
        env_cfg.scene.robot.init_state.joint_pos["elbow_joint"] = 2.356
        env_cfg.scene.robot.init_state.joint_pos["wrist_1_joint"] = -0.785
        env_cfg.scene.robot.init_state.joint_pos["wrist_2_joint"] = 1.57
        env_cfg.scene.robot.init_state.joint_pos["wrist_3_joint"] = 0

    print("Creating environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
    if args.test_bob_reward:
        print(
            "[Test] --test_bob_reward: using DummyBobWrapper (teleports target→goal at Bob step 50)."
        )
        print("[Test] Expected: Bob Rew > 0 and SR > 0 from iteration 1.")
        env = DummyBobWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            teleport_step=50,
            num_objects=args.num_objects,
        )
    elif args.dummy_goal_distance:
        print(
            "[Test] --dummy_goal_distance: using DummyGoalDistanceWrapper "
            "(snaps both objects to stored goals at Bob step 30)."
        )
        print("[Test] Expected: [DistCheck] pos < 0.01 and rot < 0.01 marked ✓.")
        env = DummyGoalDistanceWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            teleport_step=30,
            num_objects=args.num_objects,
        )
    elif args.test_movement:
        print(
            "[Test] --test_movement: using DummyMovementWrapper "
            "(teleports target to [0.15,0.5,0.05] local during Alice's phase)."
        )
        print("[Test] Expected: [MoveCheck] measured_dist≈0.362 marked ✓.")
        env = DummyMovementWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            num_objects=args.num_objects,
        )
    elif args.diag_alice_exploration:
        print(
            "[Test] --diag_alice_exploration: using DiagnosticAliceWrapper "
            "(real Alice policy, alice_pos_req=0.02m, MIN_XY_DISP=0.03m)."
        )
        print("[Test] Watch [AliceDisp] lines to diagnose valid-goal failure mode.")
        env = DiagnosticAliceWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            num_objects=args.num_objects,
        )
    elif args.dummy_alice:
        env = DummyAliceWrapper(
            env=base_env,
            device=base_env.device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            num_objects=args.num_objects,
        )
    else:
        env = AsyncDualPlayEnvWrapper(
            env=base_env,
            device=base_env.device,
            arm_config=args.arm_config,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            max_goals_per_episode=max_goals_per_episode,
            num_objects=args.num_objects,
            shaping_gamma=ppo_cfg["params"]["learn"].get("gamma", 0.99),
            shaping_coef=ppo_cfg["params"]["learn"].get("shaping_coef", 1.0),
            profiler=TrainingProfiler(
                enabled=getattr(args, "profile", False),
                device=str(base_env.device),
            ),
        )

    # --- Agents ---
    alice_ppo = PPO(
        vec_env=env,
        cfg_train=ppo_cfg["params"],
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/alice",
        asymmetric=False,
    )
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space = alice_ppo.observation_space

    # Override action space: policy operates on bin indices (4D), not the 7D env action
    if use_mc:
        import gymnasium as gym_mc

        _mc_space = gym_mc.spaces.Box(
            low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32
        )
        alice_ppo.action_space = _mc_space
        alice_ppo.desired_kl = None  # adaptive KL meaningless for discrete

    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape,
        alice_ppo.action_space.shape,
        alice_ppo.init_noise_std,
        alice_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    max_alice_steps = env.episode_manager.alice_timesteps + 10
    _rollout_len = env.episode_manager.alice_timesteps + env.episode_manager.bob_timesteps
    alice_storage_size = max(alice_ppo.num_transitions_per_env + max_alice_steps, _rollout_len + 10)
    alice_ppo.storage = alice_ppo.storage.__class__(
        alice_ppo.vec_env.num_envs,
        alice_storage_size,
        alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape,
        alice_ppo.action_space.shape,
        alice_ppo.device,
        "sequential",
    )
    alice_ppo.optimizer = torch.optim.Adam(
        alice_ppo.actor_critic.parameters(), lr=alice_ppo.learning_rate
    )

    import copy

    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"]["use_goal_encoder"] = True
    bob_cfg["policy"]["num_objects"] = args.num_objects

    bob_ppo = PPOABC(
        vec_env=env,
        cfg_train=bob_cfg,
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/bob",
        asymmetric=False,
    )
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space = bob_ppo.observation_space

    if use_mc:
        bob_ppo.action_space = _mc_space  # same 4D bin space as Alice
        bob_ppo.desired_kl = None

    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.init_noise_std,
        bob_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    # Scale goal_proj weights down at init: a large W_g saturates ReLUs before training starts.
    if (
        hasattr(bob_ppo.actor_critic, "_goal_proj")
        and bob_ppo.actor_critic._goal_proj is not None
    ):
        with torch.no_grad():
            bob_ppo.actor_critic._goal_proj.weight.mul_(0.01 / 0.5)
        print(
            f"  [Init] goal_proj scale reduced: ||W_g|| = {bob_ppo.actor_critic._goal_proj.weight.norm():.4f}"
        )

    bob_ppo.optimizer = torch.optim.Adam(
        bob_ppo.actor_critic.parameters(), lr=bob_ppo.learning_rate
    )

    # Re-initialize Bob's standard PPO storage for 56 dims.
    # Must hold the full bob_timesteps rollout (same pattern as Alice's oversized storage).
    max_bob_steps = env.episode_manager.bob_timesteps + 10
    bob_storage_size = max(bob_ppo.num_transitions_per_env + max_bob_steps, _rollout_len + 10)
    bob_ppo.storage = bob_ppo.storage.__class__(
        bob_ppo.vec_env.num_envs,
        bob_storage_size,
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.device,
        "sequential",
    )

    # ABC Buffer for Alice's successful demonstrations
    # actions_shape must match policy action dim (4D bin indices for MC, 7D for Gaussian)
    _abc_act_shape = (num_cat_dims,) if use_mc else env.action_space.shape
    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
        capacity=50000,
        obs_shape=env.bob_observation_space.shape,
        states_shape=env.bob_observation_space.shape,
        actions_shape=_abc_act_shape,
        device=env.device,
        traj_maxlen=ppo_cfg["params"]["learn"].get("abc_traj_maxlen", 500),
    )

    # LSTM hidden states (None for non-LSTM models — act_with_hidden handles both).
    _use_lstm = alice_ppo.actor_critic.use_lstm
    if _use_lstm:
        _lsz = alice_ppo.actor_critic.lstm_hidden_size
        alice_hidden = [
            torch.zeros(env.num_envs, _lsz, device=env.device),
            torch.zeros(env.num_envs, _lsz, device=env.device),
        ]
        bob_hidden = [
            torch.zeros(env.num_envs, _lsz, device=env.device),
            torch.zeros(env.num_envs, _lsz, device=env.device),
        ]
    else:
        alice_hidden = None
        bob_hidden = None

    # Sticky gripper state: starts open (+1), outer bins change it, center bins hold.
    alice_gripper_state = torch.ones(env.num_envs, 1, device=env.device)
    bob_gripper_state = torch.ones(env.num_envs, 1, device=env.device)

    # Historical policy pool: 20% of rollout envs use a past policy for stability.
    alice_pool = HistoricalPolicyPool(max_size=5)
    bob_pool = HistoricalPolicyPool(max_size=5)
    HIST_SAVE_INTERVAL = 50  # save snapshot every N bob_updates
    HIST_FRAC = 0.2  # fraction of envs using historical policy

    # Alice adaptive entropy params (read once; used in the per-iter entropy block).
    _learn = ppo_cfg["params"]["learn"]
    _alice_target_sr   = _learn.get("alice_entropy_target_sr", 0.5)
    _alice_entropy_lr  = _learn.get("alice_entropy_lr", 1e-3)
    _alice_entropy_min = _learn.get("alice_entropy_min", 0.05)
    _alice_entropy_max = _learn.get("alice_entropy_max", 1.0)
    print(
        f"[Config] Alice adaptive entropy: target_sr={_alice_target_sr}, "
        f"lr={_alice_entropy_lr}, min={_alice_entropy_min}, max={_alice_entropy_max}"
    )

    # --- Resume from checkpoint ---
    if args.chkpt_alice and os.path.isfile(args.chkpt_alice):
        alice_ppo.load(args.chkpt_alice)
        print(f"[Resume] Loaded Alice from {args.chkpt_alice}")
    if args.chkpt_bob and os.path.isfile(args.chkpt_bob):
        bob_ppo.load(args.chkpt_bob)
        print(f"[Resume] Loaded Bob from {args.chkpt_bob}")
        _abc_buf_path = os.path.join(os.path.dirname(args.chkpt_bob), "abc_buffer.pt")
        if os.path.isfile(_abc_buf_path):
            bob_ppo.abc_buffer.load(_abc_buf_path)
            print(
                f"[Resume] Loaded ABC buffer ({bob_ppo.abc_buffer.size} entries) from {_abc_buf_path}"
            )

        _ep_mgr_path = args.chkpt_bob.replace("model_", "episode_manager_")
        if os.path.isfile(_ep_mgr_path):
            ep_sd = torch.load(_ep_mgr_path, map_location=env.device)
            env.episode_manager.load_state_dict(ep_sd)
            print(f"[Resume] Loaded EpisodeManager state from {_ep_mgr_path}")
        else:
            print(
                f"[Resume] WARNING: EpisodeManager checkpoint not found at {_ep_mgr_path}. Envs will start fresh."
            )

    # --- Agents ---
    alice_updates = args.resume_iteration  # keep in sync with bob_updates so logs show global iters
    bob_updates = args.resume_iteration

    if args.num_envs < 32:
        print(
            f"[WARNING] num_envs={args.num_envs} is very low. "
            f"Paper uses 1856 envs (batch_size=4096). Recommend at least 32 for sparse rewards."
        )

    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")

    profiler = env._profiler if hasattr(env, "_profiler") and env._profiler is not None \
        else TrainingProfiler(enabled=getattr(args, "profile", False), device=env.device)

    rollout_length = ppo_cfg["params"]["learn"]["nsteps"] * args.num_envs
    alice_rew_buf = deque(maxlen=rollout_length)
    bob_rew_buf = deque(maxlen=rollout_length)
    bob_success_buf = deque(maxlen=200)
    bob_pos_err_buf = deque(maxlen=rollout_length)
    bob_rot_err_buf = deque(maxlen=rollout_length)

    best_bob_success_rate = -1.0
    last_alice_mean_rew = 0.0  # gating value for Bob's ABC loss warmup
    ema_alice_rew = 0.0        # 10-iter EMA of last_alice_mean_rew (τ=0.9)

    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    print(f"\n{'='*80}\nTRAINING RUN: {args.exp_name}\nLOG DIRECTORY: {run_dir}")
    print(f"  tensorboard --logdir {run_dir}\n{'='*80}\n")

    def perform_alice_update():
        """Run a PPO update for Alice after her rollout is complete."""
        nonlocal alice_updates, last_alice_mean_rew, ema_alice_rew

        # Storage is oversized to alice_storage_size; do NOT gate on num_transitions_per_env.
        if alice_ppo.storage.step == 0:
            return

        dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
        alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)

        loss_val, loss_surr = alice_ppo.update()
        alice_ppo.storage.clear()

        mean_alice_rew = np.mean(alice_rew_buf) if alice_rew_buf else 0.0
        last_alice_mean_rew = mean_alice_rew
        ema_alice_rew = 0.9 * ema_alice_rew + 0.1 * mean_alice_rew
        writer.add_scalar("Loss/Alice/Value", loss_val, alice_updates)
        writer.add_scalar("Loss/Alice/Surrogate", loss_surr, alice_updates)
        writer.add_scalar("Reward/Alice", mean_alice_rew, alice_updates)

        print(
            f"  [Alice Update {alice_updates}] Loss: {loss_surr:.4f} | Val: {loss_val:.4f} | Rew: {mean_alice_rew:.4f}",
            flush=True,
        )
        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(current_bob_obs):
        """Run a PPO + ABC update for Bob after his rollout is complete."""
        nonlocal bob_updates, best_bob_success_rate

        total_bob_transitions = bob_ppo.storage.step * env.num_envs
        if total_bob_transitions < bob_ppo.num_mini_batches:
            # Too few Bob transitions to fill even one mini-batch
            # (storage.step==0: no valid goals; or step>0 but Bob won instantly).
            # Still increment bob_updates to prevent deadlock — paper Algorithm 1
            # always advances both agents per training step.
            reason = (
                "no Bob transitions"
                if bob_ppo.storage.step == 0
                else f"only {total_bob_transitions} transitions < {bob_ppo.num_mini_batches} mini-batches"
            )
            print(
                f"  [Bob Update {bob_updates}] SKIPPED ({reason})",
                flush=True,
            )
            bob_rew_buf.clear()
            bob_pos_err_buf.clear()
            bob_rot_err_buf.clear()
            bob_updates += 1
            return

        with torch.no_grad():
            _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(current_bob_obs, None)

        # --- Goal Encoder Monitoring ---
        if bob_ppo.actor_critic.use_goal_encoder:
            with torch.no_grad():
                sample_obs = current_bob_obs[:8]
                # Bob obs layout — INTERLEAVED per object (matches _encode_obs reshape):
                #   1 obj (29D): robot(7) | s1(14) | g1(6) | d1(2)
                #   2 obj (51D): robot(7) | s1(14) | g1(6) | d1(2) | s2(14) | g2(6) | d2(2)
                # Per-object chunk size = 14+6+2 = 22D.
                _robot_dim = 7
                _obj_state_dim = 14
                _goal_dim = 6
                _dist_dim = 2
                _chunk = _obj_state_dim + _goal_dim + _dist_dim  # 22
                if args.num_objects == 1:
                    # s1 starts at 7, g1 starts at 7+14=21
                    s_t_batch = sample_obs[:, _robot_dim : _robot_dim + _goal_dim]
                    _goal_start = _robot_dim + _obj_state_dim
                    s_star_batch = sample_obs[:, _goal_start : _goal_start + _goal_dim]
                else:
                    # s1: [7:21], g1: [21:27]; s2: [29:43], g2: [43:49]
                    s_t_batch = torch.cat(
                        [sample_obs[:, 7:13], sample_obs[:, 29:35]], dim=-1
                    )
                    s_star_batch = torch.cat(
                        [sample_obs[:, 21:27], sample_obs[:, 43:49]], dim=-1
                    )
                g_sample = bob_ppo.actor_critic.goal_encoder(s_star_batch, s_t_batch)
                writer.add_scalar(
                    "GoalEncoder/embedding_norm",
                    g_sample.norm(dim=-1).mean().item(),
                    bob_updates,
                )
                writer.add_scalar(
                    "GoalEncoder/embedding_std",
                    g_sample.std(dim=-1).mean().item(),
                    bob_updates,
                )

        bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
        bob_ppo.current_learning_iteration = bob_updates
        loss_val, loss_surr, loss_abc, _ = bob_ppo.update(
            alice_mean_rew=last_alice_mean_rew
        )
        bob_ppo.storage.clear()

        mean_bob_rew = np.mean(bob_rew_buf) if bob_rew_buf else 0.0
        bob_success_rate = np.mean(bob_success_buf) if bob_success_buf else 0.0
        mean_pos_err = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        mean_rot_err = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0

        writer.add_scalar("Loss/Bob/Value", loss_val, bob_updates)
        writer.add_scalar("Loss/Bob/Surrogate", loss_surr, bob_updates)
        writer.add_scalar("Loss/Bob/ABC", loss_abc, bob_updates)
        writer.add_scalar("Reward/Bob", mean_bob_rew, bob_updates)
        writer.add_scalar("Metrics/Bob/SuccessRate", bob_success_rate, bob_updates)
        writer.add_scalar("Metrics/Bob/PosError", mean_pos_err, bob_updates)
        writer.add_scalar("Metrics/Bob/RotError", mean_rot_err, bob_updates)

        print(
            f"  [Bob Update {bob_updates}] Loss: {loss_surr:.4f} | Val: {loss_val:.4f} | Rew: {mean_bob_rew:.4f} | ABC: {loss_abc:.4f} | SR: {bob_success_rate:.4f} | ABCCoef: {bob_ppo.abc_coef:.4f}",
            flush=True,
        )
        # _abc_phase is set later in the iter (entropy/ABC block); log it there.

        if args.save_interval > 0 and (bob_updates + 1) % args.save_interval == 0:
            bob_ppo.save(os.path.join(bob_ppo.log_dir, f"model_{bob_updates+1}.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{bob_updates+1}.pt"))
            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
            torch.save(
                env.episode_manager.state_dict(),
                os.path.join(bob_ppo.log_dir, f"episode_manager_{bob_updates+1}.pt"),
            )

        if bob_success_rate > best_bob_success_rate:
            best_bob_success_rate = bob_success_rate
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_best.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))
            torch.save(
                env.episode_manager.state_dict(),
                os.path.join(bob_ppo.log_dir, "episode_manager_best.pt"),
            )

        bob_rew_buf.clear()
        bob_pos_err_buf.clear()
        bob_rot_err_buf.clear()

        bob_updates += 1


    # --- Graceful shutdown on SIGTERM (sent by SLURM at the hard time limit) ---
    _shutdown_requested = False

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        print("[INFO] SIGTERM received — will save emergency checkpoint after current iteration.",
              flush=True)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("Initializing environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        obs = env.reset()[0]
    print("Environment initialized. Starting training loop...")

    target_alice_timesteps = env.episode_manager.alice_timesteps

    while bob_updates < args.max_iterations:

        profiler.start_iteration()

        # --- 0. SETUP: reset per-iteration stats, LSTM hidden states, snapshot policies ---
        if hasattr(env, "reset_iter_stats"):
            env.reset_iter_stats()

        if alice_hidden is not None:
            alice_hidden[0].zero_()
            alice_hidden[1].zero_()
        if bob_hidden is not None:
            bob_hidden[0].zero_()
            bob_hidden[1].zero_()

        if bob_updates > 0 and bob_updates % HIST_SAVE_INTERVAL == 0:
            alice_pool.add(alice_ppo.actor_critic)
            bob_pool.add(bob_ppo.actor_critic)
            print(
                f"  [HistPool] Saved snapshot at iter {bob_updates} "
                f"(alice pool={alice_pool.size}, bob pool={bob_pool.size})",
                flush=True,
            )

        # --- 1. UNIFIED ASYNCHRONOUS ROLLOUT PHASE ---
        alice_ppo.storage.clear()
        bob_ppo.storage.clear()

        iter_sr_counts = [0, 0]  # [attempted, succeeded]

        with profiler.section("obs_compute"):
            obs_dict = env.env.observation_manager.compute()
            obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_alice_obs = obs[:, : env.alice_obs_dim]
        current_bob_obs = obs[:, env.alice_obs_dim :]

        # Pre-allocate Alice trajectory buffers for ABC (Max: alice_timesteps)
        _a_max_steps = env.episode_manager.alice_timesteps
        _a_pdim = num_cat_dims if use_mc else env.action_space.shape[0]
        _b_pdim = num_cat_dims if use_mc else env.action_space.shape[0]

        alice_traj_obs = torch.zeros((env.num_envs, _a_max_steps, env.alice_obs_dim), device=env.device)
        alice_traj_act = torch.zeros((env.num_envs, _a_max_steps, _a_pdim), device=env.device)
        alice_traj_len = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        hist_alice = alice_pool.sample_policy(alice_ppo.actor_critic, env.device) if alice_pool.size > 0 else None
        hist_bob = bob_pool.sample_policy(bob_ppo.actor_critic, env.device) if bob_pool.size > 0 else None

        rollout_length = env.episode_manager.alice_timesteps + env.episode_manager.bob_timesteps

        for t in range(rollout_length):
            is_alice = env.episode_manager.is_alice_phase()
            is_bob = env.episode_manager.is_bob_phase()
            alice_indices = torch.where(is_alice)[0]
            bob_indices = torch.where(is_bob)[0]

            # -----------------------------------------------------------------
            # COMPUTE ALICE ACTIONS
            # -----------------------------------------------------------------
            profiler.mark_start("alice_act")
            a_acts_active = torch.zeros((len(alice_indices), _a_pdim), device=env.device)
            a_logprob_active = torch.zeros(len(alice_indices), device=env.device)
            a_val_active = torch.zeros(len(alice_indices), 1, device=env.device)
            a_mu_active = torch.zeros_like(a_acts_active)
            a_sigma_active = torch.zeros_like(a_acts_active)

            if len(alice_indices) > 0:
                hist_ids, curr_ids = alice_pool.sample_env_subset(alice_indices, frac=HIST_FRAC)
                with torch.no_grad():
                    # Current Alice (majority)
                    h_in = ((alice_hidden[0][curr_ids], alice_hidden[1][curr_ids]) if alice_hidden else None)
                    (a_acts_curr, a_logprob_curr, a_val_curr, a_mu_curr, a_sigma_curr, new_h) = alice_ppo.actor_critic.act_with_hidden(current_alice_obs[curr_ids], None, h_in)
                    if alice_hidden and new_h is not None:
                        alice_hidden[0][curr_ids] = new_h[0]
                        alice_hidden[1][curr_ids] = new_h[1]

                    # Historical Alice (minority)
                    if len(hist_ids) > 0 and hist_alice is not None:
                        (a_acts_hist, a_logprob_hist, a_val_hist, a_mu_hist, a_sigma_hist, _) = hist_alice.act_with_hidden(current_alice_obs[hist_ids], None, None)
                    else:
                        hist_ids = torch.tensor([], dtype=torch.long, device=env.device)
                        a_acts_hist = a_logprob_hist = a_val_hist = a_mu_hist = a_sigma_hist = None

                curr_local = torch.searchsorted(alice_indices, curr_ids)
                a_acts_active[curr_local] = a_acts_curr
                a_logprob_active[curr_local] = a_logprob_curr
                a_val_active[curr_local] = a_val_curr
                a_mu_active[curr_local] = a_mu_curr
                a_sigma_active[curr_local] = a_sigma_curr

                if len(hist_ids) > 0 and a_acts_hist is not None:
                    hist_local = torch.searchsorted(alice_indices, hist_ids)
                    a_acts_active[hist_local] = a_acts_hist
                    a_logprob_active[hist_local] = a_logprob_hist
                    a_val_active[hist_local] = a_val_hist
                    a_mu_active[hist_local] = a_mu_hist
                    a_sigma_active[hist_local] = a_sigma_hist

            profiler.mark_stop("alice_act")
            # -----------------------------------------------------------------
            # COMPUTE BOB ACTIONS
            # -----------------------------------------------------------------
            profiler.mark_start("bob_act")
            b_acts_active = torch.zeros((len(bob_indices), _b_pdim), device=env.device)
            b_logprob_active = torch.zeros(len(bob_indices), device=env.device)
            b_val_active = torch.zeros(len(bob_indices), 1, device=env.device)
            b_mu_active = torch.zeros_like(b_acts_active)
            b_sigma_active = torch.zeros_like(b_acts_active)

            if len(bob_indices) > 0:
                hist_bids, curr_bids = bob_pool.sample_env_subset(bob_indices, frac=HIST_FRAC)
                with torch.no_grad():
                    # Current Bob
                    h_in = ((bob_hidden[0][curr_bids], bob_hidden[1][curr_bids]) if bob_hidden else None)
                    (b_acts_curr, b_lp_curr, b_val_curr, b_mu_curr, b_sig_curr, new_bh) = bob_ppo.actor_critic.act_with_hidden(current_bob_obs[curr_bids], None, h_in)
                    if bob_hidden and new_bh is not None:
                        bob_hidden[0][curr_bids] = new_bh[0]
                        bob_hidden[1][curr_bids] = new_bh[1]

                    # Historical Bob
                    if len(hist_bids) > 0 and hist_bob is not None:
                        (b_acts_hist, b_lp_hist, b_val_hist, b_mu_hist, b_sig_hist, _) = hist_bob.act_with_hidden(current_bob_obs[hist_bids], None, None)
                    else:
                        hist_bids = torch.tensor([], dtype=torch.long, device=env.device)
                        b_acts_hist = b_lp_hist = b_val_hist = b_mu_hist = b_sig_hist = None

                curr_bloc = torch.searchsorted(bob_indices, curr_bids)
                b_acts_active[curr_bloc] = b_acts_curr
                b_logprob_active[curr_bloc] = b_lp_curr
                b_val_active[curr_bloc] = b_val_curr
                b_mu_active[curr_bloc] = b_mu_curr
                b_sigma_active[curr_bloc] = b_sig_curr

                if len(hist_bids) > 0 and b_acts_hist is not None:
                    hist_bloc = torch.searchsorted(bob_indices, hist_bids)
                    b_acts_active[hist_bloc] = b_acts_hist
                    b_logprob_active[hist_bloc] = b_lp_hist
                    b_val_active[hist_bloc] = b_val_hist
                    b_mu_active[hist_bloc] = b_mu_hist
                    b_sigma_active[hist_bloc] = b_sig_hist

            profiler.mark_stop("bob_act")
            # -----------------------------------------------------------------
            # COMBINE ACTIONS & STEP ENVIRONMENT
            # -----------------------------------------------------------------
            a_policy = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_policy[alice_indices] = a_acts_active
            
            b_policy = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_policy[bob_indices] = b_acts_active

            # Store Alice trajectory for ABC
            alice_step_raw = env.episode_manager.phase_step[alice_indices] - 1
            valid_t = alice_step_raw < _a_max_steps
            active_alice = alice_indices[valid_t]
            active_steps = alice_step_raw[valid_t]

            phase_start = alice_step_raw == 0
            if phase_start.any():
                alice_traj_len[alice_indices[phase_start]] = 0

            if len(active_alice) > 0:
                alice_traj_obs[active_alice, active_steps] = current_alice_obs[active_alice].clone()
                alice_traj_act[active_alice, active_steps] = a_acts_active[valid_t].clone()
                new_len = (active_steps + 1).to(alice_traj_len.dtype)
                alice_traj_len[active_alice] = torch.max(alice_traj_len[active_alice], new_len)

            if use_mc:
                env_full = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
                a_act_7d, new_ags = bins_to_env_action(a_acts_active, alice_gripper_state[alice_indices])
                b_act_7d, new_bgs = bins_to_env_action(b_acts_active, bob_gripper_state[bob_indices])
                env_full[alice_indices] = a_act_7d
                env_full[bob_indices] = b_act_7d
                alice_gripper_state[alice_indices] = new_ags
                bob_gripper_state[bob_indices] = new_bgs
            else:
                env_full = a_policy + b_policy  # Masks are disjoint

            with profiler.section("env_step"):
                obs_full, rewards, dones, truncated, extras = env.step(env_full)

            # Log Bob's per-step rewards (sparse + shaping) for the Rew column in CSV
            if len(bob_indices) > 0:
                bob_rew_buf.extend(rewards[bob_indices].cpu().numpy().tolist())

            # Count Bob completions
            ep_info = extras.get("episode_manager", {})
            if ep_info:
                finished_bob = torch.where(ep_info["bob_done_this_step"])[0]
                if len(finished_bob) > 0:
                    iter_sr_counts[0] += len(finished_bob)
                    iter_sr_counts[1] += int(ep_info["bob_success_this_step"][finished_bob].sum().item())

            # -----------------------------------------------------------------
            # ALICE STORAGE & UPDATES
            # -----------------------------------------------------------------
            a_lp_full = torch.zeros(env.num_envs, device=env.device)
            a_val_full = torch.zeros(env.num_envs, 1, device=env.device)
            a_mu_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_sigma_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_lp_full[alice_indices] = a_logprob_active
            a_val_full[alice_indices] = a_val_active
            a_mu_full[alice_indices] = a_mu_active
            a_sigma_full[alice_indices] = a_sigma_active

            if alice_hidden is not None:
                done_alice = alice_indices[dones[alice_indices]]
                if len(done_alice) > 0:
                    alice_hidden[0][done_alice] = 0.0
                    alice_hidden[1][done_alice] = 0.0
            alice_gripper_state[alice_indices[dones[alice_indices]]] = 1.0

            a_masks = torch.zeros(env.num_envs, 1, device=env.device)
            a_masks[alice_indices[~dones[alice_indices]]] = 1.0

            next_alice_obs = obs_full[:, : env.alice_obs_dim]
            with profiler.section("alice_store"):
                alice_ppo.storage.add_transitions(
                    current_alice_obs,
                    next_alice_obs,
                    a_policy,
                    rewards,  # per-step: base penalties + dense shaping; terminal outcome backfilled later
                    dones,
                    a_val_full,
                    a_lp_full,
                    a_mu_full,
                    a_sigma_full,
                    a_masks,
                )
            current_alice_obs = next_alice_obs

            # -----------------------------------------------------------------
            # BOB STORAGE & UPDATES
            # -----------------------------------------------------------------
            b_lp_full = torch.zeros(env.num_envs, 1, device=env.device)
            b_val_full = torch.zeros(env.num_envs, 1, device=env.device)
            b_mu_full = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_sigma_full = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_lp_full[bob_indices] = b_logprob_active.unsqueeze(1)
            b_val_full[bob_indices] = b_val_active
            b_mu_full[bob_indices] = b_mu_active
            b_sigma_full[bob_indices] = b_sigma_active

            bob_done_this_step = ep_info.get("bob_done_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            ended_for_bob = dones | bob_done_this_step
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob[bob_indices]]] = 1.0

            if bob_hidden is not None:
                done_bob = bob_indices[dones[bob_indices]]
                if len(done_bob) > 0:
                    bob_hidden[0][done_bob] = 0.0
                    bob_hidden[1][done_bob] = 0.0
            bob_gripper_state[bob_indices[dones[bob_indices]]] = 1.0

            next_bob_obs = obs_full[:, env.alice_obs_dim :]
            with profiler.section("bob_store"):
                bob_ppo.storage.add_transitions(
                    current_bob_obs,
                    next_bob_obs,
                    b_policy,
                    rewards,
                    dones,
                    b_val_full,
                    b_lp_full,
                    b_mu_full,
                    b_sigma_full,
                    b_masks,
                )
            current_bob_obs = next_bob_obs

            # -----------------------------------------------------------------
            # ALICE OUTCOME & ABC BUFFER
            # -----------------------------------------------------------------
            if ep_info:
                bob_dones_now = ep_info.get("bob_done_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                alice_dones_now = extras.get("alice_failed_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                alice_rewards_now = extras.get("alice_total_reward", torch.zeros(env.num_envs, device=env.device))
                goal_valid = ep_info.get("goal_valid", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                bob_success = ep_info.get("bob_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                
                profiler.mark_start("reward_backfill")
                # Apply Alice outcome to storage NOW
                if bob_dones_now.any() or alice_dones_now.any() or (alice_rewards_now != 0).any():
                    filled = alice_ppo.storage.step
                    if filled > 0:
                        masks = alice_ppo.storage.masks[:filled, :, 0]
                        row_idx = torch.arange(filled, device=env.device).unsqueeze(1).expand_as(masks)
                        last_valid_rows = torch.where(masks.bool(), row_idx, torch.tensor(-1, device=env.device)).max(dim=0).values

                        has_valid = last_valid_rows >= 0

                        rewarded_envs = torch.where(alice_rewards_now != 0)[0]
                        valid_rewarded = rewarded_envs[has_valid[rewarded_envs]]
                        if len(valid_rewarded) > 0:
                            rows = last_valid_rows[valid_rewarded]
                            alice_ppo.storage.rewards[rows, valid_rewarded, 0] += alice_rewards_now[valid_rewarded]
                            alice_rew_buf.extend(alice_rewards_now[valid_rewarded].cpu().numpy().tolist())
                profiler.mark_stop("reward_backfill")

                # ABC BATCH: Add demos for Bob failures *that just finished this step*
                profiler.mark_start("abc_buffer")
                just_failed_bob = bob_dones_now & (~bob_success) & goal_valid
                valid_ids = torch.where(just_failed_bob)[0]

                min_demo_steps = max(10, env.episode_manager.alice_timesteps // 2)
                for env_id in valid_ids:
                    eid = env_id.item()
                    t_len = alice_traj_len[eid].item()
                    if t_len < min_demo_steps:
                        continue

                    traj_o = alice_traj_obs[eid, :t_len]
                    traj_a = alice_traj_act[eid, :t_len]
                    g = ep_info["goal_states"][eid].unsqueeze(0).expand(t_len, -1)

                    bc_obs = env.construct_bob_observation(traj_o, g)
                    with torch.no_grad():
                        old_lp, _, _, _, _ = bob_ppo.actor_critic.evaluate(bc_obs, None, traj_a)

                    bob_ppo.abc_buffer.add_trajectory(
                        bc_obs, bc_obs, traj_a,
                        torch.zeros(t_len, device=env.device),
                        torch.zeros(t_len, device=env.device).byte(),
                        torch.zeros(t_len, device=env.device),
                        old_lp.view(-1, 1),
                        torch.zeros_like(traj_a), torch.zeros_like(traj_a),
                        torch.zeros(t_len, 1, device=env.device),
                        torch.zeros(t_len, 1, device=env.device)
                    )
                profiler.mark_stop("abc_buffer")

        current_sr = iter_sr_counts[1] / max(1, iter_sr_counts[0])
        bob_success_buf.append(current_sr)

        # --- ENTROPY / LR / ABC CONTROLLERS ---
        # Placed here so bob_success_buf contains the just-computed SR before
        # the controllers read it. (Previously these ran at the loop top, after
        # the per-iteration clear, so the buffer was always empty → SR=0.0.)

        # Alice entropy: two-phase schedule.
        # Phase 1 (iter < 250): exponential decay 1.0 → 0.10.
        # Phase 2 (iter ≥ 250): one-sided proportional controller on Bob's SR.
        #   Entropy is raised when Bob exceeds the target SR (goals too easy),
        #   but never lowered — preventing the controller from dragging entropy
        #   to the floor while Bob is still learning.
        _ent_p = min(1.0, bob_updates / min(args.max_iterations, 250))
        if _ent_p < 1.0:
            alice_ppo.entropy_coef = 0.10 + 0.90 * math.exp(
                -args.alice_decay_alpha * _ent_p
            )
            _ent_phase = "decay"
        else:
            _bob_sr_now = np.mean(bob_success_buf) if bob_success_buf else 0.0
            _sr_error = _bob_sr_now - _alice_target_sr
            if _sr_error > 0:  # one-sided: only raise entropy, never lower
                alice_ppo.entropy_coef = float(np.clip(
                    alice_ppo.entropy_coef + _alice_entropy_lr * _sr_error,
                    _alice_entropy_min,
                    _alice_entropy_max,
                ))
            _ent_phase = f"adaptive sr_err={_sr_error:+.3f}"
            writer.add_scalar("Alice/EntropySRError", _sr_error, bob_updates)
        writer.add_scalar("Alice/EntropyCoef", alice_ppo.entropy_coef, bob_updates)
        print(f"  [Alice] Entropy Coef: {alice_ppo.entropy_coef:.4f} ({_ent_phase})", flush=True)

        # Alice LR cosine decay: lr(t) = lr_min + 0.5*(lr_max−lr_min)*(1+cos(π·t/T)).
        _alice_lr_max = alice_ppo.learning_rate
        _alice_lr_min = ppo_cfg["params"]["learn"].get("alice_lr_min", 5e-5)
        _lr_p = min(1.0, bob_updates / args.max_iterations)
        _alice_lr = _alice_lr_min + 0.5 * (_alice_lr_max - _alice_lr_min) * (
            1.0 + math.cos(math.pi * _lr_p)
        )
        for pg in alice_ppo.optimizer.param_groups:
            pg["lr"] = _alice_lr
        writer.add_scalar("Alice/LearningRate", _alice_lr, bob_updates)

        # ABC coefficient: two-phase schedule.
        # Phase 1 (iter < abc_anneal_iters): linear decay abc_coef → 0.0.
        # Phase 2 (iter ≥ abc_anneal_iters): inverse proportional controller.
        #   target = abc_coef_start * (1 - bob_sr): high when Bob fails, low when Bob succeeds.
        _abc_coef_start   = ppo_cfg["params"]["learn"].get("abc_coef", 0.5)
        _abc_coef_end     = ppo_cfg["params"]["learn"].get("abc_coef_end", 0.0)
        _abc_anneal_iters = ppo_cfg["params"]["learn"].get("abc_anneal_iters", 0)
        _abc_coef_ema     = ppo_cfg["params"]["learn"].get("abc_coef_ema", 0.95)

        if _abc_anneal_iters > 0 and bob_updates < _abc_anneal_iters:
            _abc_p = bob_updates / _abc_anneal_iters
            bob_ppo.abc_coef = _abc_coef_start + (_abc_coef_end - _abc_coef_start) * _abc_p
            _abc_phase = "anneal"
        else:
            _bob_sr_for_abc = np.mean(bob_success_buf) if bob_success_buf else 0.0
            _target_abc = float(np.clip(
                _abc_coef_start * (1.0 - _bob_sr_for_abc),
                _abc_coef_end,
                _abc_coef_start,
            ))
            bob_ppo.abc_coef = _abc_coef_ema * bob_ppo.abc_coef + (1.0 - _abc_coef_ema) * _target_abc
            _abc_phase = f"inverse sr={_bob_sr_for_abc:.3f} tgt={_target_abc:.3f}"
        writer.add_scalar("Bob/ABCCoef", bob_ppo.abc_coef, bob_updates)
        print(f"  [Bob] ABCCoef: {bob_ppo.abc_coef:.4f} ({_abc_phase})", flush=True)

        with profiler.section("alice_update"):
            perform_alice_update()
        with profiler.section("bob_update"):
            perform_bob_update(current_bob_obs)

        # --- Emergency checkpoint on SIGTERM (SLURM hard kill) ---
        if _shutdown_requested:
            _ckpt_iter = bob_updates  # already incremented inside perform_bob_update
            print(f"[INFO] Emergency checkpoint at iteration {_ckpt_iter}", flush=True)
            bob_ppo.save(os.path.join(bob_ppo.log_dir, f"model_{_ckpt_iter}.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{_ckpt_iter}.pt"))
            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
            torch.save(
                env.episode_manager.state_dict(),
                os.path.join(bob_ppo.log_dir, f"episode_manager_{_ckpt_iter}.pt"),
            )
            print(f"[INFO] Emergency checkpoint saved — exiting cleanly.", flush=True)
            break

        # --- Iteration aggregate summary ---
        _stats = env.get_iter_stats() if hasattr(env, "get_iter_stats") else {}
        _term_str = (
            "  ".join(
                f"{k}={v}" for k, v in sorted(_stats.get("terminations", {}).items())
            )
            or "none"
        )

        # TensorBoard diagnostics for Tests 2 & 3
        _valid_goals = _stats.get("valid_goals", 0)
        writer.add_scalar("Metrics/Alice/ValidGoals", _valid_goals, bob_updates)
        _abc_buf_size = bob_ppo.abc_buffer.size
        writer.add_scalar("Metrics/ABC/BufferSize", _abc_buf_size, bob_updates)
        _abc_warm = 1.0 if ema_alice_rew >= bob_ppo.abc_warmup_threshold else 0.0
        writer.add_scalar("Metrics/ABC/IsWarm", _abc_warm, bob_updates)
        writer.add_scalar("Metrics/Alice/EMAReward", ema_alice_rew, bob_updates)

        print(
            f"[Iter {bob_updates}] SR={current_sr:.2f} | "
            f"Goals valid={_valid_goals} invalid={_stats.get('invalid_goals', 0)} | "
            f"Bob succ={_stats.get('bob_successes', 0)} fail={_stats.get('bob_failures', 0)} | "
            f"Terminations: {_term_str} | "
            f"ABC buf: {bob_ppo.abc_buffer.size} | "
            f"ABC warm: {'YES' if _abc_warm else 'NO'}",
            flush=True,
        )
        _alice_total = _stats.get("alice_total", 0)
        if _alice_total > 0:
            _avg_3d  = _stats["alice_disp_3d_sum"] / _alice_total
            _avg_xy  = _stats["alice_disp_xy_sum"] / _alice_total
            _max_xy  = _stats["alice_disp_xy_max"]
            _avg_y   = _stats.get("alice_disp_y_sum", 0.0) / _alice_total
            _avg_z   = _stats.get("alice_disp_z_sum", 0.0) / _alice_total
            _not_mvd = _stats["alice_not_moved"]
            _pos_req = getattr(env, "_ALICE_POS_REQ", 0.05)
            print(
                f"  [AliceDisp] {_valid_goals}/{_alice_total} valid | "
                f"avg 3D={_avg_3d:.3f}m  avg XY={_avg_xy:.3f}m  max XY={_max_xy:.3f}m  avg Y={_avg_y:.3f}m  avg Z={_avg_z:.3f}m | "
                f"not-moved(≤{_pos_req:.2f}m): {_not_mvd}/{_alice_total}",
                flush=True,
            )
        profiler.end_iteration(bob_updates)

    profiler.print_summary()
    alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_final.pt"))
    bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_final.pt"))
    torch.save(
        env.episode_manager.state_dict(),
        os.path.join(bob_ppo.log_dir, "episode_manager_final.pt"),
    )
    print("  ✓ Saved final models")
    writer.close()

    print(f"\n{'='*80}\nTRAINING COMPLETE ({bob_updates} iterations)\n{'='*80}")
    print(f"To resume:\n  python train.py --exp_name {args.exp_name} \\")
    print(f"    --chkpt_alice runs/{args.exp_name}/alice/model_final.pt \\")
    print(f"    --chkpt_bob   runs/{args.exp_name}/bob/model_final.pt \\")
    print(f"    --resume_iteration {bob_updates}\n{'='*80}\n")


if __name__ == "__main__":
    main()
