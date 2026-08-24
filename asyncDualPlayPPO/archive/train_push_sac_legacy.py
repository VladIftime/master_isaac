"""
Push-SAC training script.

Single-agent SAC with LSTM using push primitive macro-actions, powered by skrl.
Mirrors train_push.py structure for the simulation setup, uses skrl's SAC_RNN
for the RL algorithm.

Action space: 4D continuous [-1, 1] (Xs, Ys, length, theta) or (r, phi, length, theta).
Architecture: Actor LSTM + Twin Q-networks (no LSTM).

Run locally:
  python -m asyncDualPlayPPO.train_push_sac --num_envs 16 --max_iterations 500 --exp_name push_sac_test
"""

try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print(
        "\n[ERROR] cuRobo not found. Install it in the active venv before training.\n"
    )
    import sys
    sys.exit(1)

import torch
import torch._dynamo    # noqa: F401
import torch._C         # noqa: F401
import torch.optim      # noqa: F401

from isaaclab.app import AppLauncher

import os
import signal
import sys
import yaml
import argparse
import math
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SuppressAllOutput:
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


def main():
    parser = argparse.ArgumentParser(description="Push-SAC Training (skrl)")
    parser.add_argument("--exp_name", type=str, default="push_sac_abs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--chkpt", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write terminal output to this file as well")
    parser.add_argument("--with_distractor", action="store_true",
                        help="Spawn a random cube/cylinder as clutter (no goal)")
    parser.add_argument("--rel-obs", action="store_true", dest="rel_obs",
                        help="Append object-relative goal delta (dx, dy) to observation (28D→30D).")
    parser.add_argument("--rel-act", action="store_true", dest="rel_act",
                        help="Decode push approach as object-relative (r, φ) instead of absolute (Xs, Ys).")
    parser.add_argument("--pushes_per_iter", type=int, default=5,
                        help="Pushes collected per logging iteration")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="SAC mini-batch size")
    parser.add_argument("--buffer_size", type=int, default=200000,
                        help="Replay buffer capacity")
    parser.add_argument("--warmup_pushes", type=int, default=500,
                        help="Random pushes before training starts")
    parser.add_argument("--gradient_steps", type=int, default=1,
                        help="Gradient steps per push step")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for actor and critic")
    parser.add_argument("--sequence_length", type=int, default=8,
                        help="LSTM sequence length for replay sampling")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    _log_fh = None
    if args.log_file:
        _log_fh = open(args.log_file, "a", buffering=1)
        print(f"[Init] Logging to {args.log_file}", flush=True)

    def _pr(msg: str = "", end: str = "\n"):
        sys.stdout.write(msg + end)
        sys.stdout.flush()
        if _log_fh:
            _log_fh.write(msg + end)
            _log_fh.flush()

    import torch
    import numpy as np
    import copy
    import gymnasium as gym_spaces
    from torch.utils.tensorboard import SummaryWriter

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.assets import RigidObjectCfg
    from asyncDualPlayPPO.tasks.utils.push_primitive_1arm_env import ISAACLAB_DUAL_ARM_EXT_DIR

    from asyncDualPlayPPO.tasks.utils.reach_dual_arm_diffik_env_cfg import spawn_random_block
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper, _euler_to_quat
    from asyncDualPlayPPO.tasks.utils.push_primitive_sac_env import PushPrimitiveSACEnv

    from skrl.agents.torch.sac import SAC_RNN, SAC_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from asyncDualPlayPPO.algorithms.rl.sac.models import PushPolicyRNN, PushCritic

    sac_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/sac/sac_push.yaml")
    sac_cfg = load_cfg(sac_cfg_path)

    max_pushes_per_episode = sac_cfg["params"]["training"]["max_pushes_per_episode"]
    pushes_per_iter = args.pushes_per_iter

    # ── Environment config ────────────────────────────────────────────────────
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )

    if args.with_distractor:
        env_cfg.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-0.25, 0.7, 0.05],
                rot=[0.0, 0.0, 0.0, 1.0],
            ),
            spawn=UsdFileCfg(
                func=spawn_random_block,
                usd_path=f"{ISAACLAB_DUAL_ARM_EXT_DIR}/asyncDualPlayPPO/assets/blocks/cube.usd",
                scale=(2.25, 2.25, 2.25),
                mass_props=sim_utils.MassPropertiesCfg(density=1200.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            ),
        )
        print("[Config] Distractor ENABLED.")
    else:
        env_cfg.scene.cube = None

    print("Creating environment...")
    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)

    push_env = PushEnvWrapper(
        env=base_env,
        device=base_env.device,
        num_objects=1,
        max_pushes_per_episode=max_pushes_per_episode,
        headless=args.headless,
        rel_obs=args.rel_obs,
    )
    print(f"Environment ready (rel_obs={args.rel_obs}, rel_act={args.rel_act}, obs_dim={push_env.obs_dim}D).")

    # ── cuRobo IK solver ──────────────────────────────────────────────────────
    print("[cuRobo] Initialising IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(push_env.device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    _ik_config.solver.newton_optimizer.n_iters = 30
    _ik_config.solver.newton_optimizer.inner_iters = 10
    ik_solver = IKSolver(_ik_config)
    print("[cuRobo] IK solver created.")

    print(f"[cuRobo] Warming up CUDA graph for N={push_env.num_envs} envs...")
    _wup_pos = torch.zeros(push_env.num_envs, 3, device=push_env.device)
    _wup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=push_env.device, dtype=torch.float32).expand(push_env.num_envs, 4)
    ik_solver.solve_batch(
        CuroboPose(position=_wup_pos, quaternion=_wup_quat),
        seed_config=torch.zeros(push_env.num_envs, 1, 6, device=push_env.device),
        retract_config=torch.zeros(push_env.num_envs, 6, device=push_env.device),
    )
    print("[cuRobo] Warm-up done.")

    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=push_env.device, dtype=torch.float32)

    # ── Goal marker ──
    _goal_viz = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkers",
            markers={
                "tblock": UsdFileCfg(
                    usd_path=os.path.join(os.path.dirname(__file__), "assets/blocks/t_shape.usda"),
                    scale=(2.0, 2.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),
                ),
            },
        )
    )

    def _update_goal_markers():
        goals = push_env.goal_pos_euler
        origins = push_env.env.scene.env_origins
        pos = goals[:, :3].clone()
        pos[:, 2] = 0.001
        euler = goals[:, 3:6].clone()
        euler[:, 0] = 0.0
        euler[:, 1] = 0.0
        quat = _euler_to_quat(euler)
        _goal_viz.visualize(translations=pos + origins, orientations=quat)

    # ── IK calibration ────────────────────────────────────────────────────────
    _robot_scene = push_env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - push_env.env.scene.env_origins).clone()

    print("[Setup] Calibrating IK→physics error...")
    _calib_pos = torch.zeros(push_env.num_envs, 3, device=push_env.device)
    _calib_pos[:, 1] = 0.60
    _calib_pos[:, 2] = 0.25
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_TOOL_DOWN.expand(push_env.num_envs, 4)),
        seed_config=_calib_cur.unsqueeze(1),
        retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(push_env.num_envs, 6)
    _calib_act = torch.zeros(push_env.num_envs, push_env.action_space.shape[0], device=push_env.device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        push_env.step(_calib_act)
    _finger_after = _tcp_pos_local()
    _TOTAL_IK_ERROR = (_finger_after - _calib_pos).clone()
    print(
        f"[Setup] IK error = ({float(_TOTAL_IK_ERROR[0,0]):+.3f}, "
        f"{float(_TOTAL_IK_ERROR[0,1]):+.3f}, {float(_TOTAL_IK_ERROR[0,2]):+.3f})"
    )

    # ── Create SAC environment ────────────────────────────────────────────────
    sac_env = PushPrimitiveSACEnv(
        push_env=push_env,
        ik_solver=ik_solver,
        device=push_env.device,
        rel_obs=args.rel_obs,
        rel_act=args.rel_act,
        max_pushes_per_episode=max_pushes_per_episode,
        ik_error=_TOTAL_IK_ERROR,
    )
    print("[SAC] Push primitive SAC env created.")

    # ── skrl SAC_RNN agent setup ──────────────────────────────────────────────
    obs_space = sac_env.observation_space
    act_space = sac_env.action_space
    device = push_env.device

    policy_cfg = sac_cfg["params"]["policy"]
    agent_cfg = sac_cfg["params"]["agent"]

    policy = PushPolicyRNN(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=policy_cfg["min_log_std"],
        max_log_std=policy_cfg["max_log_std"],
        num_envs=args.num_envs,
        num_layers=policy_cfg["lstm_num_layers"],
        hidden_size=policy_cfg["lstm_hidden_size"],
        sequence_length=args.sequence_length,
    )

    critic_1 = PushCritic(obs_space, act_space, device)
    critic_2 = PushCritic(obs_space, act_space, device)
    target_critic_1 = PushCritic(obs_space, act_space, device)
    target_critic_2 = PushCritic(obs_space, act_space, device)

    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    models = {
        "policy": policy,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }

    memory = RandomMemory(
        memory_size=args.buffer_size,
        num_envs=args.num_envs,
        device=device,
    )

    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["batch_size"] = args.batch_size
    cfg["discount_factor"] = agent_cfg["discount_factor"]
    cfg["polyak"] = agent_cfg["polyak"]
    cfg["actor_learning_rate"] = args.lr
    cfg["critic_learning_rate"] = args.lr
    cfg["entropy_learning_rate"] = args.lr
    cfg["learn_entropy"] = agent_cfg["learn_entropy"]
    cfg["initial_entropy_value"] = agent_cfg["initial_entropy_value"]
    cfg["target_entropy"] = agent_cfg["target_entropy"]
    cfg["gradient_steps"] = args.gradient_steps
    cfg["random_timesteps"] = args.warmup_pushes
    cfg["learning_starts"] = args.warmup_pushes
    cfg["grad_norm_clip"] = agent_cfg["grad_norm_clip"]
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["experiment"]["directory"] = ""

    agent = SAC_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )
    agent.init()
    print("[SAC] Agent initialized.")

    if args.chkpt and os.path.isfile(args.chkpt):
        agent.load(args.chkpt)
        print(f"[Resume] Loaded agent from {args.chkpt}")

    # ── Training state ────────────────────────────────────────────────────────
    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    os.makedirs(f"runs/{args.exp_name}/agent", exist_ok=True)
    os.makedirs(f"runs/{args.exp_name}/summary", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")

    best_success_rate = -1.0
    rew_buf = deque(maxlen=pushes_per_iter * args.num_envs)
    sr_buf = deque(maxlen=pushes_per_iter * args.num_envs)
    rot_sr_buf = deque(maxlen=pushes_per_iter * args.num_envs)
    pos_err_buf = deque(maxlen=pushes_per_iter * args.num_envs)
    rot_err_buf = deque(maxlen=pushes_per_iter * args.num_envs)
    ema_rew = 0.0

    print(f"\n{'='*80}\nPUSH-SAC (skrl): {args.exp_name}\nLOG DIR: {run_dir}\n{'='*80}\n")

    # ── Init environment ──────────────────────────────────────────────────────
    print("Initialising environment...")
    with SuppressAllOutput():
        obs = sac_env.reset()
    _update_goal_markers()
    print("Training loop starting...")
    sys.stdout.flush()

    timestep = 0
    total_timesteps = args.max_iterations * pushes_per_iter * args.num_envs
    iteration = 0

    _shutdown_requested = False

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        print("[INFO] SIGTERM received — checkpoint after current iteration.", flush=True)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── TRAINING LOOP ────────────────────────────────────────────────────────
    while iteration < args.max_iterations:
        push_env.episode_push_counts.clear()
        push_env.episode_successes.clear()
        sac_env.reset_ik_counters()

        for push_step in range(pushes_per_iter):
            agent.pre_interaction(timestep=timestep, timesteps=total_timesteps)

            with torch.no_grad():
                actions = agent.act(obs, timestep=timestep, timesteps=total_timesteps)

            if isinstance(actions, tuple):
                actions = actions[0]

            next_obs, reward, terminated, truncated, info = sac_env.step(actions)

            agent.record_transition(
                states=obs.detach(),
                actions=actions.detach(),
                rewards=reward.unsqueeze(-1).detach(),
                next_states=next_obs.detach(),
                terminated=terminated.unsqueeze(-1),
                truncated=truncated.unsqueeze(-1),
                infos={},
                timestep=timestep,
                timesteps=total_timesteps,
            )

            agent.post_interaction(timestep=timestep, timesteps=total_timesteps)

            rew_buf.extend(reward.cpu().tolist())
            at_goal = info["at_goal"].float()
            sr_buf.extend(at_goal.cpu().tolist())
            rot_sr_buf.extend((info["rot_err"] < 0.2).float().cpu().tolist())
            pos_err_buf.extend(info["pos_err"].cpu().tolist())
            rot_err_buf.extend(info["rot_err"].cpu().tolist())

            if terminated.any():
                _update_goal_markers()

            obs = next_obs
            timestep += 1

        # ── Logging ───────────────────────────────────────────────────────────
        mean_rew = np.mean(rew_buf) if rew_buf else 0.0
        mean_pos_err = np.mean(pos_err_buf) if pos_err_buf else 0.0
        mean_rot_err = np.mean(rot_err_buf) if rot_err_buf else 0.0
        sr = np.mean(sr_buf) if sr_buf else 0.0
        rot_sr = np.mean(rot_sr_buf) if rot_sr_buf else 0.0
        ik_fail_rate = sac_env.get_ik_fail_rate()

        ema_rew = 0.9 * ema_rew + 0.1 * mean_rew

        ep_push_counts = list(push_env.episode_push_counts)
        ep_successes = list(push_env.episode_successes)
        avg_pushes = np.mean(ep_push_counts) if ep_push_counts else float("nan")
        ep_sr = np.mean(ep_successes) if ep_successes else float("nan")
        n_episodes = len(ep_push_counts)

        writer.add_scalar("Reward/Mean", mean_rew, iteration)
        writer.add_scalar("Reward/EMA", ema_rew, iteration)
        writer.add_scalar("Metrics/SuccessRate", sr, iteration)
        writer.add_scalar("Metrics/RotationSR", rot_sr, iteration)
        writer.add_scalar("Metrics/PosError", mean_pos_err, iteration)
        writer.add_scalar("Metrics/RotError", mean_rot_err, iteration)
        writer.add_scalar("Metrics/IKFailRate", ik_fail_rate, iteration)
        writer.add_scalar("Metrics/EpisodicSR", ep_sr if not np.isnan(ep_sr) else 0.0, iteration)
        writer.add_scalar("Metrics/AvgPushesPerEpisode", avg_pushes if not np.isnan(avg_pushes) else 0.0, iteration)
        writer.add_scalar("Metrics/Episodes", n_episodes, iteration)
        buf_size = memory.memory_size if memory.filled else memory.memory_index
        writer.add_scalar("Metrics/BufferSize", buf_size, iteration)
        writer.add_scalar("Metrics/Timestep", timestep, iteration)

        if args.rel_act and args.rel_obs:
            _mode = "rel_full"
        elif args.rel_act:
            _mode = "rel_act"
        elif args.rel_obs:
            _mode = "rel_obs"
        else:
            _mode = "abs"

        avg_pushes_str = f"{avg_pushes:.1f}" if not np.isnan(avg_pushes) else "nan"
        _pr(
            f"[Iter {iteration:5d}] "
            f"Rew={mean_rew:+.4f} (EMA {ema_rew:+.4f}) | "
            f"PosErr={mean_pos_err:.4f} | RotErr={mean_rot_err:.4f} | "
            f"SR={sr:.4f} | RotSR={rot_sr:.4f} | "
            f"IK_fail={ik_fail_rate:.3f} | "
            f"AvgPushes={avg_pushes_str} | Epi={n_episodes} | "
            f"Buf={buf_size} | BestSR={best_success_rate:.4f} | {_mode}"
        )
        sys.stdout.flush()

        if sr > best_success_rate:
            best_success_rate = sr
            agent.save(os.path.join(run_dir, "agent", "best_agent.pt"))

        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(run_dir, "agent", f"agent_{iteration}.pt"))
            print(f"  [Checkpoint] Saved agent_{iteration}.pt")

        rew_buf.clear()
        sr_buf.clear()
        rot_sr_buf.clear()
        pos_err_buf.clear()
        rot_err_buf.clear()
        iteration += 1

        if _shutdown_requested:
            agent.save(os.path.join(run_dir, "agent", f"agent_{iteration}_emergency.pt"))
            print("[INFO] Emergency checkpoint saved. Shutting down.")
            break

    print(f"\nTraining complete. Best SR: {best_success_rate:.4f}")
    agent.save(os.path.join(run_dir, "agent", "agent_final.pt"))
    simulation_app.close()


if __name__ == "__main__":
    main()
