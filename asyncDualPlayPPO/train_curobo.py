# cuRobo IK variant of train.py.
#
# Differences from train.py / train_diffik.py:
#   1. cuRobo imports are at the top of the file (required before AppLauncher).
#   2. num_cat_dims is forced to 6 (XYZ + Rx + Ry + gripper).
#   3. cuRobo IKSolver replaces the implicit DiffIK controller:
#        bins → XYZ & rot delta → accumulate ee_target_local & ee_target_quat_w → solve_batch → joint positions
#   4. IK failure causes an immediate episode reset (user requirement).
#   5. Profiler section "curobo_ik" measures IK latency per rollout step.
#   6. ik_fail_rate is logged to TensorBoard each iteration.
#
# Everything else (ASP two-phase structure, ABC, GoalEncoder, historical pool,
# PPOABC, reward shaping, checkpoint logic) is untouched.
#
# Run locally:
#   python train_curobo.py --num_envs 16 --max_iterations 500 --exp_name curobo_test --headless
#
# Run on HPC:
#   sbatch hpc/train_curobo.slurm

# ── cuRobo MUST be imported before AppLauncher ────────────────────────────────
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print(
        "\n[ERROR] cuRobo not found. Install it in the active venv before training.\n"
        "  Check: apptainer exec --nv isaac-lab.sif python -c 'import curobo'\n"
    )
    import sys
    sys.exit(1)

# Lock the venv's torch submodules into sys.modules NOW, before AppLauncher prepends
# Isaac Sim's pip_prebundle path to sys.path.  Without this, `import torch._dynamo`
# inside torch.optim.Adam resolves to Isaac Sim's incompatible dynamo build and crashes.
import torch
import torch._dynamo          # noqa: F401  (side-effect: caches correct version)
import torch._C               # noqa: F401
import torch.optim            # noqa: F401

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

# ── Arm joint names for cuRobo (must match the UR5e URDF used by Isaac Lab) ──
_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Workspace clamp limits (local / env-origin-relative frame, metres)
_WS_X = (-0.50, 0.50)
_WS_Y = (0.25,  0.70)
_WS_Z = ( 0.00, 0.55)

# EE home-position offset applied after every sync (reset / phase boundary).
# The arm resets to its default joint configuration; these offsets steer the
# IK target to the preferred resting pose so the arm settles there on the
# first few steps of each phase.
_EE_HOME_X_OFFSET = 0.02   # metres forward on X
_EE_HOME_Y        = 0.50   # metres — hover directly over T-block spawn
_EE_HOME_Z        = 0.05   # metres above table (env-local Z)


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
    parser = argparse.ArgumentParser(description="Train Async Dual Play PPO — cuRobo IK variant")
    parser.add_argument("--exp_name", type=str, default="curobo_dual_play")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--alice_decay_alpha", type=float, default=3.33)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--max_alice_bob_ratio", type=int, default=None)
    parser.add_argument("--chkpt_alice", type=str, default=None)
    parser.add_argument("--chkpt_bob", type=str, default=None)
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument(
        "--arm_config",
        type=str,
        default="default",
        choices=["default", "rotated"],
    )
    parser.add_argument("--num_objects", type=int, default=1, choices=[1, 2])
    parser.add_argument("--dummy_alice", action="store_true")
    parser.add_argument("--diag_alice_exploration", action="store_true")
    parser.add_argument("--test_bob_reward", action="store_true")
    parser.add_argument("--test_abc_verbose", action="store_true")
    parser.add_argument("--test_hparams", action="store_true")
    parser.add_argument("--test_reward_pipeline", action="store_true",
                        help="Run Test 1: teleport objects to goals and verify sparse reward fires. Exits after test.")
    parser.add_argument("--alice_sandbox", action="store_true",
                        help="Test 2: disable Bob PPO updates (Bob acts randomly) to isolate Alice curriculum.")
    parser.add_argument("--dummy_goal_distance", action="store_true")
    parser.add_argument("--test_movement", action="store_true")
    parser.add_argument(
        "--diag_alice_shaping",
        action="store_true",
        help="Diagnostic: add small per-step proximity shaping for Alice to "
             "confirm she can learn contact manipulation. Remove after confirmed.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-iteration timing profiler (includes curobo_ik section).",
    )
    parser.add_argument(
        "--debug_rewards",
        action="store_true",
        help="Print per-step Alice reward breakdown (auto-enabled when num_envs < 20).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 20:
        args.debug_rewards = True

    install_noise_filter()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.utils.events import reset_objects_to_random_safe_pose as _rand_reset_objs
    from asyncDualPlayPPO.tasks.async_dual_play_curobo import AsyncDualPlayCuRoboEnvCfg as AsyncDualPlayEnvCfg
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

    from asyncDualPlayPPO.tasks.utils.rewards import (
        ALICE_BOB_FAIL_REWARD,
        ALICE_BOB_SUCCESS_REWARD,
        ALICE_VALID_GOAL_BONUS,
        ALICE_INVALID_GOAL_PENALTY,
    )

    task_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/task/AsyncDualPlay.yaml")
    ppo_cfg_path  = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    ppo_cfg  = load_cfg(ppo_cfg_path)
    task_cfg = load_cfg(task_cfg_path)

    alice_timesteps       = task_cfg.get("alice_timesteps", 100)
    bob_timesteps         = task_cfg.get("bob_timesteps", 200)
    max_goals_per_episode = task_cfg.get("max_goals_per_episode", 5)
    print(
        f"[Config] Episode structure: alice_timesteps={alice_timesteps}, "
        f"bob_timesteps={bob_timesteps}, max_goals={max_goals_per_episode}"
    )

    if args.test_hparams:
        learn = ppo_cfg["params"]["learn"]
        PAPER = {"gamma": 0.998, "lam": 0.95, "e_clip": 0.2, "entropy_coef": 0.05,
                 "learning_rate": 3e-4, "mini_epochs": 5, "critic_coef": 1.0, "abc_coef": 0.5}
        actual = {"gamma": learn.get("gamma","?"), "lam": learn.get("lam","?"),
                  "e_clip": learn.get("cliprange","?"), "entropy_coef": learn.get("ent_coef","?"),
                  "learning_rate": learn.get("optim_stepsize","?"),
                  "mini_epochs": learn.get("noptepochs","?"),
                  "critic_coef": learn.get("value_loss_coef","?"), "abc_coef": learn.get("abc_coef","?")}
        print("\n" + "=" * 60)
        print("HYPERPARAMETER AUDIT  (paper Table 2 vs loaded config)")
        print(f"{'Key':<18} {'Paper':>10} {'Loaded':>10}  {'OK?':>6}")
        print("-" * 60)
        for k, pv in PAPER.items():
            av = actual[k]
            ok = "✓" if av == pv else "✗ MISMATCH"
            print(f"  {k:<16} {str(pv):>10} {str(av):>10}  {ok}")
        print("=" * 60)
        sys.exit(0)

    if args.nsteps is not None:
        ppo_cfg["params"]["learn"]["nsteps"] = args.nsteps

    # ── 6D action space: XYZ (3) + Rx/Ry tilt (2) + gripper (1) ─────────────────────
    _pol_cfg = ppo_cfg["params"]["policy"]
    _pol_cfg["use_multicategorical"] = True
    _pol_cfg["num_cat_dims"] = 6
    use_mc         = True
    num_cat_dims   = 6
    num_bins       = _pol_cfg.get("num_bins", 11)
    _max_delta_m   = _pol_cfg.get("max_delta_m", 0.02)
    _max_delta_rot = _pol_cfg.get("max_delta_rot", 0.05)   # rad/step EE tilt
    print(
        f"[Config] cuRobo action space: {num_cat_dims}D × {num_bins} bins "
        f"(XYZ + Rx/Ry + gripper)  max_delta={_max_delta_m*100:.1f} cm  "
        f"max_rot={math.degrees(_max_delta_rot):.1f} deg/step"
    )

    # ── Helper: decode bins → XYZ delta + Rx/Ry delta + updated sticky gripper ────
    def _bins_to_xyz_rxy_gripper(bin_indices, gripper_state):
        """
        (N, 6) int bins → (N,3) XYZ delta, (N,2) Rx/Ry delta rad, (N,1) gripper.

        Dim layout:
            0-2: XYZ    → (bin − center) / center × max_delta_m
            3-4: Rx, Ry   → (bin − center) / center × max_delta_rot
            5:   Gripper  → sticky: outer bins open/close, center holds
        """
        center    = (num_bins - 1) / 2.0
        threshold = 2.0
        normalized = (bin_indices.float() - center) / center          # [-1, 1], (N,6)
        xyz  = normalized[:, :3] * _max_delta_m                       # (N, 3)
        rxry = normalized[:, 3:5] * _max_delta_rot                    # (N, 2)
        # Clamp per-step delta so a single bin can't cause a violent orientation jump
        rxry = rxry.clamp(-0.05, 0.05)
        g_bin   = bin_indices[:, 5].float()
        new_gs  = gripper_state.clone()
        new_gs[g_bin < center - threshold + 1] = -1.0                 # bins 0-2  → close
        new_gs[g_bin > center + threshold - 1] =  1.0                 # bins 8-10 → open
        return xyz, rxry, new_gs

    # ── Environment config ────────────────────────────────────────────────────────
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Replace DiffIK arm action with direct joint position control.
    # cuRobo computes joint positions externally; the env just needs to accept them.
    # scale=1.0 + use_default_offset=False → action IS the joint position in radians.
    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )

    if args.num_objects == 1:
        print("[Config] num_objects=1: removing cube from scene and observations.")
        env_cfg.scene.cube = None
        env_cfg.observations.alice_policy.cube_state     = None
        env_cfg.observations.bob_policy.cube_state       = None
        env_cfg.observations.bob_policy.cube_goal_state  = None
        env_cfg.observations.bob_policy.cube_goal_distance = None

    if args.arm_config == "rotated":
        print("[Config] Rotated arm configuration: shoulder −90°")
        env_cfg.scene.robot.init_state.joint_pos["shoulder_pan_joint"] = 1.57
        env_cfg.scene.robot.init_state.joint_pos["elbow_joint"]        = 2.356
        env_cfg.scene.robot.init_state.joint_pos["wrist_1_joint"]      = -0.785
        env_cfg.scene.robot.init_state.joint_pos["wrist_2_joint"]      = 1.57
        env_cfg.scene.robot.init_state.joint_pos["wrist_3_joint"]      = 0

    print("Creating environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)

    if args.test_bob_reward or args.test_reward_pipeline:
        env = DummyBobWrapper(env=base_env, device=base_env.device,
                              alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
                              teleport_step=10, num_objects=args.num_objects)
    elif args.dummy_goal_distance:
        env = DummyGoalDistanceWrapper(env=base_env, device=base_env.device,
                                       alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
                                       teleport_step=10, num_objects=args.num_objects)
    elif args.test_movement:
        env = DummyMovementWrapper(env=base_env, device=base_env.device,
                                   alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
                                   num_objects=args.num_objects)
    elif args.diag_alice_exploration:
        env = DiagnosticAliceWrapper(env=base_env, device=base_env.device,
                                     alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
                                     num_objects=args.num_objects)
    elif args.dummy_alice:
        env = DummyAliceWrapper(env=base_env, device=base_env.device,
                                alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
                                num_objects=args.num_objects)
    else:
        env = AsyncDualPlayEnvWrapper(
            env=base_env,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            max_goals_per_episode=max_goals_per_episode,
            num_objects=args.num_objects,
            device=base_env.device,
            arm_config=args.arm_config,
        )
        if args.diag_alice_shaping:
            env._diag_alice_shaping = True
            print("[Config] Diagnostic Alice per-step shaping ENABLED (EE→object proximity bonus).")

    print("Environment ready.")

    # ── cuRobo IK solver setup ─────────────────────────────────────────────────────
    print("[cuRobo] Initialising IK solver…")
    _tensor_args = TensorDeviceType(
        device=torch.device(env.device), dtype=torch.float32
    )
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args
    )
    ik_solver = IKSolver(_ik_config)
    print("[cuRobo] IK solver created.")

    # Warm-up: trace the CUDA graph with the training batch size so all subsequent
    # calls reuse the graph (reduces per-step latency from ~3 ms to ~0.5 ms).
    print(f"[cuRobo] Warming up CUDA graph for N={env.num_envs} envs…")
    _wup_pos  = torch.zeros(env.num_envs, 3, device=env.device)
    _wup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device,
                              dtype=torch.float32).expand(env.num_envs, 4).contiguous()
    ik_solver.solve_batch(
        CuroboPose(position=_wup_pos, quaternion=_wup_quat),
        seed_config=torch.zeros(env.num_envs, 1, 6, device=env.device),
        retract_config=torch.zeros(env.num_envs, 6, device=env.device),
    )
    print("[cuRobo] CUDA graph warm-up done.")

    # Tool-pointing-down base quaternion (wxyz). Used to initialise and reset
    # ee_target_quat_w so the arm defaults to a vertical gripper orientation.
    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)

    def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batch quaternion multiply, wxyz convention. Both (N,4) → (N,4)."""
        w = a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3]
        x = a[:,0]*b[:,1] + a[:,1]*b[:,0] + a[:,2]*b[:,3] - a[:,3]*b[:,2]
        y = a[:,0]*b[:,2] - a[:,1]*b[:,3] + a[:,2]*b[:,0] + a[:,3]*b[:,1]
        z = a[:,0]*b[:,3] + a[:,1]*b[:,2] - a[:,2]*b[:,1] + a[:,3]*b[:,0]
        return torch.stack([w, x, y, z], dim=1)

    # Robot body/joint indices needed each rollout step (found once, reused)
    _robot_scene  = env.env.scene["robot"]
    _arm_jids, _  = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _wrist3_ids,_ = _robot_scene.find_bodies("wrist_3_link")
    _lf_ids, _    = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _    = _robot_scene.find_bodies("right_inner_finger")

    # Per-env accumulated EE position target (local / env-origin-relative frame).
    # Initialised to current wrist_3_link position; reset at every phase boundary
    # and after IK failures to prevent target drift.
    ee_target_local = (
        _robot_scene.data.body_pos_w[:, _wrist3_ids[0]]
        - env.env.scene.env_origins
    ).clone()  # (N, 3)

    # Per-env accumulated EE orientation target (wxyz quaternion, world frame).
    # Initialised to tool-down; updated by composing per-step Rx/Ry delta quats.
    ee_target_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()  # (N, 4)

    # ── PPO agent setup (identical to train.py from here) ─────────────────────────
    import copy
    import gymnasium as gym_mc

    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32
    )

    alice_ppo = PPO(
        vec_env=env,
        cfg_train=ppo_cfg["params"],
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/alice",
        asymmetric=False,
    )
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space       = alice_ppo.observation_space
    alice_ppo.action_space      = _mc_space
    alice_ppo.desired_kl        = None

    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape,
        alice_ppo.action_space.shape,
        alice_ppo.init_noise_std,
        alice_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    max_alice_steps   = env.episode_manager.alice_timesteps + 10
    _rollout_len      = env.episode_manager.alice_timesteps + env.episode_manager.bob_timesteps
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

    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"]["use_goal_encoder"] = True
    bob_cfg["policy"]["num_objects"]      = args.num_objects

    bob_ppo = PPOABC(
        vec_env=env,
        cfg_train=bob_cfg,
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/bob",
        asymmetric=False,
    )
    bob_ppo.observation_space = env.bob_observation_space
    bob_ppo.state_space       = bob_ppo.observation_space
    bob_ppo.action_space      = _mc_space
    bob_ppo.desired_kl        = None

    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.init_noise_std,
        bob_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    # Scale down goal_proj at init to avoid ReLU saturation before training starts.
    if (
        hasattr(bob_ppo.actor_critic, "_goal_proj")
        and bob_ppo.actor_critic._goal_proj is not None
    ):
        with torch.no_grad():
            bob_ppo.actor_critic._goal_proj.weight.mul_(0.1)

    max_bob_steps    = env.episode_manager.bob_timesteps + 10
    bob_storage_size = max(bob_ppo.num_transitions_per_env + max_bob_steps, _rollout_len + 10)
    bob_ppo.storage  = bob_ppo.storage.__class__(
        bob_ppo.vec_env.num_envs,
        bob_storage_size,
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.device,
        "sequential",
    )

    _abc_act_shape = (num_cat_dims,)
    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
        capacity=50000,
        obs_shape=env.bob_observation_space.shape,
        states_shape=env.bob_observation_space.shape,
        actions_shape=_abc_act_shape,
        device=env.device,
        traj_maxlen=ppo_cfg["params"]["learn"].get("abc_traj_maxlen", 500),
    )

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
        bob_hidden   = None

    alice_gripper_state = torch.ones(env.num_envs, 1, device=env.device)
    bob_gripper_state   = torch.ones(env.num_envs, 1, device=env.device)

    alice_pool = HistoricalPolicyPool(max_size=5)
    bob_pool   = HistoricalPolicyPool(max_size=5)
    HIST_SAVE_INTERVAL = 50
    HIST_FRAC          = 0.2

    _learn             = ppo_cfg["params"]["learn"]
    _alice_target_sr   = _learn.get("alice_entropy_target_sr", 0.5)
    _alice_entropy_lr  = _learn.get("alice_entropy_lr", 1e-3)
    _alice_entropy_min = _learn.get("alice_entropy_min", 0.05)
    _alice_entropy_max = _learn.get("alice_entropy_max", 1.0)

    # ── Resume from checkpoint ─────────────────────────────────────────────────────
    bob_success_buf = deque(maxlen=200)

    if args.chkpt_alice and os.path.isfile(args.chkpt_alice):
        alice_ppo.load(args.chkpt_alice)
        print(f"[Resume] Loaded Alice from {args.chkpt_alice}")
    if args.chkpt_bob and os.path.isfile(args.chkpt_bob):
        bob_ppo.load(args.chkpt_bob)
        print(f"[Resume] Loaded Bob from {args.chkpt_bob}")
        _abc_buf_path = os.path.join(os.path.dirname(args.chkpt_bob), "abc_buffer.pt")
        if os.path.isfile(_abc_buf_path):
            bob_ppo.abc_buffer.load(_abc_buf_path)
            print(f"[Resume] Loaded ABC buffer ({bob_ppo.abc_buffer.size} entries)")
        _ep_mgr_path = args.chkpt_bob.replace("model_", "episode_manager_")
        if os.path.isfile(_ep_mgr_path):
            ep_sd = torch.load(_ep_mgr_path, map_location=env.device)
            env.episode_manager.load_state_dict(ep_sd)
            print(f"[Resume] Loaded EpisodeManager state")
        _train_state_path = args.chkpt_bob.replace("model_", "train_state_")
        if os.path.isfile(_train_state_path):
            _ts = torch.load(_train_state_path, map_location="cpu")
            alice_ppo.entropy_coef = float(_ts["entropy_coef"])
            bob_ppo.abc_coef       = float(_ts["abc_coef"])
            bob_success_buf.extend(_ts.get("bob_success_buf", []))
            print(f"[Resume] Restored train state: entropy_coef={alice_ppo.entropy_coef:.4f}")

    alice_updates = args.resume_iteration
    bob_updates   = args.resume_iteration

    writer   = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")
    profiler = (
        env._profiler
        if hasattr(env, "_profiler") and env._profiler is not None
        else TrainingProfiler(enabled=getattr(args, "profile", False), device=env.device)
    )

    rollout_length = ppo_cfg["params"]["learn"]["nsteps"] * args.num_envs
    alice_rew_buf   = deque(maxlen=rollout_length)
    bob_rew_buf     = deque(maxlen=rollout_length)
    bob_pos_err_buf = deque(maxlen=rollout_length)
    bob_rot_err_buf = deque(maxlen=rollout_length)

    best_bob_success_rate = -1.0
    last_alice_mean_rew   = 0.0
    ema_alice_rew         = 0.0

    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    print(f"\n{'='*80}\nTRAINING RUN: {args.exp_name}\nLOG DIR: {run_dir}\n{'='*80}\n")

    # ── Per-iteration IK diagnostics ────────────────────────────────────────────────
    _ik_fail_steps  = 0   # IK failures in active envs this iteration
    _ik_total_steps = 0   # total active-env steps this iteration

    # ── Alice / Bob update functions ───────────────────────────────────────────────
    def perform_alice_update():
        nonlocal alice_updates, last_alice_mean_rew, ema_alice_rew
        if alice_ppo.storage.step == 0:
            return
        dummy_val = torch.zeros(env.num_envs, 1, device=env.device)
        alice_ppo.storage.compute_returns(dummy_val, alice_ppo.gamma, alice_ppo.lam)
        loss_val, loss_surr = alice_ppo.update()
        alice_ppo.storage.clear()
        mean_alice_rew   = np.mean(alice_rew_buf) if alice_rew_buf else 0.0
        last_alice_mean_rew = mean_alice_rew
        ema_alice_rew    = 0.9 * ema_alice_rew + 0.1 * mean_alice_rew
        writer.add_scalar("Loss/Alice/Value",    loss_val,   alice_updates)
        writer.add_scalar("Loss/Alice/Surrogate", loss_surr, alice_updates)
        writer.add_scalar("Reward/Alice",        mean_alice_rew, alice_updates)
        print(f"  [Alice Update {alice_updates}] Loss: {loss_surr:.4f} | Val: {loss_val:.4f} | Rew: {mean_alice_rew:.4f}", flush=True)
        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(current_bob_obs):
        nonlocal bob_updates, best_bob_success_rate
        if args.alice_sandbox:
            # Test 2: Bob update suppressed — Alice curriculum runs with random Bob
            bob_updates += 1
            return
        total_bob_transitions = bob_ppo.storage.step * env.num_envs
        if total_bob_transitions < bob_ppo.num_mini_batches:
            print(f"  [Bob Update {bob_updates}] SKIPPED (only {total_bob_transitions} transitions)", flush=True)
            bob_updates += 1
            return

        with torch.no_grad():
            _, _, last_val_b, _, _ = bob_ppo.actor_critic.act(current_bob_obs, None)

        if bob_ppo.actor_critic.use_goal_encoder:
            with torch.no_grad():
                sample_obs = current_bob_obs[:8]
                _robot_dim, _obj_state_dim, _goal_dim = 7, 14, 6
                if args.num_objects == 1:
                    s_t_batch   = sample_obs[:, _robot_dim : _robot_dim + _goal_dim]
                    _gs         = _robot_dim + _obj_state_dim
                    s_star_batch = sample_obs[:, _gs : _gs + _goal_dim]
                else:
                    s_t_batch   = torch.cat([sample_obs[:, 7:13], sample_obs[:, 29:35]], dim=-1)
                    s_star_batch = torch.cat([sample_obs[:, 21:27], sample_obs[:, 43:49]], dim=-1)
                g_sample = bob_ppo.actor_critic.goal_encoder(s_star_batch, s_t_batch)
                writer.add_scalar("GoalEncoder/embedding_norm", g_sample.norm(dim=-1).mean().item(), bob_updates)
                writer.add_scalar("GoalEncoder/embedding_std",  g_sample.std(dim=-1).mean().item(),  bob_updates)

        bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
        bob_ppo.current_learning_iteration = bob_updates
        loss_val, loss_surr, loss_abc, _ = bob_ppo.update(alice_mean_rew=last_alice_mean_rew)
        bob_ppo.storage.clear()

        mean_bob_rew   = np.mean(bob_rew_buf) if bob_rew_buf else 0.0
        bob_success_rate = np.mean(bob_success_buf) if bob_success_buf else 0.0
        mean_pos_err   = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        mean_rot_err   = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0

        writer.add_scalar("Loss/Bob/Value",      loss_val,        bob_updates)
        writer.add_scalar("Loss/Bob/Surrogate",  loss_surr,       bob_updates)
        writer.add_scalar("Loss/Bob/ABC",        loss_abc,        bob_updates)
        writer.add_scalar("Reward/Bob",          mean_bob_rew,    bob_updates)
        writer.add_scalar("Metrics/Bob/SuccessRate", bob_success_rate, bob_updates)
        writer.add_scalar("Metrics/Bob/PosError",    mean_pos_err, bob_updates)
        writer.add_scalar("Metrics/Bob/RotError",    mean_rot_err, bob_updates)
        print(f"  [Bob Update {bob_updates}] Loss: {loss_surr:.4f} | Val: {loss_val:.4f} | Rew: {mean_bob_rew:.4f} | ABC: {loss_abc:.4f} | SR: {bob_success_rate:.4f} | ABCCoef: {bob_ppo.abc_coef:.4f}", flush=True)

        if bob_success_rate > best_bob_success_rate:
            best_bob_success_rate = bob_success_rate
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_best.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))
            torch.save(env.episode_manager.state_dict(), os.path.join(bob_ppo.log_dir, "episode_manager_best.pt"))
            torch.save({"entropy_coef": alice_ppo.entropy_coef, "abc_coef": bob_ppo.abc_coef,
                        "bob_success_buf": list(bob_success_buf)},
                       os.path.join(bob_ppo.log_dir, "train_state_best.pt"))

        bob_rew_buf.clear()
        bob_pos_err_buf.clear()
        bob_rot_err_buf.clear()
        bob_updates += 1

    # ── SIGTERM handler ────────────────────────────────────────────────────────────
    _shutdown_requested = False

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        print("[INFO] SIGTERM received — emergency checkpoint after current iteration.", flush=True)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── Environment initialisation ─────────────────────────────────────────────────
    print("Initialising environment (suppressing URDF/Lula warnings)...")
    with SuppressAllOutput():
        obs = env.reset()[0]
    print("Environment ready. Starting training loop...")

    if not (args.chkpt_alice and os.path.isfile(args.chkpt_alice)):
        if args.test_reward_pipeline or args.test_bob_reward or args.dummy_alice \
           or args.dummy_goal_distance or args.test_movement or args.diag_alice_exploration:
            print(f"[Init] Diagnostic mode — skipping stagger (all envs start synchronized).")
        else:
            _stagger = torch.randint(
                0, max(1, env.episode_manager.alice_timesteps),
                (env.num_envs,), device=env.device, dtype=torch.int32,
            )
            env.episode_manager.phase_step.copy_(_stagger)
            print(f"[Init] Phase stagger applied.")

    # Sync initial ee_target with actual robot TCP positions after env reset
    _lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
    _rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
    _tcp_w = (_lf_w + _rf_w) / 2.0
    ee_target_local = (_tcp_w - env.env.scene.env_origins).clone()
    ee_target_local[:, 0] += _EE_HOME_X_OFFSET
    ee_target_local[:, 1]  = _EE_HOME_Y
    ee_target_local[:, 2]  = _EE_HOME_Z
    # Orientation tracker: start at tool-down for all envs
    ee_target_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()

    # Joint-command smoothing: EMA between consecutive IK solutions.
    # alpha=1 → no smoothing (raw IK); alpha<1 → inertia like RMPFlow.
    _JC_ALPHA = 1.0   # no EMA smoothing — paper uses direct TCP servoing
    _prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()  # (N,6)

    if args.test_reward_pipeline:
        from asyncDualPlayPPO.diagnostics.test_reward_pipeline import run_test1
        run_test1(env, env.episode_manager, device=env.device)
        sys.exit(0)

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────────
    while bob_updates < args.max_iterations:

        profiler.start_iteration()
        _ik_fail_steps  = 0
        _ik_total_steps = 0

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
            print(f"  [HistPool] Saved snapshot at iter {bob_updates} (pool sizes: alice={alice_pool.size}, bob={bob_pool.size})", flush=True)

        alice_ppo.storage.clear()
        bob_ppo.storage.clear()

        iter_sr_counts = [0, 0]

        with profiler.section("obs_compute"):
            obs_dict = env.env.observation_manager.compute()
            obs = torch.cat([obs_dict["alice_policy"], obs_dict["bob_policy"]], dim=-1)
        current_alice_obs = obs[:, : env.alice_obs_dim]
        current_bob_obs   = obs[:, env.alice_obs_dim :]

        _a_max_steps = env.episode_manager.alice_timesteps
        _a_pdim = num_cat_dims
        _b_pdim = num_cat_dims

        alice_traj_obs = torch.zeros((env.num_envs, _a_max_steps, env.alice_obs_dim), device=env.device)
        alice_traj_act = torch.zeros((env.num_envs, _a_max_steps, _a_pdim), device=env.device)
        alice_traj_len = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        hist_alice = alice_pool.sample_policy(alice_ppo.actor_critic, env.device) if alice_pool.size > 0 else None
        hist_bob   = bob_pool.sample_policy(bob_ppo.actor_critic, env.device)     if bob_pool.size > 0 else None

        rollout_length = env.episode_manager.alice_timesteps + env.episode_manager.bob_timesteps

        # Debug accumulators (only used when args.debug_rewards)
        if args.debug_rewards:
            _dbg_rew_acc  = torch.zeros(env.num_envs, device=env.device)  # dense reward sum per env
            _dbg_ik_fails = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # Alice IK fail count
            _dbg_ik_bob   = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # Bob IK fail count
            _dbg_printed  = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # already printed

        for t in range(rollout_length):
            is_alice      = env.episode_manager.is_alice_phase()
            is_bob        = env.episode_manager.is_bob_phase()
            alice_indices = torch.where(is_alice)[0]
            bob_indices   = torch.where(is_bob)[0]

            alice_local_idx = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
            if len(alice_indices) > 0:
                alice_local_idx[alice_indices] = torch.arange(len(alice_indices), device=env.device)
            bob_local_idx = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
            if len(bob_indices) > 0:
                bob_local_idx[bob_indices] = torch.arange(len(bob_indices), device=env.device)

            # ── ALICE ACTIONS ─────────────────────────────────────────────────────
            profiler.mark_start("alice_act")
            a_acts_active   = torch.zeros((len(alice_indices), _a_pdim), device=env.device)
            a_logprob_active = torch.zeros(len(alice_indices), device=env.device)
            a_val_active    = torch.zeros(len(alice_indices), 1, device=env.device)
            a_mu_active     = torch.zeros_like(a_acts_active)
            a_sigma_active  = torch.zeros_like(a_acts_active)

            if len(alice_indices) > 0:
                hist_ids, curr_ids = alice_pool.sample_env_subset(alice_indices, frac=HIST_FRAC)
                with torch.no_grad():
                    h_in = ((alice_hidden[0][curr_ids], alice_hidden[1][curr_ids]) if alice_hidden else None)
                    (a_acts_curr, a_logprob_curr, a_val_curr, a_mu_curr, a_sigma_curr, new_h) = \
                        alice_ppo.actor_critic.act_with_hidden(current_alice_obs[curr_ids], None, h_in)
                    if alice_hidden and new_h is not None:
                        alice_hidden[0][curr_ids] = new_h[0]
                        alice_hidden[1][curr_ids] = new_h[1]
                    if len(hist_ids) > 0 and hist_alice is not None:
                        (a_acts_hist, a_logprob_hist, a_val_hist, a_mu_hist, a_sigma_hist, _) = \
                            hist_alice.act_with_hidden(current_alice_obs[hist_ids], None, None)
                    else:
                        hist_ids = torch.tensor([], dtype=torch.long, device=env.device)
                        a_acts_hist = a_logprob_hist = a_val_hist = a_mu_hist = a_sigma_hist = None

                curr_local = alice_local_idx[curr_ids]
                a_acts_active[curr_local]    = a_acts_curr
                a_logprob_active[curr_local] = a_logprob_curr
                a_val_active[curr_local]     = a_val_curr
                a_mu_active[curr_local]      = a_mu_curr
                a_sigma_active[curr_local]   = a_sigma_curr

                if len(hist_ids) > 0 and a_acts_hist is not None:
                    hist_local = alice_local_idx[hist_ids]
                    a_acts_active[hist_local]    = a_acts_hist
                    a_logprob_active[hist_local] = a_logprob_hist
                    a_val_active[hist_local]     = a_val_hist
                    a_mu_active[hist_local]      = a_mu_hist
                    a_sigma_active[hist_local]   = a_sigma_hist

            profiler.mark_stop("alice_act")

            # ── BOB ACTIONS ───────────────────────────────────────────────────────
            profiler.mark_start("bob_act")
            b_acts_active    = torch.zeros((len(bob_indices), _b_pdim), device=env.device)
            b_logprob_active = torch.zeros(len(bob_indices), device=env.device)
            b_val_active     = torch.zeros(len(bob_indices), 1, device=env.device)
            b_mu_active      = torch.zeros_like(b_acts_active)
            b_sigma_active   = torch.zeros_like(b_acts_active)

            if len(bob_indices) > 0:
                hist_bids, curr_bids = bob_pool.sample_env_subset(bob_indices, frac=HIST_FRAC)
                with torch.no_grad():
                    h_in = ((bob_hidden[0][curr_bids], bob_hidden[1][curr_bids]) if bob_hidden else None)
                    if args.alice_sandbox:
                        # Random Bob actions for Test 2 sandbox
                        n_c = len(curr_bids)
                        b_acts_curr = torch.randint(0, num_bins, (n_c, num_cat_dims), device=env.device).float()
                        b_lp_curr = torch.zeros(n_c, device=env.device)
                        b_val_curr = torch.zeros(n_c, 1, device=env.device)
                        b_mu_curr = torch.zeros(n_c, num_cat_dims, device=env.device)
                        b_sig_curr = torch.zeros(n_c, num_cat_dims, device=env.device)
                        new_bh = None
                    else:
                        (b_acts_curr, b_lp_curr, b_val_curr, b_mu_curr, b_sig_curr, new_bh) = \
                            bob_ppo.actor_critic.act_with_hidden(current_bob_obs[curr_bids], None, h_in)
                    if bob_hidden and new_bh is not None:
                        bob_hidden[0][curr_bids] = new_bh[0]
                        bob_hidden[1][curr_bids] = new_bh[1]
                    if len(hist_bids) > 0 and hist_bob is not None:
                        (b_acts_hist, b_lp_hist, b_val_hist, b_mu_hist, b_sig_hist, _) = \
                            hist_bob.act_with_hidden(current_bob_obs[hist_bids], None, None)
                    else:
                        hist_bids = torch.tensor([], dtype=torch.long, device=env.device)
                        b_acts_hist = b_lp_hist = b_val_hist = b_mu_hist = b_sig_hist = None

                curr_bloc = bob_local_idx[curr_bids]
                b_acts_active[curr_bloc]    = b_acts_curr
                b_logprob_active[curr_bloc] = b_lp_curr
                b_val_active[curr_bloc]     = b_val_curr
                b_mu_active[curr_bloc]      = b_mu_curr
                b_sigma_active[curr_bloc]   = b_sig_curr

                if len(hist_bids) > 0 and b_acts_hist is not None:
                    hist_bloc = bob_local_idx[hist_bids]
                    b_acts_active[hist_bloc]    = b_acts_hist
                    b_logprob_active[hist_bloc] = b_lp_hist
                    b_val_active[hist_bloc]     = b_val_hist
                    b_mu_active[hist_bloc]      = b_mu_hist
                    b_sigma_active[hist_bloc]   = b_sig_hist

            profiler.mark_stop("bob_act")

            # ── POLICY TENSORS (for storage) ──────────────────────────────────────
            a_policy = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_policy[alice_indices] = a_acts_active
            b_policy = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_policy[bob_indices]   = b_acts_active

            # ── STORE ALICE TRAJECTORY FOR ABC ───────────────────────────────────
            alice_step_raw = env.episode_manager.phase_step[alice_indices] - 1
            valid_t = alice_step_raw < _a_max_steps
            active_alice = alice_indices[valid_t]
            active_steps = alice_step_raw[valid_t]
            phase_start  = alice_step_raw == 0
            if phase_start.any():
                alice_traj_len[alice_indices[phase_start]] = 0
            if len(active_alice) > 0:
                alice_traj_obs[active_alice, active_steps] = current_alice_obs[active_alice].clone()
                alice_traj_act[active_alice, active_steps] = a_acts_active[valid_t].clone()
                new_len = (active_steps + 1).to(alice_traj_len.dtype)
                alice_traj_len[active_alice] = torch.max(alice_traj_len[active_alice], new_len)

            # ── cuRobo IK: convert bins → joint positions ─────────────────────────
            # 1. Decode XYZ + Rx/Ry delta + gripper from 6D bins
            a_xyz, a_rxry, new_ags = _bins_to_xyz_rxy_gripper(a_acts_active, alice_gripper_state[alice_indices])
            b_xyz, b_rxry, new_bgs = _bins_to_xyz_rxy_gripper(b_acts_active, bob_gripper_state[bob_indices])
            alice_gripper_state[alice_indices] = new_ags
            bob_gripper_state[bob_indices]     = new_bgs

            # 2. Accumulate per-env EE position targets, clamp to workspace.
            # Snapshot before the delta so IK-failure envs can be reverted below.
            _ee_target_local_pre  = ee_target_local.clone()
            _ee_target_quat_w_pre = ee_target_quat_w.clone()
            if len(alice_indices) > 0:
                ee_target_local[alice_indices] += a_xyz
            if len(bob_indices) > 0:
                ee_target_local[bob_indices]   += b_xyz
            ee_target_local[:, 0].clamp_(_WS_X[0], _WS_X[1])
            ee_target_local[:, 1].clamp_(_WS_Y[0], _WS_Y[1])
            ee_target_local[:, 2].clamp_(_WS_Z[0], _WS_Z[1])

            # 2b. Accumulate per-env EE orientation via quaternion composition.
            # Convert (Rx, Ry) deltas → delta quaternion, compose into running target.
            # Using quat representation avoids gimbal lock and makes phase-sync trivial.
            delta_rxry = torch.zeros(env.num_envs, 2, device=env.device)
            if len(alice_indices) > 0:
                delta_rxry[alice_indices] = a_rxry
            if len(bob_indices) > 0:
                delta_rxry[bob_indices]   = b_rxry
            with profiler.section("orient_accum"):
                _rx, _ry = delta_rxry[:, 0], delta_rxry[:, 1]
                _qx = torch.stack([torch.cos(_rx/2), torch.sin(_rx/2),
                                    torch.zeros_like(_rx), torch.zeros_like(_rx)], dim=1)
                _qy = torch.stack([torch.cos(_ry/2), torch.zeros_like(_ry),
                                    torch.sin(_ry/2), torch.zeros_like(_ry)], dim=1)
                _q_delta = _quat_mul(_qy, _qx)                               # Ry then Rx
                ee_target_quat_w = _quat_mul(ee_target_quat_w, _q_delta)
                ee_target_quat_w = ee_target_quat_w / ee_target_quat_w.norm(dim=-1, keepdim=True)

            # 3. TCP offset: cuRobo targets wrist_3_link; finger-midpoint is ~5 cm ahead.
            #    Subtract the live offset so the gripper tip arrives at the intended target.
            #    As orientation changes, the live finger positions track the arc automatically.
            lf_w  = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
            rf_w  = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
            w3_w  = _robot_scene.data.body_pos_w[:, _wrist3_ids[0]]
            tcp_offset = (lf_w + rf_w) / 2.0 - w3_w                   # (N,3) world frame
            ik_target  = ee_target_local - tcp_offset                   # subtract in local

            # 4. cuRobo batch IK for all N envs simultaneously.
            # Seed from _prev_joint_cmd (what we commanded) not physics cur_joints
            # so cuRobo finds solutions along the already-smoothed trajectory.
            cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]     # (N,6)
            with profiler.section("curobo_ik"):
                _ik_result = ik_solver.solve_batch(
                    CuroboPose(position=ik_target,
                               quaternion=ee_target_quat_w.contiguous()),
                    seed_config=_prev_joint_cmd.unsqueeze(1),   # (N,1,6)
                    retract_config=_prev_joint_cmd,              # (N,6)
                )

            _ik_success = _ik_result.success.squeeze(-1)               # (N,) bool
            _ik_fail    = ~_ik_success
            
            # 1. Initialize an empty tensor here
            _fail_active = torch.tensor([], dtype=torch.long, device=env.device)

            # Track failure rate for active envs only
            _active = torch.cat([alice_indices, bob_indices]) if (
                len(alice_indices) + len(bob_indices) > 0
            ) else torch.tensor([], dtype=torch.long, device=env.device)
            
            if len(_active) > 0:
                _ik_fail_steps  += int(_ik_fail[_active].sum().item())
                _ik_total_steps += len(_active)
                
                # 2. Populate it safely
                _fail_active = _active[_ik_fail[_active]]

            # Revert EE target accumulator for envs where IK failed so drift
            # doesn't compound across successive failures.
            if len(_fail_active) > 0:
                ee_target_local[_fail_active]  = _ee_target_local_pre[_fail_active]
                ee_target_quat_w[_fail_active] = _ee_target_quat_w_pre[_fail_active]

            # 5. Build 7D joint-position command with EMA smoothing.
            # Blend new IK solution toward previous command so motion feels like
            # RMPFlow's integrated velocity rather than discrete position jumps.
            solved   = _ik_result.solution.view(env.num_envs, 6)
            raw_cmd  = torch.where(_ik_success.unsqueeze(-1), solved, cur_joints)
            smoothed = _JC_ALPHA * raw_cmd + (1.0 - _JC_ALPHA) * _prev_joint_cmd
            _prev_joint_cmd = smoothed.detach().clone()
            env_full = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
            env_full[:, :6] = smoothed
            if len(alice_indices) > 0:
                env_full[alice_indices, 6] = alice_gripper_state[alice_indices].squeeze(-1)
            if len(bob_indices) > 0:
                env_full[bob_indices,   6] = bob_gripper_state[bob_indices].squeeze(-1)

            # ── Alice IK fail: lock arm in place (no dangerous move) ──────────────
            _alice_ik_fail_ids = torch.tensor([], dtype=torch.long, device=env.device)
            if len(alice_indices) > 0:
                _a_ikf = _ik_fail[alice_indices]
                if _a_ikf.any():
                    _alice_ik_fail_ids = alice_indices[_a_ikf]
                    env_full[_alice_ik_fail_ids, :6] = cur_joints[_alice_ik_fail_ids]
                    env_full[_alice_ik_fail_ids, 6]  = 1.0  # gripper open


            # Capture phase BEFORE env.step so we can detect transitions afterwards
            # and log the correct step count before the wrapper resets it to 0.
            _prev_phase   = env.episode_manager.current_phase.clone()
            _prev_steps   = env.episode_manager.phase_step.clone()
            _prev_initial = env.episode_manager.initial_states.clone()  # snapshot before transitions

            # ── STEP ──────────────────────────────────────────────────────────────
            with profiler.section("env_step"):
                obs_full, rewards, dones, truncated, extras = env.step(env_full)

            # ── Alice IK fail: terminate episode with -1 penalty ─────────────────
            if len(_alice_ik_fail_ids) > 0:
                rewards[_alice_ik_fail_ids] += -1.0
                dones[_alice_ik_fail_ids] = True
                if "alice_total_reward" in extras:
                    extras["alice_total_reward"][_alice_ik_fail_ids] = -1.0
                if "alice_failed_this_step" in extras:
                    extras["alice_failed_this_step"][_alice_ik_fail_ids] = True
                env.delayed_alice_reward[_alice_ik_fail_ids] = -1.0
                env.episode_manager.reset_episode(_alice_ik_fail_ids, reason="Alice IK Failure")
                env.hide_goal_ghost(_alice_ik_fail_ids)
                if args.debug_rewards:
                    _dbg_rew_acc[_alice_ik_fail_ids] += -1.0

            # Centralized sync for IK targets and commands to prevent death loops.
            # We MUST sync after env.step() so PhysX readbacks are fresh.
            _phase_changed    = (_prev_phase != env.episode_manager.current_phase)
            _alice_just_ended = _phase_changed & is_alice   
            _bob_just_ended   = _phase_changed & is_bob     
            
            # Identify ALL envs that need a target sync (Phase changes and Terminations)
            needs_sync = _phase_changed | dones.bool()

            if needs_sync.any():
                _sync_ids = torch.where(needs_sync)[0]
                
                # Refresh targets to the current (now clean) physics state using TCP
                _lf_w_sync = _robot_scene.data.body_pos_w[_sync_ids, _lf_ids[0]]
                _rf_w_sync = _robot_scene.data.body_pos_w[_sync_ids, _rf_ids[0]]
                _tcp_w_sync = (_lf_w_sync + _rf_w_sync) / 2.0
                
                ee_target_local[_sync_ids] = (
                    _tcp_w_sync - env.env.scene.env_origins[_sync_ids]
                )
                ee_target_local[_sync_ids, 0] += _EE_HOME_X_OFFSET
                ee_target_local[_sync_ids, 1]  = _EE_HOME_Y
                ee_target_local[_sync_ids, 2]  = _EE_HOME_Z
                _prev_joint_cmd[_sync_ids] = _robot_scene.data.joint_pos[_sync_ids][:, _arm_jids]

                # Snap orientation back to base tool-down for the new episode/phase
                ee_target_quat_w[_sync_ids] = _QUAT_TOOL_DOWN.to(env.device).expand(len(_sync_ids), -1).clone()

            if len(bob_indices) > 0:
                bob_rew_buf.extend(rewards[bob_indices].cpu().numpy().tolist())

            # Accumulate per-env dense rewards and IK fails for phase-end summary
            if args.debug_rewards:
                if len(alice_indices) > 0:
                    _dbg_rew_acc[alice_indices]  += rewards[alice_indices]
                    _dbg_ik_fails[alice_indices] += _ik_fail[alice_indices].long()
                if len(bob_indices) > 0:
                    _dbg_ik_bob[bob_indices] += _ik_fail[bob_indices].long()

            ep_info = extras.get("episode_manager", {})
            if ep_info:
                finished_bob = torch.where(ep_info["bob_done_this_step"])[0]
                if len(finished_bob) > 0:
                    iter_sr_counts[0] += len(finished_bob)
                    iter_sr_counts[1] += int(ep_info["bob_success_this_step"][finished_bob].sum().item())

            # ── ALICE STORAGE ──────────────────────────────────────────────────────
            a_lp_full    = torch.zeros(env.num_envs, device=env.device)
            a_val_full   = torch.zeros(env.num_envs, 1, device=env.device)
            a_mu_full    = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_sigma_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_lp_full[alice_indices]  = a_logprob_active
            a_val_full[alice_indices] = a_val_active
            a_mu_full[alice_indices]  = a_mu_active
            a_sigma_full[alice_indices] = a_sigma_active

            if alice_hidden is not None:
                # Reset Alice's memory if the episode ended OR if her phase just transitioned to Bob
                reset_alice = (dones[alice_indices]) | (_alice_just_ended[alice_indices])
                done_alice = alice_indices[reset_alice]
                if len(done_alice) > 0:
                    alice_hidden[0][done_alice] = 0.0
                    alice_hidden[1][done_alice] = 0.0
            
            # Reset gripper state to OPEN (1.0) on phase boundary/reset
            reset_alice_ids = alice_indices[(dones[alice_indices]) | (_alice_just_ended[alice_indices])]
            alice_gripper_state[reset_alice_ids] = 1.0

            a_masks = torch.zeros(env.num_envs, 1, device=env.device)
            a_masks[alice_indices[~dones[alice_indices]]] = 1.0

            next_alice_obs = obs_full[:, : env.alice_obs_dim]
            with profiler.section("alice_store"):
                alice_ppo.storage.add_transitions(
                    current_alice_obs, next_alice_obs, a_policy,
                    rewards, dones, a_val_full, a_lp_full, a_mu_full, a_sigma_full, a_masks,
                )
            current_alice_obs = next_alice_obs

            # ── BOB STORAGE ────────────────────────────────────────────────────────
            b_lp_full    = torch.zeros(env.num_envs, 1, device=env.device)
            b_val_full   = torch.zeros(env.num_envs, 1, device=env.device)
            b_mu_full    = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_sigma_full = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_lp_full[bob_indices]    = b_logprob_active.unsqueeze(1)
            b_val_full[bob_indices]   = b_val_active
            b_mu_full[bob_indices]    = b_mu_active
            b_sigma_full[bob_indices] = b_sigma_active

            bob_done_this_step = ep_info.get(
                "bob_done_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
            ended_for_bob = dones | bob_done_this_step
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob[bob_indices]]] = 1.0

            if bob_hidden is not None:
                # Reset Bob's memory if the episode ended OR if his phase just transitioned to Alice
                reset_bob = (dones[bob_indices]) | (_bob_just_ended[bob_indices])
                done_bob = bob_indices[reset_bob]
                if len(done_bob) > 0:
                    bob_hidden[0][done_bob] = 0.0
                    bob_hidden[1][done_bob] = 0.0
            
            # Reset gripper state to OPEN (1.0) on phase boundary/reset
            reset_bob_ids = bob_indices[(dones[bob_indices]) | (_bob_just_ended[bob_indices])]
            bob_gripper_state[reset_bob_ids] = 1.0

            next_bob_obs = obs_full[:, env.alice_obs_dim :]
            with profiler.section("bob_store"):
                bob_ppo.storage.add_transitions(
                    current_bob_obs, next_bob_obs, b_policy,
                    rewards, dones, b_val_full, b_lp_full, b_mu_full, b_sigma_full, b_masks,
                )
            current_bob_obs = next_bob_obs

            # ── ALICE OUTCOME & ABC BUFFER ─────────────────────────────────────────
            if ep_info:
                bob_dones_now    = ep_info.get("bob_done_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                alice_dones_now  = extras.get("alice_failed_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                alice_rewards_now = extras.get("alice_total_reward", torch.zeros(env.num_envs, device=env.device))
                goal_valid       = ep_info.get("goal_valid", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                bob_success      = ep_info.get("bob_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

                # ── PHASE-END SUMMARY (debug_rewards mode) ────────────────────────
                if args.debug_rewards:
                    _bob_pos_err = ep_info.get("bob_pos_err", torch.zeros(env.num_envs, device=env.device))
                    _bob_rot_err = ep_info.get("bob_rot_err", torch.zeros(env.num_envs, device=env.device))
                    _gstates     = ep_info.get("goal_states")

                    # Alice phase end — fire on phase transition OR failure
                    _alice_log_trigger = _alice_just_ended | extras.get("alice_failed_this_step", torch.zeros_like(_alice_just_ended))
                    for _gi in torch.where(_alice_log_trigger)[0].tolist():
                        _valid = "VALID" if goal_valid[_gi].item() else "invalid"
                        _rew   = alice_rewards_now[_gi].item()
                        _dense = _dbg_rew_acc[_gi].item()
                        _ik_f  = _dbg_ik_fails[_gi].item()
                        _step  = _prev_steps[_gi].item()
                        _start_pos = _prev_initial[_gi, 0:3].tolist()
                        _goal_pos  = (_gstates[_gi, 0:3].tolist() if _gstates is not None else [0,0,0])
                        _start_ori = _prev_initial[_gi, 3:6].tolist()
                        _goal_ori  = (_gstates[_gi, 3:6].tolist() if _gstates is not None else [0,0,0])
                        print(
                            f"  [ALICE END | iter={bob_updates} env={_gi}]"
                            f"  goal={_valid}"
                            f"  start=({_start_pos[0]:.3f},{_start_pos[1]:.3f},{_start_pos[2]:.3f})"
                            f"  ori_start=({_start_ori[0]:.3f},{_start_ori[1]:.3f},{_start_ori[2]:.3f})"
                            f"  →  final=({_goal_pos[0]:.3f},{_goal_pos[1]:.3f},{_goal_pos[2]:.3f})"
                            f"  ori_final=({_goal_ori[0]:.3f},{_goal_ori[1]:.3f},{_goal_ori[2]:.3f})"
                            f"  phase_rew={_rew:+.3f}  dense_acc={_dense:+.3f}  total={_rew+_dense:+.3f}"
                            f"  ik_fails={_ik_f}  steps={_step}",
                            flush=True,
                        )
                        _dbg_rew_acc[_gi]  = 0.0
                        _dbg_ik_fails[_gi] = 0
                        _dbg_printed[_gi]  = True

                    # Bob phase end — fire on phase transition
                    for _gi in torch.where(_bob_just_ended)[0].tolist():
                        _succ = "SUCCESS" if bob_success[_gi].item() else "fail"
                        _gv   = "valid_goal" if goal_valid[_gi].item() else "no_goal"
                        _pos  = _bob_pos_err[_gi].item()
                        _rot  = _bob_rot_err[_gi].item()
                        _step = _prev_steps[_gi].item()
                        _alice_rew_at_bob_end = alice_rewards_now[_gi].item()
                        _start_pos = _prev_initial[_gi, 0:3].tolist()
                        _start_ori = _prev_initial[_gi, 3:6].tolist()
                        _goal_pos  = (_gstates[_gi, 0:3].tolist() if _gstates is not None else [0,0,0])
                        _goal_ori  = (_gstates[_gi, 3:6].tolist() if _gstates is not None else [0,0,0])
                        _ik_b      = _dbg_ik_bob[_gi].item()
                        print(
                            f"  [BOB END   | iter={bob_updates} env={_gi}]"
                            f"  {_succ}  ({_gv})"
                            f"  start=({_start_pos[0]:.3f},{_start_pos[1]:.3f},{_start_pos[2]:.3f})"
                            f"  ori_start=({_start_ori[0]:.3f},{_start_ori[1]:.3f},{_start_ori[2]:.3f})"
                            f"  →  goal=({_goal_pos[0]:.3f},{_goal_pos[1]:.3f},{_goal_pos[2]:.3f})"
                            f"  ori_goal=({_goal_ori[0]:.3f},{_goal_ori[1]:.3f},{_goal_ori[2]:.3f})"
                            f"  pos_err={_pos:.4f}m  rot_err={_rot:.4f}rad"
                            f"  steps={_step}  ik_fails={_ik_b}"
                            f"  alice_payout={_alice_rew_at_bob_end:+.3f}",
                            flush=True,
                        )
                        _dbg_ik_bob[_gi] = 0

                profiler.mark_start("reward_backfill")
                if bob_dones_now.any() or alice_dones_now.any() or (alice_rewards_now != 0).any():
                    filled = alice_ppo.storage.step
                    if filled > 0:
                        masks = alice_ppo.storage.masks[:filled, :, 0]
                        row_idx = torch.arange(filled, device=env.device).unsqueeze(1).expand_as(masks)
                        last_valid_rows = torch.where(masks.bool(), row_idx, torch.tensor(-1, device=env.device)).max(dim=0).values
                        has_valid = last_valid_rows >= 0
                        rewarded_envs   = torch.where(alice_rewards_now != 0)[0]
                        valid_rewarded  = rewarded_envs[has_valid[rewarded_envs]]
                        if len(valid_rewarded) > 0:
                            rows = last_valid_rows[valid_rewarded]
                            alice_ppo.storage.rewards[rows, valid_rewarded, 0] += alice_rewards_now[valid_rewarded]
                            alice_rew_buf.extend(alice_rewards_now[valid_rewarded].cpu().numpy().tolist())
                profiler.mark_stop("reward_backfill")

                profiler.mark_start("abc_buffer")
                just_failed_bob = bob_dones_now & (~bob_success) & goal_valid
                valid_ids = torch.where(just_failed_bob)[0]
                min_demo_steps = max(10, env.episode_manager.alice_timesteps // 2)
                valid_trajs = []
                for env_id in valid_ids:
                    eid   = env_id.item()
                    t_len = alice_traj_len[eid].item()
                    if t_len < min_demo_steps:
                        continue
                    traj_o = alice_traj_obs[eid, :t_len]
                    traj_a = alice_traj_act[eid, :t_len]
                    g = ep_info["goal_states"][eid].unsqueeze(0).expand(t_len, -1)
                    bc_obs = env.construct_bob_observation(traj_o, g)
                    valid_trajs.append((bc_obs, traj_a))
                if valid_trajs:
                    all_obs  = torch.cat([t[0] for t in valid_trajs], dim=0)
                    all_acts = torch.cat([t[1] for t in valid_trajs], dim=0)
                    with torch.no_grad():
                        old_lp_all, _, _, _, _ = bob_ppo.actor_critic.evaluate(all_obs, None, all_acts)
                    offset = 0
                    for bc_obs_i, traj_a_i in valid_trajs:
                        t_len_i = bc_obs_i.shape[0]
                        bob_ppo.abc_buffer.add_trajectory(
                            bc_obs_i, traj_a_i, old_lp_all[offset:offset + t_len_i]
                        )
                        offset += t_len_i
                profiler.mark_stop("abc_buffer")

        current_sr = iter_sr_counts[1] / max(1, iter_sr_counts[0])
        bob_success_buf.append(current_sr)

        # ── IK diagnostics (logged with rest of iter stats below) ─────────────────
        _ik_fr = _ik_fail_steps / max(1, _ik_total_steps)
        print(f"  [cuRobo] IK fail rate: {_ik_fr:.4f} ({_ik_fail_steps}/{_ik_total_steps} active steps)", flush=True)

        # ── Entropy / LR / ABC controllers ────────────────────────────────────────
        # Fix 2: entropy_coef fixed at paper value (Table 2: 0.01); removed SR-coupled PI controller
        writer.add_scalar("Alice/EntropyCoef", alice_ppo.entropy_coef, bob_updates)
        print(f"  [Alice] Entropy Coef: {alice_ppo.entropy_coef:.4f} (fixed)", flush=True)

        _alice_lr_max = alice_ppo.learning_rate
        _alice_lr_min = ppo_cfg["params"]["learn"].get("alice_lr_min", 5e-5)
        _lr_p  = min(1.0, bob_updates / args.max_iterations)
        _alice_lr = _alice_lr_min + 0.5 * (_alice_lr_max - _alice_lr_min) * (1.0 + math.cos(math.pi * _lr_p))
        for pg in alice_ppo.optimizer.param_groups:
            pg["lr"] = _alice_lr
        writer.add_scalar("Alice/LearningRate", _alice_lr, bob_updates)

        # Fix 1: abc_coef fixed at paper value (Table 2: β=0.5); removed SR-coupling and anneal schedule
        bob_ppo.abc_coef = ppo_cfg["params"]["learn"].get("abc_coef", 0.5)
        writer.add_scalar("Bob/ABCCoef", bob_ppo.abc_coef, bob_updates)
        print(f"  [Bob] ABCCoef: {bob_ppo.abc_coef:.4f} (fixed)", flush=True)

        with profiler.section("alice_update"):
            perform_alice_update()
        with profiler.section("bob_update"):
            perform_bob_update(current_bob_obs)

        # ── Emergency checkpoint on SIGTERM ────────────────────────────────────────
        if _shutdown_requested:
            _ckpt_iter = bob_updates
            print(f"[INFO] Emergency checkpoint at iteration {_ckpt_iter}", flush=True)
            bob_ppo.save(os.path.join(bob_ppo.log_dir, f"model_{_ckpt_iter}.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{_ckpt_iter}.pt"))
            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
            torch.save(env.episode_manager.state_dict(),
                       os.path.join(bob_ppo.log_dir, f"episode_manager_{_ckpt_iter}.pt"))
            torch.save({"entropy_coef": alice_ppo.entropy_coef, "abc_coef": bob_ppo.abc_coef,
                        "bob_success_buf": list(bob_success_buf)},
                       os.path.join(bob_ppo.log_dir, f"train_state_{_ckpt_iter}.pt"))
            print("[INFO] Emergency checkpoint saved — exiting cleanly.", flush=True)
            break

        # ── Periodic checkpoint ────────────────────────────────────────────────────
        if args.save_interval > 0 and bob_updates % args.save_interval == 0:
            bob_ppo.save(os.path.join(bob_ppo.log_dir, f"model_{bob_updates}.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, f"model_{bob_updates}.pt"))
            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
            torch.save(env.episode_manager.state_dict(),
                       os.path.join(bob_ppo.log_dir, f"episode_manager_{bob_updates}.pt"))
            torch.save({"entropy_coef": alice_ppo.entropy_coef, "abc_coef": bob_ppo.abc_coef,
                        "bob_success_buf": list(bob_success_buf)},
                       os.path.join(bob_ppo.log_dir, f"train_state_{bob_updates}.pt"))

        # ── Iteration summary ──────────────────────────────────────────────────────
        _stats    = env.get_iter_stats() if hasattr(env, "get_iter_stats") else {}
        _term_str = (
            "  ".join(f"{k}={v}" for k, v in sorted(_stats.get("terminations", {}).items()))
            or "none"
        )
        _valid_goals   = _stats.get("valid_goals", 0)
        _invalid_goals = _stats.get("invalid_goals", 0)
        _bob_succ      = _stats.get("bob_successes", 0)
        _bob_fail      = _stats.get("bob_failures", 0)
        writer.add_scalar("Metrics/Alice/ValidGoals",   _valid_goals,   bob_updates)
        writer.add_scalar("Metrics/Alice/InvalidGoals", _invalid_goals, bob_updates)
        _goal_validity_rate = _valid_goals / max(1, _valid_goals + _invalid_goals)
        writer.add_scalar("Metrics/Alice/GoalValidityRate", _goal_validity_rate, bob_updates)
        _alice_disp_3d_sum = _stats.get("alice_disp_3d_sum", 0.0)
        _alice_total_phases = _stats.get("alice_total", 1)
        _mean_disp_3d = _alice_disp_3d_sum / max(1, _alice_total_phases)
        writer.add_scalar("Metrics/Alice/MeanDisp3D", _mean_disp_3d, bob_updates)
        _abc_buf_size = bob_ppo.abc_buffer.size
        writer.add_scalar("Metrics/ABC/BufferSize", _abc_buf_size, bob_updates)
        _abc_warm = 1.0 if bob_ppo.abc_buffer.size > 0 else 0.0
        writer.add_scalar("Metrics/ABC/IsWarm", _abc_warm, bob_updates)
        writer.add_scalar("Metrics/Alice/EMAReward", ema_alice_rew, bob_updates)
        writer.add_scalar("Metrics/IKFailRate", _ik_fr, bob_updates)
        _ik_overhead = profiler.get_section_frac("curobo_ik", relative_to="env_step")
        writer.add_scalar("Metrics/IKOverheadFrac", _ik_overhead, bob_updates)
        if _ik_overhead > 0.10:
            print(
                f"  [cuRobo] WARNING: IK overhead {_ik_overhead:.1%} > 10% of env_step "
                f"— consider reducing num_envs or upgrading GPU.",
                flush=True,
            )

        print(
            f"[Iter {bob_updates}] SR={current_sr:.2f} | IK_fail={_ik_fr:.3f} | "
            f"Goals valid={_valid_goals} invalid={_invalid_goals} | "
            f"Bob succ={_bob_succ} fail={_bob_fail} | "
            f"Terminations: {_term_str} | "
            f"ABC buf: {_abc_buf_size} | "
            f"ABC warm: {'YES' if _abc_warm else 'NO'}",
            flush=True,
        )
        _alice_total = _stats.get("alice_total", 0)
        if _alice_total > 0:
            _avg_3d = _stats["alice_disp_3d_sum"] / _alice_total
            _avg_xy = _stats["alice_disp_xy_sum"] / _alice_total
            _max_xy = _stats["alice_disp_xy_max"]
            _avg_y  = _stats.get("alice_disp_y_sum", 0.0) / _alice_total
            _avg_z  = _stats.get("alice_disp_z_sum", 0.0) / _alice_total
            _not_mv = _stats["alice_not_moved"]
            _pos_req = getattr(env, "_ALICE_POS_REQ", 0.05)
            print(
                f"  [AliceDisp] {_valid_goals}/{_alice_total} valid | "
                f"avg 3D={_avg_3d:.3f}m  avg XY={_avg_xy:.3f}m  max XY={_max_xy:.3f}m  avg Y={_avg_y:.3f}m  avg Z={_avg_z:.3f}m | "
                f"not-moved(≤{_pos_req:.2f}m): {_not_mv}/{_alice_total}",
                flush=True,
            )
        profiler.end_iteration(bob_updates)

    # ── End of training ────────────────────────────────────────────────────────────
    profiler.print_summary()
    alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_final.pt"))
    bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_final.pt"))
    bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
    torch.save(env.episode_manager.state_dict(),
               os.path.join(bob_ppo.log_dir, "episode_manager_final.pt"))
    torch.save({"entropy_coef": alice_ppo.entropy_coef, "abc_coef": bob_ppo.abc_coef,
                "bob_success_buf": list(bob_success_buf)},
               os.path.join(bob_ppo.log_dir, "train_state_final.pt"))
    print("  ✓ Saved final models")
    writer.close()

    print(f"\n{'='*80}\nTRAINING COMPLETE ({bob_updates} iterations)\n{'='*80}")
    print(f"To resume:\n  python train_curobo.py --exp_name {args.exp_name} \\")
    print(f"    --chkpt_alice runs/{args.exp_name}/alice/model_final.pt \\")
    print(f"    --chkpt_bob   runs/{args.exp_name}/bob/model_final.pt \\")
    print(f"    --resume_iteration {bob_updates}\n{'='*80}\n")


if __name__ == "__main__":
    main()