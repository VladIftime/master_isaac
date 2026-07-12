"""
Push-ASP PBRS Model I training script (T-block + SE(2) d_pose + time-based ASP + Bob penalty).

Same as Model G but adds a Bob time penalty R_B += -gamma_sp * t_B,
implementing the full symmetric Sukhbaatar et al. 2018 reward structure:
  R_A = gamma_sp * max(0, t_B - t_A)    (Alice reward)
  R_B = -gamma_sp * t_B                 (Bob penalty)
where t_A = Alice's push count, t_B = Bob's push count, gamma_sp = 0.5.

This creates a near-zero-sum competitive game: Alice gains from proposing goals
that Bob takes many pushes to solve; Bob is penalized for inefficiency, adding
urgency to solve quickly. The two time-based rewards together form the canonical
Sukhbaatar self-play structure.

Shallow goal penalty removed (time-based reward makes it redundant).
ABC disabled. Bob reward structure unchanged (PBRS d_pose + sparse completion),
minus the time penalty at phase end.

Architecture:
  - Alice: PI-encoder + LSTM, 4D x 21-bin MultiCategorical, no GoalEncoder
  - Bob:   PI-encoder + GoalEncoder + LSTM, 4D x 21-bin MultiCategorical
  - Bob rewards: PBRS dense reward using single d_pose potential + sparse completion
                 - gamma_sp * t_B (time penalty at phase end)
  - Alice rewards: validation (+1 valid, -3 invalid) + time-based outcome

Run locally:
  python -m asyncDualPlayPPO.train_i_tasp_dpose_bobpen --num_envs 16 --max_iterations 500 --exp_name push_pbrs_tasp_dpose_bp_test --headless
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
        "  Check: apptainer exec --nv isaac-lab.sif python -c 'import curobo'\n"
    )
    import sys
    sys.exit(1)

import torch
import torch._dynamo     # noqa: F401
import torch._C          # noqa: F401
import torch.optim       # noqa: F401

import isaaclab.app
from isaaclab.app import AppLauncher

import os
import signal
import sys
import yaml
import math
import argparse
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25,  0.70)
_WS_Z = (0.25, 0.55)


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


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train Push-Primitive ASP with time-based reward + d_pose + Bob penalty (PBRS Model I)"
    )
    parser.add_argument("--exp_name", type=str, default="push_pbrs_tasp_dpose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--chkpt_alice", type=str, default=None)
    parser.add_argument("--chkpt_bob", type=str, default=None)
    parser.add_argument("--resume_iteration", type=int, default=0)
    parser.add_argument("--alice_pushes", type=int, default=5,
                        help="Number of push macro-actions Alice gets per phase")
    parser.add_argument("--bob_pushes", type=int, default=10,
                        help="Number of push macro-actions Bob gets per phase")
    parser.add_argument("--max_goals_per_episode", type=int, default=2,
                        help="Number of Alice-Bob cycles per episode")
    parser.add_argument("--no_abc", action="store_true",
                        help="Disable ABC — recommended for push-primitive ASP")
    parser.add_argument("--no_hist_pool", action="store_true",
                        help="Disable historical policy pool")
    parser.add_argument("--debug_rewards", action="store_true")
    parser.add_argument("--char_length", type=float, default=0.07,
                        help="SE(2) characteristic length L (radians to metres)")
    parser.add_argument("--dpose_threshold", type=float, default=0.055,
                        help="d_pose success threshold (metres)")
    parser.add_argument("--alice_reward_scale", type=float, default=0.5,
                        help="Sukhbaatar gamma_sp: R_A = gamma_sp * max(0, t_B - t_A)")
    parser.add_argument("--bob_time_penalty", action="store_true", default=True,
                        help="Apply Bob time penalty R_B += -gamma_sp * t_B")
    parser.add_argument("--no_bob_time_penalty", action="store_false", dest="bob_time_penalty",
                        help="Disable Bob time penalty (revert to Model G behavior)")
    parser.add_argument("--bob_time_penalty_scale", type=float, default=0.5,
                        help="gamma_sp for Bob time penalty (default matches alice_reward_scale)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 50:
        args.debug_rewards = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import numpy as np
    import copy
    import gymnasium as gym_mc
    from torch.utils.tensorboard import SummaryWriter

    import random as _random
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    _random.seed(args.seed)
    print(f"[SEED] RNG seeded with {args.seed}")

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push_asp import PushASPEnvWrapper
    from asyncDualPlayPPO.tasks.utils.wrapper_push_asp import _OBS_ROBOT_DIM
    from asyncDualPlayPPO.tasks.utils.action_push import (
        decode_push_action, compute_push_waypoints,
    )
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative,
        TBLOCK_MIN_R, TBLOCK_MAX_R,
    )
    from asyncDualPlayPPO.tasks.utils.events import (
        reset_objects_to_random_safe_pose, reset_robot_joints,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import GPUDemonstrationBuffer
    from asyncDualPlayPPO.utils.historical_pool import HistoricalPolicyPool
    from asyncDualPlayPPO.tasks.utils.reward_pbrs import (
        potential_dpose, compute_dpose, compute_pbrs_reward_dpose,
        PBRS_K_DPOSE, PBRS_W_DPOSE,
        TIP_OVER_THRESHOLD, PBRS_COMPLETION_BONUS,
        PBRS_CATASTROPHE_PENALTY,
    )

    ppo_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    ppo_cfg = load_cfg(ppo_cfg_path)

    alice_pushes = args.alice_pushes
    bob_pushes = args.bob_pushes
    max_goals_per_episode = args.max_goals_per_episode

    print(
        f"[Config] Push-ASP PBRS-I (time-based + d_pose + Bob penalty): alice_pushes={alice_pushes}, bob_pushes={bob_pushes}, "
        f"max_goals={max_goals_per_episode}, L={args.char_length}, thr={args.dpose_threshold}, "
        f"gamma_sp={args.alice_reward_scale}, "
        f"bob_penalty={'ON (gamma=' + str(args.bob_time_penalty_scale) + ')' if args.bob_time_penalty else 'OFF'}, "
        f"ABC={'OFF' if args.no_abc else 'ON'}"
    )

    num_cat_dims = 4
    num_bins = 21
    use_lstm = True
    print(f"[Config] Push action space: {num_cat_dims}D x {num_bins} bins")

    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )

    print("Creating environment...")
    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)

    env = PushASPEnvWrapper(
        env=base_env,
        alice_pushes=alice_pushes,
        bob_pushes=bob_pushes,
        max_goals_per_episode=max_goals_per_episode,
        num_objects=1,
        dpose_obs=True,
        char_length=args.char_length,
        dpose_threshold=args.dpose_threshold,
        time_based_alice=True,
        alice_reward_scale=args.alice_reward_scale,
        device=base_env.device,
    )
    print("Environment ready.")

    print("[cuRobo] Initialising IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(env.device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    _ik_config.solver.newton_optimizer.n_iters = 30
    _ik_config.solver.newton_optimizer.inner_iters = 10
    ik_solver = IKSolver(_ik_config)
    print("[cuRobo] IK solver created.")

    print(f"[cuRobo] Warming up CUDA graph for N={env.num_envs} envs...")
    _wup_pos = torch.zeros(env.num_envs, 3, device=env.device)
    _wup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device,
                              dtype=torch.float32).expand(env.num_envs, 4)
    ik_solver.solve_batch(
        CuroboPose(position=_wup_pos, quaternion=_wup_quat),
        seed_config=torch.zeros(env.num_envs, 1, 6, device=env.device),
        retract_config=torch.zeros(env.num_envs, 6, device=env.device),
    )
    print("[cuRobo] Warm-up done.")

    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)

    _blk_dir = os.path.join(os.path.dirname(__file__), "assets/blocks")
    _goal_viz = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarker",
            markers={
                "tblock": UsdFileCfg(
                    usd_path=os.path.join(_blk_dir, "t_shape.usda"),
                    scale=(2.0, 2.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),
                ),
            },
        )
    )
    _ident_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    _push_viz_start = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushStart",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
    )
    _push_viz_end = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushEnd",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
    )
    _push_viz_arrow = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushArrow",
            markers={
                "cylinder": sim_utils.CylinderCfg(
                    radius=0.005, height=0.30,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
            },
        )
    )

    def _euler_xyz_to_quat_local(euler: torch.Tensor) -> torch.Tensor:
        roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack([w, x, y, z], dim=-1)

    def _update_goal_markers():
        gs = env.episode_manager.goal_states
        N = env.num_envs
        origins = env.env.scene.env_origins
        pos = torch.zeros(N, 3, device=env.device)
        pos[:, 2] = -1.0
        quat = _QUAT_TOOL_DOWN.expand(N, 4).clone()
        if gs is not None:
            is_bob = env.episode_manager.is_bob_phase()
            bob_ids = torch.where(is_bob)[0]
            if len(bob_ids) > 0:
                pos[bob_ids, :2] = gs[bob_ids, :2]
                pos[bob_ids, 2] = 0.001
                euler = gs[bob_ids, 3:6].clone()
                euler[:, 0] = 0.0
                euler[:, 1] = 0.0
                quat[bob_ids] = _euler_xyz_to_quat_local(euler)
        _goal_viz.visualize(translations=pos + origins, orientations=quat)

    def _update_push_markers(Xs, Ys, length, theta):
        try:
            N = Xs.shape[0]
            origins = env.env.scene.env_origins
            z_table = 0.002
            ident = _ident_quat.to(env.device).expand(N, 4)

            Xf = Xs + length * torch.cos(theta)
            Yf = Ys + length * torch.sin(theta)

            start_pos = torch.stack([Xs, Ys, torch.full((N,), z_table, device=env.device)], dim=-1) + origins
            end_pos = torch.stack([Xf, Yf, torch.full((N,), z_table, device=env.device)], dim=-1) + origins
            _push_viz_start.visualize(translations=start_pos, orientations=ident)
            _push_viz_end.visualize(translations=end_pos, orientations=ident)

            mid_w = torch.stack([
                (Xs + Xf) / 2, (Ys + Yf) / 2,
                torch.full((N,), z_table, device=env.device),
            ], dim=-1) + origins
            half = math.pi / 4
            ch, sh = math.cos(half), math.sin(half)
            arrow_quat = torch.stack([
                torch.full((N,), ch, device=env.device),
                -sh * torch.sin(theta),
                sh * torch.cos(theta),
                torch.zeros(N, device=env.device),
            ], dim=-1)
            _push_viz_arrow.visualize(translations=mid_w, orientations=arrow_quat)
        except Exception:
            pass

    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    print("[Setup] Calibrating IK-physics error...")
    _calib_pos = torch.zeros(env.num_envs, 3, device=env.device)
    _calib_pos[:, 1] = 0.60
    _calib_pos[:, 2] = 0.25
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_TOOL_DOWN.expand(env.num_envs, 4)),
        seed_config=_calib_cur.unsqueeze(1),
        retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(env.num_envs, 6)
    _calib_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        env.step(_calib_act)
    _finger_after = _tcp_pos_local()
    _TOTAL_IK_ERROR = (_finger_after - _calib_pos).clone()
    print(
        f"[Setup] IK error = ({float(_TOTAL_IK_ERROR[0,0]):+.3f}, "
        f"{float(_TOTAL_IK_ERROR[0,1]):+.3f}, {float(_TOTAL_IK_ERROR[0,2]):+.3f})"
    )

    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    alice_cfg = copy.deepcopy(ppo_cfg["params"])
    alice_cfg["policy"]["use_pi_encoder"] = True
    alice_cfg["policy"]["use_multicategorical"] = True
    alice_cfg["policy"]["use_lstm"] = True
    alice_cfg["policy"]["use_goal_encoder"] = False
    alice_cfg["policy"]["num_cat_dims"] = num_cat_dims
    alice_cfg["policy"]["num_bins"] = num_bins
    alice_cfg["policy"]["robot_state_dim"] = 6

    alice_ppo = PPO(
        vec_env=env,
        cfg_train=alice_cfg,
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/alice",
        asymmetric=False,
    )
    alice_ppo.observation_space = env.alice_observation_space
    alice_ppo.state_space = alice_ppo.observation_space
    alice_ppo.action_space = _mc_space
    alice_ppo.desired_kl = None

    alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
        alice_ppo.observation_space.shape,
        alice_ppo.state_space.shape,
        alice_ppo.action_space.shape,
        alice_ppo.init_noise_std,
        alice_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    alice_rollout_len = alice_pushes + bob_pushes
    alice_storage_size = max(alice_ppo.num_transitions_per_env + alice_pushes + 5,
                              alice_rollout_len + 5)
    alice_ppo.storage = alice_ppo.storage.__class__(
        alice_ppo.vec_env.num_envs, alice_storage_size,
        alice_ppo.observation_space.shape, alice_ppo.state_space.shape,
        alice_ppo.action_space.shape, alice_ppo.device, "sequential",
    )
    alice_ppo.optimizer = torch.optim.Adam(
        alice_ppo.actor_critic.parameters(), lr=alice_ppo.learning_rate,
    )

    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"]["use_pi_encoder"] = True
    bob_cfg["policy"]["use_multicategorical"] = True
    bob_cfg["policy"]["use_lstm"] = True
    bob_cfg["policy"]["use_goal_encoder"] = True
    bob_cfg["policy"]["num_cat_dims"] = num_cat_dims
    bob_cfg["policy"]["num_bins"] = num_bins
    bob_cfg["policy"]["num_objects"] = 1
    bob_cfg["policy"]["robot_state_dim"] = 6
    bob_cfg["policy"]["goal_embed_dim"] = 8

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
    bob_ppo.action_space = _mc_space
    bob_ppo.desired_kl = None

    bob_ppo.actor_critic = bob_ppo.actor_critic.__class__(
        bob_ppo.observation_space.shape,
        bob_ppo.state_space.shape,
        bob_ppo.action_space.shape,
        bob_ppo.init_noise_std,
        bob_ppo.model_cfg,
        asymmetric=False,
    ).to(env.device)

    if hasattr(bob_ppo.actor_critic, "_goal_proj") and bob_ppo.actor_critic._goal_proj is not None:
        with torch.no_grad():
            bob_ppo.actor_critic._goal_proj.weight.mul_(0.1)

    bob_storage_size = max(bob_ppo.num_transitions_per_env + bob_pushes + 5,
                            alice_rollout_len + 5)
    bob_ppo.storage = bob_ppo.storage.__class__(
        bob_ppo.vec_env.num_envs, bob_storage_size,
        bob_ppo.observation_space.shape, bob_ppo.state_space.shape,
        bob_ppo.action_space.shape, bob_ppo.device, "sequential",
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

    if alice_ppo.actor_critic.use_lstm:
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

    alice_pool = HistoricalPolicyPool(max_size=5)
    bob_pool = HistoricalPolicyPool(max_size=5)
    HIST_SAVE_INTERVAL = 100
    HIST_FRAC = 0.2

    bob_success_buf = deque(maxlen=200)

    # ── Group-consistency guard (self-play checkpoint set is multi-file) ──────
    # A preemption can leave scratch with a mismatched set (e.g. new bob + old
    # alice, or desynced latest_iter). Per-file atomicity does not prevent that.
    # Each checkpoint embeds its iteration; if alice/bob/latest_iter disagree,
    # refuse the mismatched pair and start THIS run fresh (loud warning) rather
    # than silently resuming co-evolved adversaries from different points.
    if (args.chkpt_alice and args.chkpt_bob
            and os.path.isfile(args.chkpt_alice) and os.path.isfile(args.chkpt_bob)):
        def _peek_iter(_p):
            try:
                return torch.load(_p, map_location="cpu").get("iteration")
            except Exception:
                return None
        _ai, _bi = _peek_iter(args.chkpt_alice), _peek_iter(args.chkpt_bob)
        if _ai is not None and _bi is not None and _ai != _bi:
            print(f"[Resume][GUARD] MISMATCHED Alice/Bob checkpoints (alice_iter={_ai} "
                  f"!= bob_iter={_bi}; latest_iter.txt={args.resume_iteration} is advisory) — "
                  f"refusing to resume co-evolved adversaries from different points; "
                  f"starting this run FRESH.", flush=True)
            args.chkpt_alice = None
            args.chkpt_bob = None
            args.resume_iteration = 0

    _resume_best_sr = -1.0  # restored from train_state below if resuming (Fix#2)
    if args.chkpt_alice and os.path.isfile(args.chkpt_alice):
        alice_ppo.load(args.chkpt_alice)
        print(f"[Resume] Loaded Alice from {args.chkpt_alice}")
    if args.chkpt_bob and os.path.isfile(args.chkpt_bob):
        bob_ppo.load(args.chkpt_bob)
        print(f"[Resume] Loaded Bob from {args.chkpt_bob}")
        if bob_ppo.loaded_iteration is not None:
            args.resume_iteration = int(bob_ppo.loaded_iteration)
            print(f"[Resume] Iteration from checkpoint (authoritative): {args.resume_iteration}")
        _abc_buf_path = os.path.join(os.path.dirname(args.chkpt_bob), "abc_buffer.pt")
        if os.path.isfile(_abc_buf_path):
            bob_ppo.abc_buffer.load(_abc_buf_path)
            print(f"[Resume] Loaded ABC buffer ({bob_ppo.abc_buffer.size} entries)")
        _ep_mgr_path = os.path.join(os.path.dirname(args.chkpt_bob), "episode_manager_latest.pt")
        if os.path.isfile(_ep_mgr_path):
            ep_sd = torch.load(_ep_mgr_path, map_location=env.device)
            env.episode_manager.load_state_dict(ep_sd)
            print(f"[Resume] Loaded EpisodeManager state")
        _train_state_path = os.path.join(os.path.dirname(args.chkpt_bob), "train_state_latest.pt")
        if os.path.isfile(_train_state_path):
            _ts = torch.load(_train_state_path, map_location="cpu")
            alice_ppo.entropy_coef = float(_ts.get("entropy_coef", 0.005))
            bob_ppo.abc_coef = float(_ts.get("abc_coef", 0.5))
            bob_success_buf.extend(_ts.get("bob_success_buf", []))
            _resume_best_sr = float(_ts.get("best_bob_success_rate", -1.0))
            print(f"[Resume] Restored train state: entropy_coef={alice_ppo.entropy_coef:.4f}")

    alice_updates = args.resume_iteration
    bob_updates = args.resume_iteration

    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")

    _rollout_len = alice_pushes + bob_pushes
    alice_rew_buf = deque(maxlen=args.num_envs * alice_pushes)
    bob_rew_buf = deque(maxlen=args.num_envs * bob_pushes)
    bob_pos_err_buf = deque(maxlen=args.num_envs * bob_pushes)
    bob_rot_err_buf = deque(maxlen=args.num_envs * bob_pushes)
    bob_pos_sr_buf = deque(maxlen=args.num_envs * bob_pushes)
    bob_rot_sr_buf = deque(maxlen=args.num_envs * bob_pushes)
    # Combined-gate SR (pos<0.05m AND rot<0.2rad) — matches the validation gate (V2 fix).
    bob_combined_sr_buf = deque(maxlen=args.num_envs * bob_pushes)

    best_bob_success_rate = _resume_best_sr
    last_alice_mean_rew = 0.0
    ema_alice_rew = 0.0

    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    print(f"\n{'='*80}\nTRAINING RUN: {args.exp_name}\nLOG DIR: {run_dir}\n{'='*80}\n")

    def perform_alice_update():
        nonlocal alice_updates, last_alice_mean_rew, ema_alice_rew
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
        print(f"  [Alice Update {alice_updates}] Loss: {loss_surr:.4f} | "
              f"Val: {loss_val:.4f} | Rew: {mean_alice_rew:.4f}", flush=True)
        alice_rew_buf.clear()
        alice_updates += 1

    def perform_bob_update(current_bob_obs):
        nonlocal bob_updates, best_bob_success_rate
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
                _robot_dim = 6
                s_t_batch = sample_obs[:, _robot_dim: _robot_dim + 6]
                _gs = _robot_dim + 14
                s_star_batch = sample_obs[:, _gs: _gs + 6]
                g_sample = bob_ppo.actor_critic.goal_encoder(s_star_batch, s_t_batch)
                writer.add_scalar("GoalEncoder/embedding_norm",
                                  g_sample.norm(dim=-1).mean().item(), bob_updates)

        bob_ppo.storage.compute_returns(last_val_b, bob_ppo.gamma, bob_ppo.lam)
        bob_ppo.current_learning_iteration = bob_updates
        loss_val, loss_surr, loss_abc, _ = bob_ppo.update(alice_mean_rew=last_alice_mean_rew)
        bob_ppo.storage.clear()

        mean_bob_rew = np.mean(bob_rew_buf) if bob_rew_buf else 0.0
        bob_success_rate = np.mean(bob_success_buf) if bob_success_buf else 0.0
        mean_pos_err = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        mean_rot_err = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0
        bob_pos_sr_val = np.mean(bob_pos_sr_buf) if bob_pos_sr_buf else 0.0
        bob_rot_sr_val = np.mean(bob_rot_sr_buf) if bob_rot_sr_buf else 0.0
        bob_combined_sr = np.mean(bob_combined_sr_buf) if bob_combined_sr_buf else 0.0

        writer.add_scalar("Loss/Bob/Value", loss_val, bob_updates)
        writer.add_scalar("Loss/Bob/Surrogate", loss_surr, bob_updates)
        writer.add_scalar("Reward/Bob", mean_bob_rew, bob_updates)
        writer.add_scalar("Metrics/Bob/SuccessRate", bob_success_rate, bob_updates)
        writer.add_scalar("Metrics/Bob/PosError", mean_pos_err, bob_updates)
        writer.add_scalar("Metrics/Bob/RotError", mean_rot_err, bob_updates)
        writer.add_scalar("Metrics/Bob/PositionSR", bob_pos_sr_val, bob_updates)
        writer.add_scalar("Metrics/Bob/RotationSR", bob_rot_sr_val, bob_updates)
        writer.add_scalar("Metrics/Bob/CombinedSR", bob_combined_sr, bob_updates)

        print(f"  [Bob Update {bob_updates}] Loss: {loss_surr:.4f} | "
              f"Val: {loss_val:.4f} | Rew: {mean_bob_rew:.4f} | "
              f"SR: {bob_success_rate:.4f}", flush=True)

        if bob_combined_sr > best_bob_success_rate:
            best_bob_success_rate = bob_combined_sr
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "model_best.pt"))
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "model_best.pt"))
            torch.save(env.episode_manager.state_dict(),
                       os.path.join(bob_ppo.log_dir, "episode_manager_best.pt"))
            torch.save({"entropy_coef": alice_ppo.entropy_coef,
                        "abc_coef": bob_ppo.abc_coef,
                        "bob_success_buf": list(bob_success_buf), "best_bob_success_rate": best_bob_success_rate},
                       os.path.join(bob_ppo.log_dir, "train_state_best.pt"))

        bob_rew_buf.clear()
        bob_updates += 1

    _shutdown_requested = False

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        print("[INFO] SIGTERM received — emergency checkpoint after current iteration.", flush=True)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("Initialising environment...")
    with SuppressAllOutput():
        obs = env.reset()
    _update_goal_markers()

    _stagger = torch.randint(
        0, max(1, alice_pushes),
        (env.num_envs,), device=env.device, dtype=torch.int32,
    )
    env.episode_manager.phase_step.copy_(_stagger)
    print(f"[Init] Phase stagger applied.")

    _close_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
    _close_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
    _close_act[:, 6] = -1.0
    env.step(_close_act)

    ee_pos_local = _tcp_pos_local()
    ee_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()
    prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
    prev_phi_dpose = torch.zeros(env.num_envs, device=env.device)

    print("Starting training loop...")

    rollout_len = _rollout_len

    while bob_updates < args.max_iterations:
        if alice_hidden is not None:
            alice_hidden[0].zero_()
            alice_hidden[1].zero_()
        if bob_hidden is not None:
            bob_hidden[0].zero_()
            bob_hidden[1].zero_()

        if bob_updates > 0 and bob_updates % HIST_SAVE_INTERVAL == 0 and not args.no_hist_pool:
            alice_pool.add(alice_ppo.actor_critic)
            bob_pool.add(bob_ppo.actor_critic)
            print(f"  [HistPool] Saved snapshot at iter {bob_updates} "
                  f"(alice={alice_pool.size}, bob={bob_pool.size})", flush=True)

        alice_ppo.storage.clear()
        bob_ppo.storage.clear()

        if hasattr(env, "reset_iter_stats"):
            env.reset_iter_stats()

        iter_sr_counts = [0, 0]
        _iter_terminated = 0
        _iter_ik_fails = 0
        _iter_ik_steps = 0

        full_push_obs = env._get_push_obs()
        current_alice_obs = env._get_alice_obs(full_push_obs)
        current_bob_obs = env._get_bob_obs(full_push_obs)
        env.capture_pre_push(full_push_obs)
        _obj_pos_pre = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
        _obj_euler_pre = full_push_obs[:, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
        _goal_pos_pre = full_push_obs[:, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 14 + 3]
        _goal_euler_pre = full_push_obs[:, _OBS_ROBOT_DIM + 14 + 3:_OBS_ROBOT_DIM + 14 + 6]
        prev_phi_dpose = potential_dpose(
            _obj_pos_pre, _goal_pos_pre,
            _obj_euler_pre[..., 2], _goal_euler_pre[..., 2],
            args.char_length,
        )

        _a_pdim = num_cat_dims
        _b_pdim = num_cat_dims

        alice_traj_obs = torch.zeros((env.num_envs, alice_pushes, env.alice_obs_dim),
                                      device=env.device)
        alice_traj_act = torch.zeros((env.num_envs, alice_pushes, _a_pdim),
                                      device=env.device)
        alice_traj_len = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        hist_alice = (alice_pool.sample_policy(alice_ppo.actor_critic, env.device)
                      if alice_pool.size > 0 and not args.no_hist_pool else None)
        hist_bob = (bob_pool.sample_policy(bob_ppo.actor_critic, env.device)
                    if bob_pool.size > 0 and not args.no_hist_pool else None)

        for t in range(rollout_len):
            is_alice = env.episode_manager.is_alice_phase()
            is_bob = env.episode_manager.is_bob_phase()
            alice_indices = torch.where(is_alice)[0]
            bob_indices = torch.where(is_bob)[0]

            _update_goal_markers()

            alice_local_idx = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
            if len(alice_indices) > 0:
                alice_local_idx[alice_indices] = torch.arange(len(alice_indices), device=env.device)
            bob_local_idx = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
            if len(bob_indices) > 0:
                bob_local_idx[bob_indices] = torch.arange(len(bob_indices), device=env.device)

            _alice_hidden_pre = (
                (alice_hidden[0].clone(), alice_hidden[1].clone())
                if alice_hidden is not None else None
            )
            a_acts_active = torch.zeros((len(alice_indices), _a_pdim), device=env.device)
            a_logprob_active = torch.zeros(len(alice_indices), device=env.device)
            a_val_active = torch.zeros(len(alice_indices), 1, device=env.device)
            a_mu_active = torch.zeros_like(a_acts_active)
            a_sigma_active = torch.zeros_like(a_acts_active)

            if len(alice_indices) > 0:
                hist_ids, curr_ids = (
                    alice_pool.sample_env_subset(alice_indices, frac=HIST_FRAC)
                    if not args.no_hist_pool and hist_alice is not None
                    else (torch.tensor([], dtype=torch.long, device=env.device), alice_indices)
                )
                with torch.no_grad():
                    h_in = ((alice_hidden[0][curr_ids], alice_hidden[1][curr_ids])
                            if alice_hidden else None)
                    (a_acts_curr, a_logprob_curr, a_val_curr, a_mu_curr, a_sigma_curr, new_h) = \
                        alice_ppo.actor_critic.act_with_hidden(
                            current_alice_obs[curr_ids], None, h_in,
                        )
                    if alice_hidden and new_h is not None:
                        alice_hidden[0][curr_ids] = new_h[0]
                        alice_hidden[1][curr_ids] = new_h[1]
                    if len(hist_ids) > 0 and hist_alice is not None:
                        (a_acts_hist, a_logprob_hist, a_val_hist, a_mu_hist, a_sigma_hist, _) = \
                            hist_alice.act_with_hidden(
                                current_alice_obs[hist_ids], None, None,
                            )
                    else:
                        a_acts_hist = a_logprob_hist = a_val_hist = a_mu_hist = a_sigma_hist = None

                curr_local = alice_local_idx[curr_ids]
                a_acts_active[curr_local] = a_acts_curr
                a_logprob_active[curr_local] = a_logprob_curr
                a_val_active[curr_local] = a_val_curr
                a_mu_active[curr_local] = a_mu_curr
                a_sigma_active[curr_local] = a_sigma_curr

                if len(hist_ids) > 0 and a_acts_hist is not None:
                    hist_local = alice_local_idx[hist_ids]
                    a_acts_active[hist_local] = a_acts_hist
                    a_logprob_active[hist_local] = a_logprob_hist
                    a_val_active[hist_local] = a_val_hist
                    a_mu_active[hist_local] = a_mu_hist
                    a_sigma_active[hist_local] = a_sigma_hist
                    if _alice_hidden_pre is not None:
                        _alice_hidden_pre[0][hist_ids] = 0.0
                        _alice_hidden_pre[1][hist_ids] = 0.0

            if _alice_hidden_pre is not None:
                _alice_hidden_pre[0][~is_alice] = 0.0
                _alice_hidden_pre[1][~is_alice] = 0.0

            _bob_hidden_pre = (
                (bob_hidden[0].clone(), bob_hidden[1].clone())
                if bob_hidden is not None else None
            )
            b_acts_active = torch.zeros((len(bob_indices), _b_pdim), device=env.device)
            b_logprob_active = torch.zeros(len(bob_indices), device=env.device)
            b_val_active = torch.zeros(len(bob_indices), 1, device=env.device)
            b_mu_active = torch.zeros_like(b_acts_active)
            b_sigma_active = torch.zeros_like(b_acts_active)

            if len(bob_indices) > 0:
                hist_bids, curr_bids = (
                    bob_pool.sample_env_subset(bob_indices, frac=HIST_FRAC)
                    if not args.no_hist_pool and hist_bob is not None
                    else (torch.tensor([], dtype=torch.long, device=env.device), bob_indices)
                )
                with torch.no_grad():
                    h_in = ((bob_hidden[0][curr_bids], bob_hidden[1][curr_bids])
                            if bob_hidden else None)
                    (b_acts_curr, b_lp_curr, b_val_curr, b_mu_curr, b_sig_curr, new_bh) = \
                        bob_ppo.actor_critic.act_with_hidden(
                            current_bob_obs[curr_bids], None, h_in,
                        )
                    if bob_hidden and new_bh is not None:
                        bob_hidden[0][curr_bids] = new_bh[0]
                        bob_hidden[1][curr_bids] = new_bh[1]
                    if len(hist_bids) > 0 and hist_bob is not None:
                        (b_acts_hist, b_lp_hist, b_val_hist, b_mu_hist, b_sig_hist, _) = \
                            hist_bob.act_with_hidden(
                                current_bob_obs[hist_bids], None, None,
                            )
                    else:
                        b_acts_hist = b_lp_hist = b_val_hist = b_mu_hist = b_sig_hist = None

                curr_bloc = bob_local_idx[curr_bids]
                b_acts_active[curr_bloc] = b_acts_curr
                b_logprob_active[curr_bloc] = b_lp_curr
                b_val_active[curr_bloc] = b_val_curr
                b_mu_active[curr_bloc] = b_mu_curr
                b_sigma_active[curr_bloc] = b_sig_curr

                if len(hist_bids) > 0 and b_acts_hist is not None:
                    hist_bloc = bob_local_idx[hist_bids]
                    b_acts_active[hist_bloc] = b_acts_hist
                    b_logprob_active[hist_bloc] = b_lp_hist
                    b_val_active[hist_bloc] = b_val_hist
                    b_mu_active[hist_bloc] = b_mu_hist
                    b_sigma_active[hist_bloc] = b_sig_hist
                    if _bob_hidden_pre is not None:
                        _bob_hidden_pre[0][hist_bids] = 0.0
                        _bob_hidden_pre[1][hist_bids] = 0.0

            if _bob_hidden_pre is not None:
                _bob_hidden_pre[0][~is_bob] = 0.0
                _bob_hidden_pre[1][~is_bob] = 0.0

            a_policy = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_policy[alice_indices] = a_acts_active
            b_policy = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_policy[bob_indices] = b_acts_active

            alice_step_raw = env.episode_manager.phase_step[alice_indices] - 1
            valid_t = alice_step_raw < alice_pushes
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

            _obj_xy_all = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 2]
            _obj_yaw_all = full_push_obs[:, _OBS_ROBOT_DIM + 5]

            min_r = TBLOCK_MIN_R
            max_r = TBLOCK_MAX_R
            max_l = 0.20
            if len(alice_indices) > 0:
                a_Xs, a_Ys, a_len, a_theta = decode_push_action_relative(
                    a_acts_active,
                    _obj_xy_all[alice_indices],
                    _obj_yaw_all[alice_indices],
                    num_bins=num_bins,
                    min_r=min_r,
                    max_r=max_r,
                    max_len=max_l,
                )
            else:
                a_Xs = a_Ys = a_len = a_theta = torch.zeros(0, device=env.device)

            if len(bob_indices) > 0:
                b_Xs, b_Ys, b_len, b_theta = decode_push_action_relative(
                    b_acts_active,
                    _obj_xy_all[bob_indices],
                    _obj_yaw_all[bob_indices],
                    num_bins=num_bins,
                    min_r=min_r,
                    max_r=max_r,
                    max_len=max_l,
                )
            else:
                b_Xs = b_Ys = b_len = b_theta = torch.zeros(0, device=env.device)

            Xs = torch.zeros(env.num_envs, device=env.device)
            Ys = torch.zeros(env.num_envs, device=env.device)
            length = torch.zeros(env.num_envs, device=env.device)
            theta = torch.zeros(env.num_envs, device=env.device)
            Xs[alice_indices] = a_Xs
            Ys[alice_indices] = a_Ys
            length[alice_indices] = a_len
            theta[alice_indices] = a_theta
            Xs[bob_indices] = b_Xs
            Ys[bob_indices] = b_Ys
            length[bob_indices] = b_len
            theta[bob_indices] = b_theta

            _margin = 0.02
            Xs.clamp_(_WS_X[0] + _margin, _WS_X[1] - _margin)
            Ys.clamp_(_WS_Y[0] + _margin, _WS_Y[1] - _margin)
            _Xf = Xs + length * torch.cos(theta)
            _Yf = Ys + length * torch.sin(theta)
            _Xf.clamp_(_WS_X[0] + _margin, _WS_X[1] - _margin)
            _Yf.clamp_(_WS_Y[0] + _margin, _WS_Y[1] - _margin)
            length = torch.sqrt((_Xf - Xs) ** 2 + (_Yf - Ys) ** 2)
            theta = torch.atan2(_Yf - Ys, _Xf - Xs)

            _update_push_markers(Xs, Ys, length, theta)

            push_wps = compute_push_waypoints(
                Xs=Xs, Ys=Ys,
                length=length, theta=theta,
                current_ee_pos=ee_pos_local,
                current_ee_quat=ee_quat_w,
                device=env.device,
            )

            terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for wp_idx, (wp_pos, wp_quat, _wp_grip) in enumerate(push_wps):
                ik_target = wp_pos - _TOTAL_IK_ERROR
                ik_target[:, 0].clamp_(_WS_X[0], _WS_X[1])
                ik_target[:, 1].clamp_(_WS_Y[0], _WS_Y[1])
                ik_target[:, 2].clamp_(_WS_Z[0], _WS_Z[1])

                result = ik_solver.solve_batch(
                    CuroboPose(position=ik_target, quaternion=wp_quat),
                    seed_config=prev_joint_cmd.unsqueeze(1),
                    retract_config=prev_joint_cmd,
                )

                ik_ok = result.success.squeeze(-1)
                cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]
                _iter_ik_steps += env.num_envs
                _iter_ik_fails += int((~ik_ok).sum().item())

                solved = result.solution.view(env.num_envs, 6)
                elbow_bad = solved[:, 2] < 0.0
                if elbow_bad.any():
                    ik_ok[elbow_bad] = False
                raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, cur_joints)
                if terminated.any():
                    raw_cmd[terminated] = cur_joints[terminated]
                if (~ik_ok).any():
                    prev_joint_cmd[~ik_ok] = cur_joints[~ik_ok]
                prev_joint_cmd[ik_ok] = raw_cmd[ik_ok].detach().clone()

                env_full = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
                env_full[:, :6] = raw_cmd
                env_full[:, 6] = -1.0

                obs_ret, _, step_terminated, truncated, _ = env.step(env_full)
                terminated |= step_terminated

                _tcp_z_check = _tcp_pos_local()[:, 2]
                terminated |= (_tcp_z_check < -0.01)

            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

            full_push_obs = env._get_push_obs()
            current_alice_obs = env._get_alice_obs(full_push_obs)
            current_bob_obs = env._get_bob_obs(full_push_obs)

            _obj_pos_all = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
            _oob_ws = ((_obj_pos_all[:, 0] < _WS_X[0]) | (_obj_pos_all[:, 0] > _WS_X[1]) |
                       (_obj_pos_all[:, 1] < _WS_Y[0]) | (_obj_pos_all[:, 1] > _WS_Y[1]))

            bob_rewards = torch.zeros(env.num_envs, device=env.device)
            bob_sparse = torch.zeros(env.num_envs, device=env.device)
            pbrs_result = None
            _catastrophe = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            if len(bob_indices) > 0:
                _obj_pos_now = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
                _obj_euler_now = full_push_obs[:, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
                _goal_pos_now = full_push_obs[:, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 14 + 3]
                _goal_euler_now = full_push_obs[:, _OBS_ROBOT_DIM + 14 + 3:_OBS_ROBOT_DIM + 14 + 6]
                pbrs_result = compute_pbrs_reward_dpose(
                    _obj_pos_now, _obj_euler_now, _goal_pos_now, _goal_euler_now,
                    prev_phi_dpose,
                    env._bob_gave_completion,
                    char_length=args.char_length,
                    success_threshold=args.dpose_threshold,
                )
                bob_rewards = pbrs_result["reward"]
                env._bob_gave_completion = pbrs_result["gave_completion"]
                bob_sparse = pbrs_result["reward"] - pbrs_result["dense"]
                _tipped = pbrs_result["tipped"]
                _launched = _obj_pos_now[:, 2] > 0.10
                _catastrophe = (_tipped | _launched | _oob_ws) & is_bob
                bob_rewards[_catastrophe & ~terminated] = -10.0
            bob_rewards[terminated] = 0.0

            _n_alice = len(alice_indices)
            _n_bob = len(bob_indices)
            _bob_rew_mean = bob_rewards[bob_indices].mean().item() if _n_bob > 0 else 0.0
            if pbrs_result is not None and _n_bob > 0:
                print(
                    f"  [Step {t:3d}] A={_n_alice} B={_n_bob}  "
                    f"bob_rew={_bob_rew_mean:+.3f}  "
                    f"dense={pbrs_result['dense'][bob_indices].mean().item():+.4f}  "
                    f"d_pose={pbrs_result['d_pose'][bob_indices].mean().item():.4f}  "
                    f"at_goal={pbrs_result['at_goal'][bob_indices].sum().item():.0f}/{_n_bob}  "
                    f"cata={_catastrophe.sum().item()}",
                    flush=True,
                )
            else:
                print(f"  [Step {t:3d}] A={_n_alice} B={_n_bob}", flush=True)

            if args.debug_rewards and pbrs_result is not None:
                for e in range(env.num_envs):
                    if is_bob[e]:
                        _flags = []
                        if pbrs_result["at_goal"][e]:
                            _flags.append("AT_GOAL")
                        if pbrs_result["tipped"][e]:
                            _flags.append("TIPPED")
                        if _catastrophe[e]:
                            _flags.append("CATA")
                        if terminated[e]:
                            _flags.append("TERM")
                        _flag_str = " ".join(_flags) if _flags else ""
                        print(
                            f"    env {e:3d} [BOB ]: rew={bob_rewards[e].item():+.3f}  "
                            f"Phi={prev_phi_dpose[e].item():.3f}->{pbrs_result['phi_now'][e].item():.3f}  "
                            f"dense={pbrs_result['dense'][e].item():+.4f}  "
                            f"d_pose={pbrs_result['d_pose'][e].item():.4f}m  "
                            f"{_flag_str}",
                            flush=True,
                        )
                    elif is_alice[e]:
                        _a_ox = full_push_obs[e, _OBS_ROBOT_DIM].item()
                        _a_oy = full_push_obs[e, _OBS_ROBOT_DIM + 1].item()
                        _a_oz = full_push_obs[e, _OBS_ROBOT_DIM + 2].item()
                        _a_yaw = full_push_obs[e, _OBS_ROBOT_DIM + 5].item()
                        print(
                            f"    env {e:3d} [ALICE]: push={env.episode_manager.phase_step[e].item()}  "
                            f"obj=({_a_ox:+.3f},{_a_oy:+.3f},{_a_oz:+.3f}) yaw={_a_yaw:+.3f}  "
                            f"len={float(length[e]):.3f} theta={math.degrees(float(theta[e])):.0f} deg",
                            flush=True,
                        )

            bob_achieved_completion = pbrs_result["at_goal"] if (len(bob_indices) > 0 and pbrs_result is not None) else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

            alice_rewards_now = torch.zeros(env.num_envs, device=env.device)
            alice_done_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            bob_done_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            bob_success_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            goal_valid_now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            bob_pos_err_now = torch.zeros(env.num_envs, device=env.device)
            bob_rot_err_now = torch.zeros(env.num_envs, device=env.device)
            bob_progress_rew = torch.zeros(env.num_envs, device=env.device)
            bob_phase_penalty = torch.zeros(env.num_envs, device=env.device)

            _prev_initial = env.episode_manager.initial_states.clone()

            phase_info = env.episode_manager.step()

            alice_done_mask = phase_info.get("alice_done", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            bob_done_mask = phase_info.get("bob_done", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

            alice_done_ids = torch.where(alice_done_mask)[0]
            if len(alice_done_ids) > 0:
                valid_ids, invalid_ids = env.handle_alice_phase_end(
                    alice_done_ids, full_push_obs,
                )
                alice_rewards_now[alice_done_ids] = env.delayed_alice_reward[alice_done_ids]
                env.delayed_alice_reward[alice_done_ids] = 0.0
                alice_done_now[invalid_ids] = True
                goal_valid_now[valid_ids] = True

            _goal_valid_pre = env.episode_manager.goal_valid.clone()

            bob_done_ids = torch.where(bob_done_mask)[0]
            if len(bob_done_ids) > 0:
                (bsucc, bpos, brot, bdone, bprog) = env.handle_bob_phase_end(
                    bob_done_ids, full_push_obs,
                )
                bob_success_now[bob_done_ids] = bsucc[bob_done_ids]
                bob_pos_err_now[bob_done_ids] = bpos[bob_done_ids]
                bob_rot_err_now[bob_done_ids] = brot[bob_done_ids]
                bob_done_now[bob_done_ids] = bdone[bob_done_ids]

                # Bob time penalty (Model I — full Sukhbaatar symmetric reward)
                if args.bob_time_penalty and len(bob_done_ids) > 0:
                    t_B = env.episode_manager.phase_step[bob_done_ids].float()
                    bob_phase_penalty[bob_done_ids] = -args.bob_time_penalty_scale * t_B

            if bob_achieved_completion.any():
                completion_ids = torch.where(bob_achieved_completion)[0]
                bob_progress_rew += env.handle_bob_early_success(
                    completion_ids, full_push_obs,
                )
                bob_done_now[completion_ids] = True
                bob_success_now[completion_ids] = True
                # Bob time penalty for early-success envs
                if args.bob_time_penalty:
                    t_B_comp = env.episode_manager.phase_step[completion_ids].float()
                    bob_phase_penalty[completion_ids] = -args.bob_time_penalty_scale * t_B_comp
                _c_pos = env._get_obj_pos(full_push_obs)[completion_ids]
                _c_goal = env._get_goal_pos(full_push_obs)[completion_ids]
                _c_euler = env._get_obj_euler(full_push_obs)[completion_ids]
                _c_geuler = env._get_goal_euler(full_push_obs)[completion_ids]
                bob_pos_err_now[completion_ids] = (_c_pos - _c_goal).norm(dim=-1)
                _rdiff = (_c_euler - _c_geuler) % (2.0 * math.pi)
                _rdiff = torch.where(_rdiff > math.pi, 2.0 * math.pi - _rdiff, _rdiff)
                bob_rot_err_now[completion_ids] = _rdiff.max(dim=-1)[0]

            _cata_early = _catastrophe & is_bob & ~bob_done_mask & ~bob_achieved_completion
            if _cata_early.any():
                cata_ids = torch.where(_cata_early)[0]
                env.handle_bob_phase_end(cata_ids, full_push_obs)
                bob_done_now[cata_ids] = True
                # Bob time penalty for catastrophe envs
                if args.bob_time_penalty:
                    t_B_cata = env.episode_manager.phase_step[cata_ids].float()
                    bob_phase_penalty[cata_ids] = -args.bob_time_penalty_scale * t_B_cata

            _alice_oob = _oob_ws & is_alice & ~alice_done_mask
            if _alice_oob.any():
                _aoob_ids = torch.where(_alice_oob)[0]
                env.episode_manager.reset_episode(_aoob_ids, reason="Object Outside WS")
                env.hide_goal_ghost(_aoob_ids)
                _sp = env._rand_reset_objs(_aoob_ids)
                reset_robot_joints(env.env, _aoob_ids)
                env.env.scene.write_data_to_sim()
                env.episode_manager.initial_states[_aoob_ids] = env._initial_states_from_spawn(
                    _sp, len(_aoob_ids))
                alice_rewards_now[_aoob_ids] = -3.0
                alice_done_now[_aoob_ids] = True
                env.set_table_color(_aoob_ids, (0.8, 0.1, 0.1))

            if args.debug_rewards:
                for _gi in alice_done_ids.tolist():
                    _valid = "VALID" if goal_valid_now[_gi].item() else "INVALID"
                    _rew = alice_rewards_now[_gi].item()
                    _sx = _prev_initial[_gi, 0].item()
                    _sy = _prev_initial[_gi, 1].item()
                    _sz = _prev_initial[_gi, 2].item()
                    _syaw = _prev_initial[_gi, 5].item()
                    _final_pos = full_push_obs[_gi, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
                    _final_ori = full_push_obs[_gi, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
                    _fx = _final_pos[0].item()
                    _fy = _final_pos[1].item()
                    _fz = _final_pos[2].item()
                    _fyaw = _final_ori[2].item()
                    _disp3d = math.sqrt((_fx-_sx)**2 + (_fy-_sy)**2 + (_fz-_sz)**2)
                    _disp2d = math.sqrt((_fx-_sx)**2 + (_fy-_sy)**2)
                    _reason_parts = []
                    if not goal_valid_now[_gi].item():
                        if _fz > 0.05:
                            _reason_parts.append("AIRBORNE")
                        if _disp2d < 0.05:
                            _reason_parts.append("NO_DISP")
                        elif _disp2d < 0.10:
                            _reason_parts.append("SHALLOW")
                        if _fx < -0.50 or _fx > 0.50 or _fy < 0.25 or _fy > 0.70:
                            _reason_parts.append("OOZ")
                    _reason_str = "(" + "+".join(_reason_parts) + ")" if _reason_parts else ""
                    print(
                        f"  [ALICE END | iter={bob_updates} env={_gi}] {_valid}{_reason_str}  rew={_rew:+.1f}  "
                        f"disp3D={_disp3d:.4f}m  "
                        f"obj_start=({_sx:+.3f},{_sy:+.3f},{_sz:+.3f}) yaw={_syaw:+.3f}  "
                        f"obj_end=({_fx:+.3f},{_fy:+.3f},{_fz:+.3f}) yaw={_fyaw:+.3f}  "
                        f"Dxy=({_fx-_sx:+.3f},{_fy-_sy:+.3f}) Dyaw={_fyaw-_syaw:+.3f}",
                        flush=True,
                    )

            if args.debug_rewards:
                _gs = env.episode_manager.goal_states
                _gv_tensor = env.episode_manager.goal_valid
                for _gi in (torch.where(bob_done_mask)[0]).tolist():
                    _succ = "SUCCESS" if bob_success_now[_gi].item() else "fail"
                    _gv = "valid_goal" if _gv_tensor[_gi].item() else "no_goal"
                    _pos = bob_pos_err_now[_gi].item()
                    _rot = bob_rot_err_now[_gi].item()
                    _start_pos = _prev_initial[_gi, 0:3].tolist()
                    _start_yaw = _prev_initial[_gi, 5].item()
                    if _gs is not None:
                        _goal_pos = _gs[_gi, 0:3].tolist()
                        _goal_yaw = _gs[_gi, 5].item()
                    else:
                        _goal_pos = [0.0, 0.0, 0.0]
                        _goal_yaw = 0.0
                    _obj_pos = full_push_obs[_gi, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3].tolist()
                    _obj_yaw = full_push_obs[_gi, _OBS_ROBOT_DIM + 5].item()
                    _phi = pbrs_result["phi_now"][_gi].item() if pbrs_result is not None else 0.0
                    _d_pose_val = compute_dpose(
                        torch.tensor([_obj_pos]).to(env.device),
                        torch.tensor([_goal_pos]).to(env.device),
                        torch.tensor([_obj_yaw]).to(env.device),
                        torch.tensor([_goal_yaw]).to(env.device),
                        args.char_length,
                    ).item()
                    print(
                        f"  [BOB END   | iter={bob_updates} env={_gi}] {_succ} ({_gv})  "
                        f"obj_start=({_start_pos[0]:+.3f},{_start_pos[1]:+.3f},{_start_pos[2]:+.3f}) yaw={_start_yaw:+.3f}  "
                        f"goal=({_goal_pos[0]:+.3f},{_goal_pos[1]:+.3f},{_goal_pos[2]:+.3f}) yaw={_goal_yaw:+.3f}  "
                        f"obj_final=({_obj_pos[0]:+.3f},{_obj_pos[1]:+.3f},{_obj_pos[2]:+.3f}) yaw={_obj_yaw:+.3f}  "
                        f"pos_err={_pos:.4f}m  rot_err={_rot:.4f}rad  "
                        f"d_pose={_d_pose_val:.4f}m  "
                        f"Phi={_phi:.3f}",
                        flush=True,
                    )

            full_push_obs = env._get_push_obs()
            next_alice_obs = env._get_alice_obs(full_push_obs)
            next_bob_obs = env._get_bob_obs(full_push_obs)

            rewards = torch.zeros(env.num_envs, device=env.device)
            if len(bob_indices) > 0:
                rewards[bob_indices] = bob_rewards[bob_indices]
            rewards += bob_progress_rew + bob_phase_penalty

            if len(bob_indices) > 0:
                bob_rew_buf.extend(rewards[bob_indices].cpu().tolist())
                if pbrs_result is not None:
                    writer.add_scalar("Reward/Bob/Dense/DPose", pbrs_result["dense"][bob_indices].mean().item(), bob_updates)
                    writer.add_scalar("Metrics/Bob/DPose", pbrs_result["d_pose"][bob_indices].mean().item(), bob_updates)
                # Log Bob time penalty (Model I)
                if args.bob_time_penalty:
                    _bp_mean = bob_phase_penalty[bob_indices].mean().item()
                    writer.add_scalar("Reward/Bob/TimePenalty", _bp_mean, bob_updates)

            dones_all = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

            if alice_hidden is not None:
                done_a = alice_indices[alice_done_mask[alice_indices]]
                if len(done_a) > 0:
                    alice_hidden[0][done_a] = 0.0
                    alice_hidden[1][done_a] = 0.0

            if bob_hidden is not None:
                done_b = bob_indices[bob_done_mask[bob_indices]]
                if len(done_b) > 0:
                    bob_hidden[0][done_b] = 0.0
                    bob_hidden[1][done_b] = 0.0

            if bob_hidden is not None:
                early_done = bob_done_now & ~bob_done_mask
                if early_done.any():
                    early_ids = torch.where(early_done)[0]
                    bob_hidden[0][early_ids] = 0.0
                    bob_hidden[1][early_ids] = 0.0

            needs_ee_reset = alice_done_mask | bob_done_mask | bob_done_now | terminated
            if needs_ee_reset.any():
                reset_eids = torch.where(needs_ee_reset)[0]
                ee_pos_local[reset_eids] = _tcp_pos_local()[reset_eids]
                ee_quat_w[reset_eids] = _QUAT_TOOL_DOWN.expand(len(reset_eids), 4).to(env.device)
                prev_joint_cmd[reset_eids] = _robot_scene.data.joint_pos[:, _arm_jids][reset_eids]

            env.push_count += 1

            a_lp_full = torch.zeros(env.num_envs, device=env.device)
            a_val_full = torch.zeros(env.num_envs, 1, device=env.device)
            a_mu_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_sigma_full = torch.zeros((env.num_envs, _a_pdim), device=env.device)
            a_lp_full[alice_indices] = a_logprob_active
            a_val_full[alice_indices] = a_val_active
            a_mu_full[alice_indices] = a_mu_active
            a_sigma_full[alice_indices] = a_sigma_active

            a_masks = torch.zeros(env.num_envs, 1, device=env.device)
            a_masks[alice_indices] = 1.0

            alice_ppo.storage.add_transitions(
                current_alice_obs, next_alice_obs, a_policy,
                rewards, dones_all, a_val_full, a_lp_full,
                a_mu_full, a_sigma_full, a_masks,
                hidden_state=_alice_hidden_pre,
            )
            current_alice_obs = next_alice_obs

            b_lp_full = torch.zeros(env.num_envs, 1, device=env.device)
            b_val_full = torch.zeros(env.num_envs, 1, device=env.device)
            b_mu_full = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_sigma_full = torch.zeros((env.num_envs, _b_pdim), device=env.device)
            b_lp_full[bob_indices] = b_logprob_active.unsqueeze(1)
            b_val_full[bob_indices] = b_val_active
            b_mu_full[bob_indices] = b_mu_active
            b_sigma_full[bob_indices] = b_sigma_active

            ended_for_bob = dones_all | bob_done_now
            b_masks = torch.zeros(env.num_envs, 1, device=env.device)
            b_masks[bob_indices[~ended_for_bob[bob_indices]]] = 1.0

            bob_ppo.storage.add_transitions(
                current_bob_obs, next_bob_obs, b_policy,
                rewards, dones_all, b_val_full, b_lp_full,
                b_mu_full, b_sigma_full, b_masks,
                hidden_state=_bob_hidden_pre,
            )
            current_bob_obs = next_bob_obs

            bob_done_envs = torch.where(bob_done_now)[0]
            if len(bob_done_envs) > 0:
                bob_pos_err_buf.extend(bob_pos_err_now[bob_done_envs].cpu().tolist())
                bob_rot_err_buf.extend(bob_rot_err_now[bob_done_envs].cpu().tolist())
                bob_pos_sr_buf.extend(
                    (bob_pos_err_now[bob_done_envs] < 0.05).cpu().tolist(),
                )
                bob_rot_sr_buf.extend(
                    (bob_rot_err_now[bob_done_envs] < 0.01).cpu().tolist(),
                )
                bob_combined_sr_buf.extend(
                    ((bob_pos_err_now[bob_done_envs] < 0.05)
                     & (bob_rot_err_now[bob_done_envs] < 0.2)).cpu().tolist(),
                )
                iter_sr_counts[0] += len(bob_done_envs)
                succ_count = int(bob_success_now[bob_done_envs].sum().item())
                iter_sr_counts[1] += succ_count
                if args.debug_rewards:
                    print(f"  [TRACK] push step: bob_done={len(bob_done_envs)}  bob_succ={succ_count}  "
                          f"iter_sr=({iter_sr_counts[0]},{iter_sr_counts[1]})", flush=True)

            alice_reward_envs = torch.where(
                (alice_rewards_now != 0) | alice_done_now,
            )[0]
            if len(alice_reward_envs) > 0:
                filled = alice_ppo.storage.step
                if filled > 0:
                    masks = alice_ppo.storage.masks[:filled, :, 0]
                    row_idx = torch.arange(filled, device=env.device).unsqueeze(1).expand_as(masks)
                    last_valid = torch.where(
                        masks.bool(), row_idx,
                        torch.tensor(-1, device=env.device),
                    ).max(dim=0).values
                    has_valid = last_valid >= 0
                    rewarded_envs = alice_reward_envs[has_valid[alice_reward_envs]]
                    if len(rewarded_envs) > 0:
                        rows = last_valid[rewarded_envs]
                        alice_ppo.storage.rewards[rows, rewarded_envs, 0] += \
                            alice_rewards_now[rewarded_envs]
                        alice_rew_buf.extend(alice_rewards_now[rewarded_envs].cpu().tolist())

            if not args.no_abc:
                try:
                    just_failed_bob = bob_done_now & (~bob_success_now) & _goal_valid_pre
                    valid_ids = torch.where(just_failed_bob)[0]
                    min_demo_steps = max(2, alice_pushes // 2)
                    valid_trajs = []
                    for env_id in valid_ids:
                        eid = env_id.item()
                        t_len = alice_traj_len[eid].item()
                        if t_len < min_demo_steps:
                            continue
                        traj_o = alice_traj_obs[eid, :t_len]
                        traj_a = alice_traj_act[eid, :t_len].long()
                        g = env.episode_manager.goal_states
                        if g is not None:
                            g = g[eid].unsqueeze(0).expand(t_len, -1)
                        else:
                            g = torch.zeros(t_len, 6, device=env.device)
                        bc_obs = env.construct_bob_observation(traj_o, g)
                        valid_trajs.append((bc_obs, traj_a))
                    if valid_trajs:
                        all_obs2 = torch.cat([t[0] for t in valid_trajs], dim=0)
                        all_acts2 = torch.cat([t[1] for t in valid_trajs], dim=0)
                        with torch.no_grad():
                            old_lp_all, _, _, _, _ = bob_ppo.actor_critic.evaluate(
                                all_obs2, None, all_acts2,
                            )
                        offset = 0
                        for bc_obs_i, traj_a_i in valid_trajs:
                            t_len_i = bc_obs_i.shape[0]
                            bob_ppo.abc_buffer.add_trajectory(
                                bc_obs_i, traj_a_i,
                                old_lp_all[offset:offset + t_len_i],
                            )
                            offset += t_len_i
                except Exception as e:
                    if args.debug_rewards:
                        print(f"  [ABC] WARNING: buffer population error: {e}", flush=True)

            env.capture_pre_push(full_push_obs)
            _obj_pos_pre = full_push_obs[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
            _obj_euler_pre = full_push_obs[:, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
            _goal_pos_pre = full_push_obs[:, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 14 + 3]
            _goal_euler_pre = full_push_obs[:, _OBS_ROBOT_DIM + 14 + 3:_OBS_ROBOT_DIM + 14 + 6]
            prev_phi_dpose = potential_dpose(
                _obj_pos_pre, _goal_pos_pre,
                _obj_euler_pre[..., 2], _goal_euler_pre[..., 2],
                args.char_length,
            )

        perform_alice_update()
        perform_bob_update(current_bob_obs)

        current_sr = iter_sr_counts[1] / max(1, iter_sr_counts[0])
        if args.debug_rewards:
            print(f"  [DEBUG SR] iter_sr_counts[0]={iter_sr_counts[0]}  iter_sr_counts[1]={iter_sr_counts[1]}  "
                  f"bob_done_now_sum={int(bob_done_now.sum().item()) if hasattr(bob_done_now, 'sum') else 'N/A'}", flush=True)
        bob_success_buf.append(current_sr)

        writer.add_scalar("Alice/EntropyCoef", alice_ppo.entropy_coef, bob_updates)
        print(f"  [Alice] Entropy Coef: {alice_ppo.entropy_coef:.4f} (fixed)", flush=True)

        _alice_lr_max = alice_ppo.learning_rate
        _alice_lr_min = ppo_cfg["params"]["learn"].get("alice_lr_min", 5e-5)
        _lr_p = min(1.0, bob_updates / args.max_iterations)
        _alice_lr = _alice_lr_min + 0.5 * (_alice_lr_max - _alice_lr_min) * (
            1.0 + math.cos(math.pi * _lr_p)
        )
        for pg in alice_ppo.optimizer.param_groups:
            pg["lr"] = _alice_lr
        writer.add_scalar("Alice/LearningRate", _alice_lr, bob_updates)

        bob_ppo.abc_coef = ppo_cfg["params"]["learn"].get("abc_coef", 0.5)

        _stats = env.get_iter_stats() if hasattr(env, "get_iter_stats") else {}
        _valid_goals = _stats.get("valid_goals", 0)
        _invalid_goals = _stats.get("invalid_goals", 0)
        _bob_succ = _stats.get("bob_successes", 0)
        _bob_fail = _stats.get("bob_failures", 0)
        writer.add_scalar("Metrics/Alice/ValidGoals", _valid_goals, bob_updates)
        writer.add_scalar("Metrics/Alice/InvalidGoals", _invalid_goals, bob_updates)
        gvr = _valid_goals / max(1, _valid_goals + _invalid_goals)
        writer.add_scalar("Metrics/Alice/GoalValidityRate", gvr, bob_updates)
        _alice_disp = _stats.get("alice_disp_3d_sum", 0.0)
        _alice_total = _stats.get("alice_total", 1)
        _mean_disp = _alice_disp / max(1, _alice_total)
        writer.add_scalar("Metrics/Alice/MeanDisp3D", _mean_disp, bob_updates)
        writer.add_scalar("Metrics/Alice/EMAReward", ema_alice_rew, bob_updates)
        _ik_fail_rate = _iter_ik_fails / max(1, _iter_ik_steps)
        writer.add_scalar("Metrics/IKFailRate", _ik_fail_rate, bob_updates)
        writer.add_scalar("Diagnostics/GradientUpdates", bob_updates, bob_updates)

        _mean_pos_err = np.mean(bob_pos_err_buf) if bob_pos_err_buf else 0.0
        _mean_rot_err = np.mean(bob_rot_err_buf) if bob_rot_err_buf else 0.0
        _bp_label = f"Bp={args.bob_time_penalty_scale}" if args.bob_time_penalty else "Bp=OFF"

        print(
            f"[Iter {bob_updates}] SR={current_sr:.2f} | "
            f"PosErr={_mean_pos_err:.4f}m RotErr={_mean_rot_err:.4f}rad | "
            f"Goals valid={_valid_goals} invalid={_invalid_goals} | "
            f"Bob succ={_bob_succ} fail={_bob_fail} | "
            f"IK_fail={_ik_fail_rate:.3f} | "
            f"Term={_iter_terminated} | "
            f"{_bp_label} | "
            f"dpose L={args.char_length} thr={args.dpose_threshold}",
            flush=True,
        )

        if _shutdown_requested:
            print(f"[INFO] Emergency checkpoint at iter {bob_updates}", flush=True)
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "latest_checkpoint.pt"), iteration=bob_updates)
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "latest_checkpoint.pt"), iteration=bob_updates)
            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
            torch.save(env.episode_manager.state_dict(),
                       os.path.join(bob_ppo.log_dir, "episode_manager_latest.pt"))
            torch.save({"entropy_coef": alice_ppo.entropy_coef,
                        "abc_coef": bob_ppo.abc_coef,
                        "bob_success_buf": list(bob_success_buf), "best_bob_success_rate": best_bob_success_rate},
                       os.path.join(bob_ppo.log_dir, "train_state_latest.pt"))
            with open(os.path.join(run_dir, "latest_iter.txt"), "w") as _f:
                _f.write(str(bob_updates))
            print(f"[INFO] Emergency checkpoint saved (iter {bob_updates}) — exiting cleanly.", flush=True)
            break

        if args.save_interval > 0 and bob_updates % args.save_interval == 0:
            bob_ppo.save(os.path.join(bob_ppo.log_dir, "latest_checkpoint.pt"), iteration=bob_updates)
            alice_ppo.save(os.path.join(alice_ppo.log_dir, "latest_checkpoint.pt"), iteration=bob_updates)
            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
            torch.save(env.episode_manager.state_dict(),
                       os.path.join(bob_ppo.log_dir, "episode_manager_latest.pt"))
            torch.save({"entropy_coef": alice_ppo.entropy_coef,
                        "abc_coef": bob_ppo.abc_coef,
                        "bob_success_buf": list(bob_success_buf), "best_bob_success_rate": best_bob_success_rate},
                       os.path.join(bob_ppo.log_dir, "train_state_latest.pt"))
            with open(os.path.join(run_dir, "latest_iter.txt"), "w") as _f:
                _f.write(str(bob_updates))
            print(f"  [Checkpoint] Saved latest_checkpoint.pt (iter {bob_updates})")

    alice_ppo.save(os.path.join(alice_ppo.log_dir, "latest_checkpoint.pt"), iteration=bob_updates)
    bob_ppo.save(os.path.join(bob_ppo.log_dir, "latest_checkpoint.pt"), iteration=bob_updates)
    bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
    torch.save(env.episode_manager.state_dict(),
               os.path.join(bob_ppo.log_dir, "episode_manager_latest.pt"))
    torch.save({"entropy_coef": alice_ppo.entropy_coef,
                "abc_coef": bob_ppo.abc_coef,
                "bob_success_buf": list(bob_success_buf), "best_bob_success_rate": best_bob_success_rate},
               os.path.join(bob_ppo.log_dir, "train_state_latest.pt"))
    with open(os.path.join(run_dir, "latest_iter.txt"), "w") as _f:
        _f.write(str(bob_updates))
    print("  Saved final models")
    writer.close()

    print(f"\n{'='*80}\nTRAINING COMPLETE ({bob_updates} iterations)\n{'='*80}")
    print(f"To resume:\n  python -m asyncDualPlayPPO.train_i_tasp_dpose_bobpen --exp_name {args.exp_name} \\")
    print(f"    --chkpt_alice runs/{args.exp_name}/alice/latest_checkpoint.pt \\")
    print(f"    --chkpt_bob   runs/{args.exp_name}/bob/latest_checkpoint.pt \\")
    print(f"    --char_length {args.char_length} --dpose_threshold {args.dpose_threshold} \\")
    print(f"    --resume_iteration {bob_updates}\n{'='*80}\n")


if __name__ == "__main__":
    main()
