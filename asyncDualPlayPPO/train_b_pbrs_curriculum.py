"""
Push-PPO PBRS Model B — PBRS dense reward with forced curriculum

Single-agent PPO with push primitive macro-actions.  Uses potential-based
reward shaping (PBRS) for dense position+rotation feedback.  Forced
curriculum: phase 1 = position-only (w_rot=0), phase 2 = rotation ramp
once position error is consistently low.

Action space: 4D MultiCategorical (r, phi, length, theta) — object-relative.
  r, phi   = approach radius+angle relative to object
  length   = push length
  theta    = push orientation angle (world frame)
  Gripper always closed — no control over it.

Architecture:
  Agent predicts 4D push params → push waypoints → cuRobo IK per
  waypoint → physics step → PBRS reward after push completes → PPO update.

Run locally:
  python -m asyncDualPlayPPO.train_b_pbrs_curriculum --num_envs 16 --max_iterations 500 --exp_name push_pbrs_b
"""

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
    parser = argparse.ArgumentParser(description="Push-PPO PBRS Model B")
    parser.add_argument("--exp_name", type=str, default="push_pbrs_b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--chkpt", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--resume_iteration", type=int, default=0,
                        help="Iteration to start from when resuming")
    parser.add_argument("--resume_best_sr", type=float, default=-1.0,
                        help="Best success rate to restore on resume")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write terminal output to this file as well")
    parser.add_argument("--with_distractor", action="store_true",
                        help="Spawn a random cube/cylinder as clutter (no goal)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    rel_obs = True
    rel_act = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ── Log file: opened once, written to alongside every print() ────────────
    _log_fh = None
    if args.log_file:
        _log_fh = open(args.log_file, "a", buffering=1)
        print(f"[Init] Logging to {args.log_file}", flush=True)

    def _pr(msg: str = "", end: str = "\n"):
        """Print to stdout AND log file (if active)."""
        sys.stdout.write(msg + end)
        sys.stdout.flush()
        if _log_fh:
            _log_fh.write(msg + end)
            _log_fh.flush()

    import torch
    import numpy as np
    import copy
    import gymnasium as gym_mc
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
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper
    from asyncDualPlayPPO.tasks.utils.action_push import (
        decode_push_action, compute_push_waypoints, total_push_substeps,
    )
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
    from asyncDualPlayPPO.algorithms.rl.ppo.storage import RolloutStorage

    from asyncDualPlayPPO.tasks.utils.reward_pbrs import (
        PBRS_K_POS, PBRS_K_ROT, PBRS_W_POS, PBRS_W_ROT,
        PBRS_POS_THRESHOLD, PBRS_COS_ROT_THRESHOLD,
        PBRS_COMPLETION_BONUS, PBRS_ROTATION_BONUS, PBRS_TIP_PENALTY,
        potential_pos, potential_rot, compute_pbrs_reward, check_done_pbrs,
    )

    ppo_cfg_path = os.path.join(os.path.dirname(__file__), "cfg/ppo/ppo_continuous.yaml")
    ppo_cfg = load_cfg(ppo_cfg_path)

    # ── Push hyperparameters ─────────────────────────────────────────────────
    max_pushes_per_episode = 5
    push_nsteps = 15          # pushes per PPO rollout (per env) — fixed temporal window

    noptepochs = 3

    # Dynamic minibatches: keep mini-batch size roughly constant so GPU memory
    # usage doesn't explode at high env counts.  Aim for ~16 envs per minibatch.
    envs_per_minibatch = 16
    nminibatches = max(1, args.num_envs // envs_per_minibatch)
    while nminibatches > 1 and args.num_envs % nminibatches != 0:
        nminibatches -= 1

    cliprange = 0.2
    ent_coef = 0.002
    gamma = 0.95
    lam = 0.95
    learning_rate = 3e-4

    _policy_cfg = ppo_cfg["params"]["policy"]

    num_cat_dims = 4
    num_bins = 21
    use_lstm = _policy_cfg.get("use_lstm", True)

    print(f"[Push-PPO] Config: {push_nsteps} pushes/rollout, {num_cat_dims}D×{num_bins} bins, LSTM={use_lstm}")

    # ── Environment config ────────────────────────────────────────────────────
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )

    # Distractor: random cube as physics clutter (no goal, not in observations)
    from asyncDualPlayPPO.tasks.utils.reach_dual_arm_diffik_env_cfg import spawn_random_block
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
        print("[Config] Distractor ENABLED — random cube as clutter.")
    else:
        env_cfg.scene.cube = None

    print("Creating environment...")
    with SuppressAllOutput():
        base_env = ManagerBasedRLEnv(cfg=env_cfg)

    env = PushEnvWrapper(
        env=base_env,
        device=base_env.device,
        num_objects=1,
        max_pushes_per_episode=max_pushes_per_episode,
        headless=args.headless,
        rel_obs=rel_obs,
    )
    print(f"Environment ready (rel_obs={rel_obs}, obs_dim={env.obs_dim}D).")

    # ── cuRobo IK solver ──────────────────────────────────────────────────────
    print("[cuRobo] Initialising IK solver...")
    _tensor_args = TensorDeviceType(device=torch.device(env.device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    # ── Tune LBFGS solver: fewer iterations, same quality for dense waypoints
    #     (Don't touch particle_optimizer — its sample buffers are pre-allocated)
    _ik_config.solver.newton_optimizer.n_iters = 30       # was 100
    _ik_config.solver.newton_optimizer.inner_iters = 10   # was 25
    ik_solver = IKSolver(_ik_config)
    print("[cuRobo] IK solver created (tuned for batch).")

    # Warm-up CUDA graph
    print(f"[cuRobo] Warming up CUDA graph for N={env.num_envs} envs...")
    _wup_pos = torch.zeros(env.num_envs, 3, device=env.device)
    _wup_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32).expand(env.num_envs, 4)
    ik_solver.solve_batch(
        CuroboPose(position=_wup_pos, quaternion=_wup_quat),
        seed_config=torch.zeros(env.num_envs, 1, 6, device=env.device),
        retract_config=torch.zeros(env.num_envs, 6, device=env.device),
    )
    print("[cuRobo] Warm-up done.")

    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)

    _debug_per_env = args.num_envs <= 50

    # ── Goal marker visualizer (VisualizationMarkers — no physics, no collision) ──
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
        """Sync VisualizationMarkers with env.goal_pos_euler (no-op if no goals)."""
        goals = env.goal_pos_euler  # (N, 6) local [x,y,z,roll,pitch,yaw]
        N = env.num_envs
        origins = env.env.scene.env_origins
        pos = goals[:, :3].clone()
        pos[:, 2] = 0.001  # flat on table
        euler = goals[:, 3:6].clone()
        euler[:, 0] = 0.0  # zero roll
        euler[:, 1] = 0.0  # zero pitch
        quat = _euler_to_quat(euler)
        _goal_viz.visualize(translations=pos + origins, orientations=quat)

    # Import _euler_to_quat from wrapper_push for the helper
    from asyncDualPlayPPO.tasks.utils.wrapper_push import _euler_to_quat
    env._euler_to_quat_imported = True

    # ── Push debug markers: green sphere at start, red at end, blue cylinder arrow ──
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

    _ident_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    def _update_push_markers(Xs, Ys, Xf, Yf, theta):
        try:
            N = Xs.shape[0]
            origins = env.env.scene.env_origins
            z_table = 0.002
            ident = _ident_quat.to(env.device).expand(N, 4)

            start_pos = torch.stack([Xs, Ys, torch.full((N,), z_table, device=env.device)], dim=-1) + origins
            end_pos   = torch.stack([Xf, Yf, torch.full((N,), z_table, device=env.device)], dim=-1) + origins

            _push_viz_start.visualize(translations=start_pos, orientations=ident)
            _push_viz_end.visualize(translations=end_pos, orientations=ident)

            # Cylinder at midpoint, oriented along push direction
            mid_w = torch.stack([(Xs + Xf) / 2, (Ys + Yf) / 2,
                                 torch.full((N,), z_table, device=env.device)], dim=-1) + origins
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
            pass  # markers are non-critical debug visualisation

    # Workspace clamp limits (local / env-origin-relative frame, metres)
    _WS_X = (-0.50, 0.50)
    _WS_Y = (0.25,  0.70)
    _WS_Z = ( 0.25, 0.55)  # floor = tool0 min reachable Z (TCP ~0.093 + offset ~0.139)

    # ── Robot body/joint indices ──────────────────────────────────────────────
    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    # Calibrate total IK→physics error: cuRobo targets tool0 but physics
    # model has Robotiq gripper merged into wrist_3.  Measure where the
    # fingers actually end up vs where we asked cuRobo to put tool0.
    print("[Setup] Calibrating IK→physics error...")
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

    # ── PPO agent ─────────────────────────────────────────────────────────────
    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    agent_cfg = copy.deepcopy(ppo_cfg["params"])
    agent_cfg["learn"]["nsteps"] = push_nsteps
    agent_cfg["learn"]["noptepochs"] = noptepochs
    agent_cfg["learn"]["nminibatches"] = nminibatches
    agent_cfg["learn"]["cliprange"] = cliprange
    agent_cfg["learn"]["ent_coef"] = ent_coef
    agent_cfg["learn"]["gamma"] = gamma
    agent_cfg["learn"]["lam"] = lam
    agent_cfg["learn"]["optim_stepsize"] = learning_rate
    agent_cfg["policy"]["num_bins"] = num_bins
    agent_cfg["policy"]["num_cat_dims"] = num_cat_dims

    agent = PPO(
        vec_env=env,
        cfg_train=agent_cfg,
        device=env.device,
        sampler="sequential",
        log_dir=f"runs/{args.exp_name}/agent",
        asymmetric=False,
    )
    agent.observation_space = env.observation_space
    agent.state_space = env.state_space
    agent.action_space = _mc_space
    agent.desired_kl = None

    agent.actor_critic = ActorCriticPush(
        agent.observation_space.shape,
        agent.state_space.shape,
        agent.action_space.shape,
        agent.init_noise_std,
        agent.model_cfg,
        asymmetric=False,
    ).to(env.device)

    agent.storage = RolloutStorage(
        agent.vec_env.num_envs,
        push_nsteps,
        agent.observation_space.shape,
        agent.state_space.shape,
        agent.action_space.shape,
        agent.device,
        "sequential",
    )

    agent.optimizer = torch.optim.Adam(
        agent.actor_critic.parameters(), lr=agent.learning_rate,
    )

    if args.chkpt and os.path.isfile(args.chkpt):
        agent.load(args.chkpt)
        print(f"[Resume] Loaded agent from {args.chkpt}")

    # ── LSTM hidden state ─────────────────────────────────────────────────────
    if use_lstm:
        _lsz = agent.actor_critic.lstm_hidden_size
        hidden_state = [
            torch.zeros(env.num_envs, _lsz, device=env.device),
            torch.zeros(env.num_envs, _lsz, device=env.device),
        ]
    else:
        hidden_state = None

    # ── Training state ────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}/summary")
    run_dir = os.path.abspath(f"runs/{args.exp_name}")
    best_success_rate = -1.0
    iteration = args.resume_iteration if args.chkpt else 0
    rew_buf     = deque(maxlen=push_nsteps * env.num_envs)
    sr_buf      = deque(maxlen=push_nsteps * env.num_envs)
    rot_sr_buf  = deque(maxlen=push_nsteps * env.num_envs)
    pos_err_buf = deque(maxlen=push_nsteps * env.num_envs)
    rot_err_buf = deque(maxlen=push_nsteps * env.num_envs)
    dense_pos_buf = deque(maxlen=push_nsteps * env.num_envs)
    dense_rot_buf = deque(maxlen=push_nsteps * env.num_envs)
    completion_buf = deque(maxlen=push_nsteps * env.num_envs)
    rot_bonus_buf = deque(maxlen=push_nsteps * env.num_envs)
    tip_buf = deque(maxlen=push_nsteps * env.num_envs)
    ema_rew     = 0.0

    # ── PBRS state ────────────────────────────────────────────────────────────
    prev_phi_pos = torch.zeros(env.num_envs, device=env.device)
    prev_phi_rot = torch.zeros(env.num_envs, device=env.device)
    gave_completion = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    gave_rot_bonus = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # ── Curriculum state ──────────────────────────────────────────────────────
    curriculum_active = False
    curriculum_ramp_start = None
    CURRICULUM_RAMP_ITERS = 200
    CURRICULUM_POS_THRESHOLD = 0.08
    CURRICULUM_LOOKBACK = 50
    ema_pos_err_hist = deque(maxlen=CURRICULUM_LOOKBACK)
    w_pos = PBRS_W_POS
    w_rot = 0.0
    pos_term_threshold = 0.05

    print(f"\n{'='*80}\nPUSH-PPO PBRS-B (curriculum): {args.exp_name}\nLOG DIR: {run_dir}\n{'='*80}\n")

    # ── Init environment ──────────────────────────────────────────────────────
    print("Initialising environment...")
    with SuppressAllOutput():
        obs = env.reset()
    _update_goal_markers()
    print("Training loop starting...")
    sys.stdout.flush()

    # ── Close gripper once (always closed in push primitive) ──────────────
    _close_act = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
    _close_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
    _close_act[:, 6] = -1.0
    env.step(_close_act)

    # Per-env state accumulators
    episode_reward = torch.zeros(env.num_envs, device=env.device)
    ee_pos_local = _tcp_pos_local()
    ee_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()
    prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

    # ── SIGTERM handler ──────────────────────────────────────────────────────
    _shutdown_requested = False

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        print("[INFO] SIGTERM received — checkpoint after current iteration.", flush=True)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    def _capture_pbrs_potentials(obs_t):
        nonlocal prev_phi_pos, prev_phi_rot
        obj_pos_c = obs_t[:, env.robot_dim:env.robot_dim + 3]
        obj_euler_c = obs_t[:, env.robot_dim + 3:env.robot_dim + 6]
        goal_pos_c = obs_t[:, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
        goal_euler_c = obs_t[:, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]
        prev_phi_pos = potential_pos(obj_pos_c, goal_pos_c)
        prev_phi_rot = potential_rot(obj_euler_c[:, 2], goal_euler_c[:, 2])

    # ── TRAINING LOOP ────────────────────────────────────────────────────────
    while iteration < args.max_iterations:
        agent.storage.clear()

        total_ik_fails = 0
        total_ik_steps = 0
        env.episode_push_counts.clear()
        env.episode_successes.clear()

        obs_pre_push = obs.clone()
        env.capture_pre_push(obs)
        _capture_pbrs_potentials(obs)

        for push_step in range(push_nsteps):
            # ── Agent predicts push action ────────────────────────────────────
            with torch.no_grad():
                h_in = (hidden_state[0], hidden_state[1]) if hidden_state else None
                actions, log_prob, value, mu, sigma, stored_h_in, new_h = agent.actor_critic.act_with_hidden(
                    obs, None, h_in,
                )
                if hidden_state is not None and new_h is not None:
                    hidden_state[0] = new_h[0]
                    hidden_state[1] = new_h[1]

            obj_x = obs[:, env.robot_dim]
            obj_y = obs[:, env.robot_dim + 1]
            obj_yaw = obs[:, env.robot_dim + 5]
            obj_xy = torch.stack([obj_x, obj_y], dim=-1)
            Xs, Ys, length, theta = decode_push_action_relative(
                actions, obj_xy, obj_yaw, num_bins=num_bins,
            )
            Xf = Xs + length * torch.cos(theta)
            Yf = Ys + length * torch.sin(theta)

            waypoints = compute_push_waypoints(
                Xs=Xs, Ys=Ys, length=length, theta=theta,
                current_ee_pos=ee_pos_local,
                current_ee_quat=ee_quat_w,
                device=env.device,
            )

            _update_push_markers(Xs, Ys, Xf, Yf, theta)

            if _debug_per_env:
                has_len = length.abs() > 0.001
                for e in range(env.num_envs):
                    if has_len[e]:
                        r_e = float(torch.sqrt((Xs[e] - obs_pre_push[e, env.robot_dim])**2 + (Ys[e] - obs_pre_push[e, env.robot_dim + 1])**2).item())
                        _pr(
                            f"    env {e}: bins=({', '.join(f'{int(actions[e,i].item()):2d}' for i in range(4))})  "
                            f"r={r_e:.3f} len={float(length[e]):.3f} θ={math.degrees(float(theta[e])):.0f}°  "
                            f"→ Xf={float(Xf[e]):+.3f} Yf={float(Yf[e]):+.3f}"
                        )

            # ── Execute push trajectory (gripper always closed) ────────────────
            terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for wp_idx, (wp_pos, wp_quat, _wp_grip) in enumerate(waypoints):
                # Arm-through-table check: reject waypoints below table surface
                arm_below_table = wp_pos[:, 2] < 0.005
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
                if arm_below_table.any():
                    ik_ok[arm_below_table] = False
                cur_joints = _robot_scene.data.joint_pos[:, _arm_jids]

                total_ik_steps += env.num_envs
                total_ik_fails += int((~ik_ok).sum().item())

                solved = result.solution.view(env.num_envs, 6)
                elbow_bad = solved[:, 2] < 0.0
                if elbow_bad.any():
                    ik_ok[elbow_bad] = False
                raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, prev_joint_cmd)
                if terminated.any():
                    raw_cmd[terminated] = cur_joints[terminated]
                prev_joint_cmd = raw_cmd.detach().clone()

                env_full = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
                env_full[:, :6] = raw_cmd
                env_full[:, 6] = -1.0  # always closed

                obs, _, step_terminated, truncated, _ = env.step(env_full)
                terminated |= step_terminated

            # ── After push: compute PBRS reward & done ───────────────────────
            cur_obj_pos = obs[:, env.robot_dim:env.robot_dim + 3]
            cur_obj_euler = obs[:, env.robot_dim + 3:env.robot_dim + 6]
            goal_pos = obs[:, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
            goal_euler = obs[:, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]

            pbrs_result = compute_pbrs_reward(
                cur_obj_pos, cur_obj_euler, goal_pos, goal_euler,
                prev_phi_pos, prev_phi_rot, gave_completion, gave_rot_bonus,
                w_pos=w_pos, w_rot=w_rot, enable_rot_sparse=curriculum_active,
            )

            reward = pbrs_result["reward"]
            reward[terminated] = -10.0

            gave_completion = pbrs_result["gave_completion"]
            gave_rot_bonus = pbrs_result["gave_rot_bonus"]
            prev_phi_pos = pbrs_result["phi_pos_now"]
            prev_phi_rot = pbrs_result["phi_rot_now"]

            env.at_goal = pbrs_result["at_goal"]
            env._last_pos_err = pbrs_result["pos_err"]
            env._last_rot_err = pbrs_result["cos_rot_err"]
            env.push_count += 1

            done, done_reasons = check_done_pbrs(
                obs, terminated, env.push_count, max_pushes_per_episode,
                pbrs_result["at_goal"], robot_dim=env.robot_dim,
                obj_state_dim=env.obj_state_dim,
                pos_term_threshold=pos_term_threshold,
            )
            catastrophe = done_reasons["launched"] | done_reasons["tipped"] | done_reasons["oob"]
            reward[catastrophe & ~terminated] = -10.0
            episode_reward += reward

            # ── Record transition ────────────────────────────────────────────
            agent.storage.add_transitions(
                obs_pre_push, obs_pre_push, actions, reward, done,
                value, log_prob, mu, sigma,
                masks=(~done).float(),
                hidden_state=stored_h_in,
            )

            rew_buf.extend(reward.cpu().tolist())
            cur_at_goal = pbrs_result["at_goal"].float()
            sr_buf.extend(cur_at_goal.cpu().tolist())
            rot_sr_buf.extend((pbrs_result["cos_rot_err"] < PBRS_COS_ROT_THRESHOLD).float().cpu().tolist())
            pos_err_buf.extend(pbrs_result["pos_err"].cpu().tolist())
            rot_err_buf.extend(pbrs_result["cos_rot_err"].cpu().tolist())
            dense_pos_buf.extend(pbrs_result["dense_pos"].cpu().tolist())
            dense_rot_buf.extend(pbrs_result["dense_rot"].cpu().tolist())
            completion_buf.extend(pbrs_result["pos_success"].float().cpu().tolist())
            rot_bonus_buf.extend(pbrs_result["rot_success"].float().cpu().tolist())
            tip_buf.extend(pbrs_result["tipped"].float().cpu().tolist())

            # Per-push compact summary
            _cur_phase = "P1" if not curriculum_active else (f"P2({w_rot:.1f})" if w_rot < PBRS_W_ROT else "P3")
            _pr(
                f"  [Push {push_step:3d}|{_cur_phase}] "
                f"rew={reward.mean().item():+.3f}  "
                f"dense_pos={pbrs_result['dense_pos'].mean().item():+.4f}  "
                f"dense_rot={pbrs_result['dense_rot'].mean().item():+.4f}  "
                f"pos_err={pbrs_result['pos_err'].mean().item():.4f}  "
                f"cos_rot={pbrs_result['cos_rot_err'].mean().item():.4f}  "
                f"at_goal={cur_at_goal.sum().item():.0f}/{env.num_envs}  "
                f"w_rot={w_rot:.1f} pos_term={pos_term_threshold:.3f}"
            )

            if _debug_per_env:
                for e in range(env.num_envs):
                    _flags = []
                    if pbrs_result["pos_success"][e]:
                        _flags.append("POS_OK")
                    if pbrs_result["rot_success"][e]:
                        _flags.append("ROT_OK")
                    if pbrs_result["tipped"][e]:
                        _flags.append("TIPPED")
                    if terminated[e]:
                        _flags.append("TERM")
                    if done[e]:
                        _flags.append("DONE")
                    _flag_str = " ".join(_flags) if _flags else ""
                    _pr(
                        f"    env {e:3d}: rew={reward[e].item():+.3f}  "
                        f"Φp={prev_phi_pos[e].item():.3f}→{pbrs_result['phi_pos_now'][e].item():.3f}  "
                        f"Φr={prev_phi_rot[e].item():.3f}→{pbrs_result['phi_rot_now'][e].item():.3f}  "
                        f"d_pos={pbrs_result['dense_pos'][e].item():+.4f}  "
                        f"d_rot={pbrs_result['dense_rot'][e].item():+.4f}  "
                        f"pos_err={pbrs_result['pos_err'][e].item():.4f}m  "
                        f"cos_rot={pbrs_result['cos_rot_err'][e].item():.4f}  "
                        f"{_flag_str}"
                    )

            # ── Handle done envs ──────────────────────────────────────────────
            if done.any():
                done_ids = torch.where(done)[0]
                # Snapshot goal/object positions before reset clears them
                obj_pos_done   = obs[done_ids, env.robot_dim:env.robot_dim + 3]
                obj_euler_done = obs[done_ids, env.robot_dim + 3:env.robot_dim + 6]
                goal_pos_done  = obs[done_ids, env.robot_dim + env.obj_state_dim:
                                      env.robot_dim + env.obj_state_dim + 3]
                goal_euler_done = obs[done_ids, env.robot_dim + env.obj_state_dim + 3:
                                       env.robot_dim + env.obj_state_dim + 6]
                pos_err_done   = (obj_pos_done - goal_pos_done).norm(dim=-1)
                rot_diff = (obj_euler_done - goal_euler_done) % (2.0 * torch.pi)
                rot_diff = torch.where(rot_diff > torch.pi, 2.0 * torch.pi - rot_diff, rot_diff)
                rot_err_done  = rot_diff.max(dim=-1)[0]
                ep_rews_done = episode_reward[done_ids].clone()
                episode_reward[done_ids] = 0.0
                ep_pushes_pre = len(env.episode_push_counts)
                env.reset_done_envs(done)
                _update_goal_markers()
                ep_pushes_post = len(env.episode_push_counts)
                n_new = ep_pushes_post - ep_pushes_pre
                if n_new > 0:
                    new_pushes = env.episode_push_counts[-n_new:]
                    new_successes = env.episode_successes[-n_new:]
                    for i, (p, s) in enumerate(zip(new_pushes, new_successes)):
                        status = "SUCCESS" if s else "fail"
                        gi = min(i, len(done_ids) - 1)
                        eid = done_ids[gi]
                        g_pos = goal_pos_done[gi]
                        g_rot = goal_euler_done[gi]
                        o_pos = obj_pos_done[gi]
                        o_rot = obj_euler_done[gi]
                        s_pos = env._ep_start_pos[max(0, eid)]
                        s_rot = env._ep_start_euler[max(0, eid)]
                        pe = pos_err_done[gi]
                        re = float(rot_err_done[gi])
                        er = float(ep_rews_done[gi])
                        _reason_parts = []
                        if done_reasons["success"][eid]:
                            _reason_parts.append("SUCCESS")
                        if done_reasons["max_pushes"][eid]:
                            _reason_parts.append("MAX_PUSH")
                        if done_reasons["tipped"][eid]:
                            _reason_parts.append("TIPPED")
                        if done_reasons["launched"][eid]:
                            _reason_parts.append("LAUNCHED")
                        if done_reasons["oob"][eid]:
                            _reason_parts.append("OOB")
                        if done_reasons["terminated"][eid]:
                            _reason_parts.append("PHYSICS")
                        if done_reasons["pos_only"][eid]:
                            _reason_parts.append("POS_ONLY")
                        _reason_str = "+".join(_reason_parts) if _reason_parts else "UNKNOWN"
                        _pr(
                            f"  [Episode] pushes={p}  {status}  end={_reason_str}  rew={er:+.3f}  "
                            f"start=({s_pos[0]:+.3f},{s_pos[1]:+.3f},{s_pos[2]:+.3f}) "
                            f"yaw={s_rot[2]:+.3f}  "
                            f"goal=({g_pos[0]:+.3f},{g_pos[1]:+.3f},{g_pos[2]:+.3f}) "
                            f"yaw={g_rot[2]:+.3f}  "
                            f"final=({o_pos[0]:+.3f},{o_pos[1]:+.3f},{o_pos[2]:+.3f}) "
                            f"yaw={o_rot[2]:+.3f}  "
                            f"err_pos={pe:.3f}m  err_rot={re:.3f}rad"
                        )
                if hidden_state is not None:
                    hidden_state[0][done] = 0.0
                    hidden_state[1][done] = 0.0
                # Reset PBRS state for done envs
                prev_phi_pos[done] = 0.0
                prev_phi_rot[done] = 0.0
                gave_completion[done] = False
                gave_rot_bonus[done] = False
                # Reset all done envs — terminated envs need explicit reset too
                # so their observations don't hold exploded/auto-reset state.
                needs_reset = done
                if needs_reset.any():
                    reset_ids = torch.where(needs_reset)[0]
                    env.env.reset(env_ids=reset_ids)
                    env._randomize_object_spawn(reset_ids)
                    env._sample_goals_filtered(reset_ids)
                    env._update_goal_in_extras()
                    env._move_goal_ghost(reset_ids)
                    env._ep_started[reset_ids] = False
                    obs_new = env._get_push_obs()
                    obs[needs_reset] = obs_new[needs_reset]
                    _update_goal_markers()
                ee_pos_local[done] = _tcp_pos_local()[done]
                ee_quat_w[done] = _QUAT_TOOL_DOWN.expand(done.sum().item(), 4).to(env.device)
                prev_joint_cmd[done] = _robot_scene.data.joint_pos[:, _arm_jids][done]

            # ── Update EE position tracker for next push ─────────────────────
            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(env.num_envs, 4).clone()

            obs_pre_push = obs.clone()
            env.capture_pre_push(obs)
            _capture_pbrs_potentials(obs)

        # ── PPO UPDATE ────────────────────────────────────────────────────────
        with torch.no_grad():
            last_val = agent.actor_critic.critic(obs)
        agent.storage.compute_returns(last_val, agent.gamma, agent.lam)
        loss_val, loss_surr = agent.update()
        agent.storage.clear()

        mean_rew    = np.mean(rew_buf) if rew_buf else 0.0
        mean_pos_err = np.mean(pos_err_buf) if pos_err_buf else 0.0
        mean_rot_err = np.mean(rot_err_buf) if rot_err_buf else 0.0
        sr = np.mean(sr_buf) if sr_buf else 0.0
        rot_sr = np.mean(rot_sr_buf) if rot_sr_buf else 0.0
        ik_fail_rate = total_ik_fails / max(1, total_ik_steps)
        mean_dense_pos = np.mean(dense_pos_buf) if dense_pos_buf else 0.0
        mean_dense_rot = np.mean(dense_rot_buf) if dense_rot_buf else 0.0
        mean_completion = np.mean(completion_buf) if completion_buf else 0.0
        mean_rot_bonus = np.mean(rot_bonus_buf) if rot_bonus_buf else 0.0
        mean_tip = np.mean(tip_buf) if tip_buf else 0.0

        ema_rew = 0.9 * ema_rew + 0.1 * mean_rew

        ep_push_counts = list(env.episode_push_counts)
        ep_successes = list(env.episode_successes)
        avg_pushes = np.mean(ep_push_counts) if ep_push_counts else float("nan")
        ep_sr = np.mean(ep_successes) if ep_successes else float("nan")
        n_episodes = len(ep_push_counts)

        loss_delta = loss_surr - getattr(agent, "_last_loss_surr", 0.0)
        agent._last_loss_surr = loss_surr

        writer.add_scalar("Loss/Agent/Value",        loss_val,      iteration)
        writer.add_scalar("Loss/Agent/Surrogate",    loss_surr,     iteration)
        writer.add_scalar("Reward/Mean",             mean_rew,      iteration)
        writer.add_scalar("Reward/EMA",              ema_rew,       iteration)
        writer.add_scalar("Metrics/SuccessRate",     sr,            iteration)
        writer.add_scalar("Metrics/RotationSR",       rot_sr,        iteration)
        writer.add_scalar("Metrics/PosError",        mean_pos_err,  iteration)
        writer.add_scalar("Metrics/RotError",        mean_rot_err,  iteration)
        writer.add_scalar("Metrics/IKFailRate",      ik_fail_rate,  iteration)
        writer.add_scalar("Metrics/EpisodicSR",      ep_sr if not np.isnan(ep_sr) else 0.0, iteration)
        writer.add_scalar("Metrics/AvgPushesPerEpisode", avg_pushes if not np.isnan(avg_pushes) else 0.0, iteration)
        writer.add_scalar("Metrics/Episodes",        n_episodes,    iteration)
        writer.add_scalar("PBRS/DensePos",           mean_dense_pos, iteration)
        writer.add_scalar("PBRS/DenseRot",           mean_dense_rot, iteration)
        writer.add_scalar("PBRS/CompletionRate",     mean_completion, iteration)
        writer.add_scalar("PBRS/RotBonusRate",       mean_rot_bonus, iteration)
        writer.add_scalar("PBRS/TipRate",            mean_tip,      iteration)
        writer.add_scalar("Curriculum/w_rot",        w_rot,         iteration)
        writer.add_scalar("Curriculum/phase",        2 if curriculum_active else 1, iteration)
        writer.add_scalar("Curriculum/pos_term_threshold", pos_term_threshold, iteration)

        # ── Curriculum check ──────────────────────────────────────────────────
        ema_pos_err_hist.append(mean_pos_err)
        if not curriculum_active and len(ema_pos_err_hist) == CURRICULUM_LOOKBACK:
            if all(e < CURRICULUM_POS_THRESHOLD for e in ema_pos_err_hist):
                curriculum_active = True
                curriculum_ramp_start = iteration
                _pr(f"[Curriculum] Phase 2: rotation reward ramp started at iter {iteration}")

        if curriculum_active:
            ramp_progress = min(1.0, (iteration - curriculum_ramp_start) / CURRICULUM_RAMP_ITERS)
            w_rot = PBRS_W_ROT * ramp_progress
            pos_term_threshold = 0.05 - 0.03 * ramp_progress
        else:
            w_rot = 0.0
            pos_term_threshold = 0.05
        w_pos = PBRS_W_POS

        # Single compact iteration line — machine-parseable
        avg_pushes_str = f"{avg_pushes:.1f}" if not np.isnan(avg_pushes) else "nan"
        trend = "↓" if loss_delta < -0.01 else ("↑" if loss_delta > 0.01 else "→")
        _mode = "pbrs_b"
        _pr(
            f"[Iter {iteration:5d}] "
            f"Loss={loss_surr:.4f}{trend} | Val={loss_val:.4f} | "
            f"Rew={mean_rew:+.4f} (EMA {ema_rew:+.4f}) | "
            f"PosErr={mean_pos_err:.4f} | RotErr={mean_rot_err:.4f} | SR={sr:.4f} | RotSR={rot_sr:.4f} | "
            f"IK_fail={ik_fail_rate:.3f} | "
            f"AvgPushes={avg_pushes_str} | Epi={n_episodes} | "
            f"BestSR={best_success_rate:.4f} | {_mode}"
        )
        sys.stdout.flush()

        if sr > best_success_rate:
            best_success_rate = sr
            agent.save(os.path.join(agent.log_dir, "model_best.pt"))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(agent.log_dir, "latest_checkpoint.pt"))
            with open(os.path.join(agent.log_dir, "latest_iter.txt"), "w") as _f:
                _f.write(str(iteration))
            print(f"  [Checkpoint] Saved latest_checkpoint.pt (iter {iteration})")

        rew_buf.clear()
        sr_buf.clear()
        rot_sr_buf.clear()
        pos_err_buf.clear()
        rot_err_buf.clear()
        dense_pos_buf.clear()
        dense_rot_buf.clear()
        completion_buf.clear()
        rot_bonus_buf.clear()
        tip_buf.clear()
        iteration += 1

        if _shutdown_requested:
            agent.save(os.path.join(agent.log_dir, "latest_checkpoint.pt"))
            with open(os.path.join(agent.log_dir, "latest_iter.txt"), "w") as _f:
                _f.write(str(iteration))
            print(f"[INFO] Emergency checkpoint saved (iter {iteration}). Shutting down.")
            break

    print(f"\nTraining complete. Best SR: {best_success_rate:.4f}")
    agent.save(os.path.join(agent.log_dir, "latest_checkpoint.pt"))
    with open(os.path.join(agent.log_dir, "latest_iter.txt"), "w") as _f:
        _f.write(str(iteration))
    simulation_app.close()


if __name__ == "__main__":
    main()
