"""
Validate a trained Push-ASP (Model C) Bob against test configurations.

Loads Bob's PPOABC model with GoalEncoder and evaluates Bob's ability to
push objects to predefined goal positions.  Supports both T-block and disc
objects in the same scene (30 tests: 10 disc + 10 pos-only + 10 pos+rot).
Alice is optionally loaded for reference.  Visual markers, airborne
detection, and per-push logging.

Usage:
  python -m asyncDualPlayPPO.tests.validate_push_asp \
      --chkpt_bob runs/hpc_pbrs_asp_528env/bob/model_best.pt \
      --num_tests 30 --headless --csv results_asp.csv
"""

import argparse
import signal
import os
import sys
import math
from dataclasses import dataclass
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose as CuroboPose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml
except ModuleNotFoundError:
    print("[ERROR] cuRobo not found.")
    sys.exit(1)

import torch
import torch._dynamo    # noqa
import torch._C         # noqa
import torch.optim      # noqa

from isaaclab.app import AppLauncher

from asyncDualPlayPPO.tasks.utils.validation_configs import (
    ALL_TESTS, get_test_config, get_test_count,
)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25, 0.70)
_WS_Z = (0.25, 0.55)


@dataclass
class ASPValidationResult:
    test_index: int
    test_name: str
    test_type: str
    object_type: str
    success: bool
    pushes_used: int
    final_pos_error: float
    final_rot_error: float
    area_coverage: float
    trial_count: int = 1
    success_count: int = 0


def _euler_xyz_to_quat_local(euler):
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def _area_coverage(pos_err, rot_err):
    pc = max(0.0, 1.0 - pos_err / 0.10)
    rc = max(0.0, 1.0 - rot_err / 0.40)
    return pc * rc * 100.0


def main():
    parser = argparse.ArgumentParser(description="Validate Push-ASP Bob")
    parser.add_argument("--chkpt_bob", type=str, required=True, help="Path to Bob checkpoint")
    parser.add_argument("--chkpt_alice", type=str, default=None, help="Path to Alice checkpoint")
    parser.add_argument("--num_tests", type=int, default=30, help="Number of test scenes")
    parser.add_argument("--max_pushes", type=int, default=30, help="Max pushes per test")
    parser.add_argument("--max_tries", type=int, default=3, help="Max retries per test")
    parser.add_argument("--rot_threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians (default 0.2)")
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV file")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    signal.signal(signal.SIGINT, lambda *_: (simulation_app.close(), os._exit(1)))

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.utils import configclass
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import SceneEntityCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push_asp import PushASPEnvWrapper, _OBS_ROBOT_DIM
    from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints
    from asyncDualPlayPPO.tasks.utils.action_push_relative import decode_push_action_relative
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.tasks.utils.observations import (
        ee_poses as _obs_ee_poses,
        object_states as _obs_object_states,
        goal_states as _obs_goal_states,
        goal_distance as _obs_goal_distance,
    )
    import copy
    import gymnasium as gym_mc
    import yaml
    import numpy as np

    @configclass
    class ValidateEnvCfg(PushTaskCuRoboEnvCfg):
        @configclass
        class ValidateSceneCfg(PushTaskCuRoboEnvCfg.PushTaskSceneCfg):
            disc_object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DiscObject",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=[0.0, 0.0, -1.0],
                    rot=[0.0, 0.0, 0.0, 1.0],
                ),
                spawn=sim_utils.CylinderCfg(
                    radius=0.05,
                    height=0.06,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        max_depenetration_velocity=10000.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(density=300.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6,
                        dynamic_friction=0.6,
                        restitution=0.1,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.6, 0.9),
                    ),
                ),
            )
        scene: ValidateSceneCfg = ValidateSceneCfg(num_envs=1, env_spacing=2.5)

    ppo_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "cfg/ppo/ppo_continuous.yaml")
    with open(ppo_cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    num_cat_dims = 4
    num_bins = 21

    if os.path.isdir(args.chkpt_bob):
        for _cand in [
            os.path.join(args.chkpt_bob, "bob", "model_best.pt"),
            os.path.join(args.chkpt_bob, "bob", "latest_checkpoint.pt"),
        ]:
            if os.path.isfile(_cand):
                print(f"[Resolve] Directory given, using {_cand}")
                args.chkpt_bob = _cand
                break
        else:
            print(f"[ERROR] No checkpoint found in {args.chkpt_bob}/bob/")
            sys.exit(1)

    _chkpt = torch.load(args.chkpt_bob, map_location="cpu", weights_only=False)
    _chkpt_state = _chkpt.get("model_state_dict", _chkpt)
    _pi_w0 = _chkpt_state.get("pi_encoder.obj_encoder.0.weight")
    _is_noge = _pi_w0 is not None and _pi_w0.shape[1] == 22
    _has_goal_encoder = False if _is_noge else True
    print(f"[Detect] Checkpoint has pi_encoder.0.weight shape={_pi_w0.shape if _pi_w0 is not None else 'N/A'}, "
          f"GoalEncoder={_has_goal_encoder}")
    del _chkpt, _chkpt_state
    torch.cuda.empty_cache()

    chkpt_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.chkpt_bob)))
    if args.csv is None:
        args.csv = os.path.join(chkpt_run_dir, "validation_results_asp.csv")
    elif not os.path.isabs(args.csv):
        args.csv = os.path.join(chkpt_run_dir, args.csv)

    bob_cfg = copy.deepcopy(ppo_cfg["params"])
    bob_cfg["policy"]["use_pi_encoder"] = True
    bob_cfg["policy"]["use_multicategorical"] = True
    bob_cfg["policy"]["use_lstm"] = True
    bob_cfg["policy"]["use_goal_encoder"] = _has_goal_encoder
    bob_cfg["policy"]["num_cat_dims"] = num_cat_dims
    bob_cfg["policy"]["num_bins"] = num_bins
    bob_cfg["policy"]["robot_state_dim"] = 6
    if _has_goal_encoder:
        bob_cfg["policy"]["num_objects"] = 1
        bob_cfg["policy"]["goal_embed_dim"] = 8
    else:
        bob_cfg["policy"]["pi_obj_dim"] = 22

    env_cfg = ValidateEnvCfg()
    env_cfg.scene.num_envs = 1

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES,
        scale=1.0, use_default_offset=False,
    )

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushASPEnvWrapper(
        env=base_env, alice_pushes=5, bob_pushes=args.max_pushes,
        max_goals_per_episode=1, num_objects=1, rel_obs=True, device=device,
    )

    # ── Visual markers ─────────────────────────────────────────────────────────
    _blk_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/blocks")
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
    _goal_viz_disc = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkerDisc",
            markers={
                "disc": sim_utils.CylinderCfg(
                    radius=0.05,
                    height=0.001,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),
                ),
            },
        )
    )
    _push_viz_start = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushStart",
            markers={"sphere": sim_utils.SphereCfg(radius=0.015,
                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))},
        )
    )
    _push_viz_end = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushEnd",
            markers={"sphere": sim_utils.SphereCfg(radius=0.015,
                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))},
        )
    )
    _push_viz_arrow = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PushArrow",
            markers={"cylinder": sim_utils.CylinderCfg(radius=0.005, height=0.30,
                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)))},
        )
    )
    _ident_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    def _update_goal_marker(gx, gy, gyaw=0.0, obj_type="tblock"):
        origins = env.env.scene.env_origins
        pos = torch.tensor([[gx, gy, 0.001]], device=device) + origins
        euler = torch.zeros(1, 3, device=device)
        euler[0, 2] = gyaw
        quat = _euler_xyz_to_quat_local(euler)
        hide_pos = torch.tensor([[0.0, 0.0, -1.0]], device=device) + origins
        hide_quat = _QUAT_TOOL_DOWN.expand(1, 4)
        if obj_type == "disc":
            _goal_viz_disc.visualize(translations=pos, orientations=quat)
            _goal_viz.visualize(translations=hide_pos, orientations=hide_quat)
        else:
            _goal_viz.visualize(translations=pos, orientations=quat)
            _goal_viz_disc.visualize(translations=hide_pos, orientations=hide_quat)

    def _update_push_markers(Xs, Ys, Xf, Yf, angle):
        N = 1
        origins = env.env.scene.env_origins
        z_t = 0.002
        ident = _ident_quat.to(device).expand(N, 4)
        sp = torch.stack([Xs.float(), Ys.float(), torch.full((N,), z_t, device=device)], dim=-1) + origins
        ep = torch.stack([Xf.float(), Yf.float(), torch.full((N,), z_t, device=device)], dim=-1) + origins
        _push_viz_start.visualize(translations=sp, orientations=ident)
        _push_viz_end.visualize(translations=ep, orientations=ident)
        mid = torch.stack([(Xs + Xf) / 2, (Ys + Yf) / 2,
                           torch.full((N,), z_t, device=device)], dim=-1) + origins
        half = math.pi / 4
        ch, sh = math.cos(half), math.sin(half)
        aq = torch.stack([torch.full((N,), ch, device=device),
                          -sh * torch.sin(angle), sh * torch.cos(angle),
                          torch.zeros(N, device=device)], dim=-1)
        _push_viz_arrow.visualize(translations=mid, orientations=aq)

    # ── cuRobo IK ─────────────────────────────────────────────────────────────
    _tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    _ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    _robot_cfg = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
    _ik_config = IKSolverConfig.load_from_robot_config(
        _robot_cfg, world_model=None, tensor_args=_tensor_args,
    )
    _ik_config.solver.newton_optimizer.n_iters = 30
    _ik_config.solver.newton_optimizer.inner_iters = 10
    ik_solver = IKSolver(_ik_config)

    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    # ── IK→physics calibration ────────────────────────────────────────────────
    print("[Setup] Calibrating IK→physics error...")
    _calib_pos = torch.zeros(1, 3, device=device)
    _calib_pos[:, 1] = 0.60
    _calib_pos[:, 2] = 0.25
    _calib_cur = _robot_scene.data.joint_pos[:, _arm_jids]
    _calib_res = ik_solver.solve_batch(
        CuroboPose(position=_calib_pos, quaternion=_QUAT_TOOL_DOWN.expand(1, 4)),
        seed_config=_calib_cur.unsqueeze(1),
        retract_config=_calib_cur,
    )
    _calib_cmd = _calib_res.solution.view(1, 6)
    _calib_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _calib_act[:, :6] = _calib_cmd
    _calib_act[:, 6] = 1.0
    for _ in range(30):
        env.step(_calib_act)
    _finger_after = _tcp_pos_local()
    _TOTAL_IK_ERROR = (_finger_after - _calib_pos).clone()
    print(f"[Setup] IK error = ({float(_TOTAL_IK_ERROR[0,0]):+.3f}, "
          f"{float(_TOTAL_IK_ERROR[0,1]):+.3f}, {float(_TOTAL_IK_ERROR[0,2]):+.3f})")

    # ── Observation builder ───────────────────────────────────────────────────
    _ee_cfg = SceneEntityCfg("robot", body_names="wrist_3_link")
    _tblock_obj_cfg = SceneEntityCfg("target_object")
    _tblock_grip_cfg = SceneEntityCfg("robot", body_names="wrist_3_link")
    _disc_obj_cfg = SceneEntityCfg("disc_object")

    def _build_obs(obj_type="tblock"):
        obj_cfg = _disc_obj_cfg if obj_type == "disc" else _tblock_obj_cfg
        ee = _obs_ee_poses(env.env, _ee_cfg)
        obj = _obs_object_states(env.env, obj_cfg, _tblock_grip_cfg, None)
        goal = _obs_goal_states(env.env, obj_cfg)
        dist = _obs_goal_distance(env.env, obj_cfg)
        return torch.cat([ee, obj, goal, dist], dim=-1)

    # ── Load Bob ──────────────────────────────────────────────────────────────
    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    bob_ppo = PPOABC(
        vec_env=env, cfg_train=bob_cfg, device=device,
        sampler="sequential", log_dir="/tmp/validate_push_asp",
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
    ).to(device)

    if hasattr(bob_ppo.actor_critic, "_goal_proj") and bob_ppo.actor_critic._goal_proj is not None:
        with torch.no_grad():
            bob_ppo.actor_critic._goal_proj.weight.mul_(0.1)

    bob_ppo.load(args.chkpt_bob)
    bob_ppo.actor_critic.eval()
    _lsz = bob_ppo.actor_critic.lstm_hidden_size
    bob_hidden = [torch.zeros(1, _lsz, device=device), torch.zeros(1, _lsz, device=device)]
    print(f"[Validate] Loaded Bob from {args.chkpt_bob}")

    if args.chkpt_alice and os.path.isfile(args.chkpt_alice):
        alice_cfg = copy.deepcopy(ppo_cfg["params"])
        alice_cfg["policy"]["use_pi_encoder"] = True
        alice_cfg["policy"]["use_multicategorical"] = True
        alice_cfg["policy"]["use_lstm"] = True
        alice_cfg["policy"]["use_goal_encoder"] = False
        alice_cfg["policy"]["num_cat_dims"] = num_cat_dims
        alice_cfg["policy"]["num_bins"] = num_bins
        alice_cfg["policy"]["robot_state_dim"] = 6
        alice_ppo = PPO(
            vec_env=env, cfg_train=alice_cfg, device=device,
            sampler="sequential", log_dir="/tmp/validate_push_asp_alice",
            asymmetric=False,
        )
        alice_ppo.observation_space = env.alice_observation_space
        alice_ppo.state_space = alice_ppo.observation_space
        alice_ppo.action_space = _mc_space
        alice_ppo.desired_kl = None
        alice_ppo.actor_critic = alice_ppo.actor_critic.__class__(
            alice_ppo.observation_space.shape, alice_ppo.state_space.shape,
            alice_ppo.action_space.shape, alice_ppo.init_noise_std,
            alice_ppo.model_cfg, asymmetric=False,
        ).to(device)
        alice_ppo.load(args.chkpt_alice)
        alice_ppo.actor_critic.eval()
        print(f"[Validate] Loaded Alice from {args.chkpt_alice}")
    else:
        alice_ppo = None

    print(f"[Validate] Mode: rel_obs=True, GoalEncoder={'ON' if _has_goal_encoder else 'OFF (Model D)'}, "
          f"rot_threshold={args.rot_threshold:.3f} rad, max_pushes={args.max_pushes}")

    # ── Init ──────────────────────────────────────────────────────────────────
    obs = env.reset()
    env.env.sim.step()

    _close_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _close_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
    _close_act[:, 6] = -1.0
    env.step(_close_act)

    # ── Run tests ─────────────────────────────────────────────────────────────
    results: List[ASPValidationResult] = []
    test_cfgs_data: List[dict] = []
    n_tests = min(args.num_tests, get_test_count())

    for test_idx in range(1, n_tests + 1):
        cfg = get_test_config(test_idx)
        if cfg is None:
            continue

        _obj_type = cfg.object_type
        _active_obj_name = "disc_object" if _obj_type == "disc" else "target_object"
        _inactive_obj_name = "target_object" if _obj_type == "disc" else "disc_object"
        _spawn_z = 0.03 if _obj_type == "disc" else 0.05
        _min_r = 0.06 if _obj_type == "disc" else 0.04
        _max_r = 0.12 if _obj_type == "disc" else 0.08

        print(f"\n[Test {test_idx}/{n_tests}] {cfg.name} #{cfg.test_id}")

        TRIAL_COUNT = args.max_tries
        trial_successes = 0
        trial_pushes = []
        best_pos_err = float('inf')
        best_rot_err = float('inf')

        for trial in range(TRIAL_COUNT):

            _reset_jpos = _robot_scene.data.joint_pos.clone()
            _reset_jpos[:, _arm_jids] = _calib_cmd
            _robot_scene.write_joint_state_to_sim(
                _reset_jpos, torch.zeros_like(_reset_jpos),
            )

            _active_obj = env.env.scene[_active_obj_name]
            _inactive_obj = env.env.scene[_inactive_obj_name]
            _active_obj.write_root_pose_to_sim(torch.tensor([[
                cfg.main_start.x, cfg.main_start.y, _spawn_z, 1.0, 0.0, 0.0, 0.0
            ]], device=device))
            _inactive_obj.write_root_pose_to_sim(torch.tensor([[
                0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0
            ]], device=device))
            _active_obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            _inactive_obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
            env.env.sim.step()

            _update_goal_marker(cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw, _obj_type)

            goal_6d = torch.zeros(1, 6, device=device)
            goal_6d[0, 0] = cfg.main_goal_x
            goal_6d[0, 1] = cfg.main_goal_y
            goal_6d[0, 2] = 0.02
            goal_6d[0, 5] = cfg.main_goal_yaw

            env.episode_manager.goal_states = goal_6d
            env.episode_manager.current_phase[:] = 1
            env.episode_manager.phase_step[:] = 0
            env.episode_manager.goal_count[:] = 1
            env.episode_manager.goal_valid[:] = True
            env.episode_manager.completion_given[:] = False

            env._update_goal_in_extras()
            full_obs = _build_obs(_obj_type)
            bob_obs = full_obs.clone()

            _init_obj_pos = full_obs[0, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
            _init_obj_euler = full_obs[0, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
            _init_goal_pos = full_obs[0, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 14 + 3]
            _init_goal_euler = full_obs[0, _OBS_ROBOT_DIM + 14 + 3:_OBS_ROBOT_DIM + 14 + 6]
            _init_pos_err = (_init_obj_pos - _init_goal_pos).norm().item()
            _init_rot_err = _rot_distance_rad(_init_obj_euler.unsqueeze(0), _init_goal_euler.unsqueeze(0)).item()
            if _obj_type == "disc":
                _init_rot_err = 0.0
            _init_oob_2d = float((_init_obj_pos[:2] - _init_goal_pos[:2]).norm().item())
            print(f"  [{cfg.test_type}] goal=({cfg.main_goal_x:+.3f},{cfg.main_goal_y:+.3f}) yaw={cfg.main_goal_yaw:+.3f}  "
                  f"start=({cfg.main_start.x:+.3f},{cfg.main_start.y:+.3f})  "
                  f"init_pos_err={_init_pos_err:.4f}m  init_rot_err={_init_rot_err:.3f}rad")

            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

            bob_hidden[0].zero_()
            bob_hidden[1].zero_()

            trial_ok = False
            pushes_used = 0
            stop_reason = "max_pushes"
            pos_err = 0.0
            rot_err = 0.0
            prev_pos_err = _init_pos_err
            prev_rot_err = _init_rot_err

            for push_i in range(args.max_pushes):
                with torch.no_grad():
                    h_in = (bob_hidden[0], bob_hidden[1])
                    (b_acts, _, _, _, _, new_bh) = bob_ppo.actor_critic.act_with_hidden(
                        bob_obs, None, h_in,
                    )
                    if new_bh is not None:
                        bob_hidden[0] = new_bh[0]
                        bob_hidden[1] = new_bh[1]

                obj_x = full_obs[0, _OBS_ROBOT_DIM].item()
                obj_y = full_obs[0, _OBS_ROBOT_DIM + 1].item()
                obj_yaw = float(full_obs[0, _OBS_ROBOT_DIM + 5].item())
                Xs, Ys, length, theta = decode_push_action_relative(
                    b_acts,
                    torch.tensor([[obj_x, obj_y]], device=device),
                    torch.tensor([obj_yaw], device=device),
                    num_bins=num_bins,
                    min_r=_min_r,
                    max_r=_max_r,
                )
                Xf = Xs + length * torch.cos(theta)
                Yf = Ys + length * torch.sin(theta)

                waypoints = compute_push_waypoints(
                    Xs=Xs, Ys=Ys, length=length, theta=theta,
                    current_ee_pos=ee_pos_local,
                    current_ee_quat=ee_quat_w, device=device,
                )

                _update_push_markers(Xs, Ys, Xf, Yf, theta)

                terminated = torch.zeros(1, dtype=torch.bool, device=device)
                for wp_idx, (wp_pos, wp_quat, _wp_grip) in enumerate(waypoints):
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
                    solved = result.solution.view(1, 6)
                    elbow_bad = solved[:, 2] < 0.0
                    if elbow_bad.any():
                        ik_ok[elbow_bad] = False
                    raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, prev_joint_cmd)
                    if terminated.any():
                        raw_cmd[terminated] = cur_joints[terminated]
                    prev_joint_cmd = raw_cmd.detach().clone()

                    env_full = torch.zeros(1, env.action_space.shape[0], device=device)
                    env_full[:, :6] = raw_cmd
                    env_full[:, 6] = -1.0
                    obs_ret, _, step_terminated, _, _ = env.step(env_full)
                    terminated |= step_terminated

                    _tcp_z_check = _tcp_pos_local()[:, 2]
                    terminated |= (_tcp_z_check < -0.01)

                pushes_used += 1

                ee_pos_local = _tcp_pos_local()
                ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
                prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
                env._update_goal_in_extras()
                full_obs = _build_obs(_obj_type)
                bob_obs = full_obs.clone()

                cur_obj_pos = full_obs[0, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
                cur_obj_euler = full_obs[0, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
                goal_pos = full_obs[0, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 14 + 3]
                goal_euler = full_obs[0, _OBS_ROBOT_DIM + 14 + 3:_OBS_ROBOT_DIM + 14 + 6]

                pos_err = (cur_obj_pos - goal_pos).norm().item()
                rot_err = _rot_distance_rad(
                    cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)
                ).item()
                if _obj_type == "disc":
                    rot_err = 0.0

                obj_z = float(full_obs[0, _OBS_ROBOT_DIM + 2])
                tipped = (abs(float(cur_obj_euler[0])) > 0.3 or abs(float(cur_obj_euler[1])) > 0.3)
                oob_2d = float((cur_obj_pos[:2] - goal_pos[:2]).norm())

                r_dec = float(torch.sqrt((Xs - obj_x)**2 + (Ys - obj_y)**2).item())
                _cov = _area_coverage(pos_err, rot_err)
                print(f"  push {push_i:2d}: bins=({', '.join(f'{int(b_acts[0,i].item()):2d}' for i in range(4))})  "
                      f"r={r_dec:.3f} len={float(length.item()):.3f} θ={math.degrees(float(theta.item())):.0f}°  "
                      f"pos={pos_err:.4f}m rot={rot_err:.3f}rad z={obj_z:.3f} cov={_cov:.1f}%")

                if terminated[0]:
                    stop_reason = "physics"
                    break

                if _obj_type == "disc" or cfg.test_type == "disc_pos":
                    _success_check = pos_err < 0.05
                else:
                    _success_check = pos_err < 0.05 and rot_err < args.rot_threshold
                if _success_check:
                    trial_ok = True

                if obj_z > 0.10:
                    stop_reason = "launched"
                    pos_err = prev_pos_err
                    rot_err = prev_rot_err
                    break
                if tipped:
                    stop_reason = "tipped"
                    pos_err = prev_pos_err
                    rot_err = prev_rot_err
                    break
                if oob_2d > _init_oob_2d + 0.20:
                    stop_reason = "oob"
                    break

                if pos_err < best_pos_err:
                    best_pos_err = pos_err
                    best_rot_err = rot_err
                prev_pos_err = pos_err
                prev_rot_err = rot_err
                env.capture_pre_push(full_obs)

            if trial_ok:
                trial_successes += 1
            trial_pushes.append(pushes_used)
            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_rot_err = rot_err

        avg_pushes = int(np.mean(trial_pushes)) if trial_pushes else 0
        sr_pct = trial_successes / TRIAL_COUNT * 100

        result = ASPValidationResult(
            test_index=test_idx,
            test_name=f"{cfg.name} #{cfg.test_id}",
            test_type=cfg.test_type,
            object_type=_obj_type,
            success=(trial_successes > 0),
            pushes_used=avg_pushes,
            final_pos_error=best_pos_err,
            final_rot_error=best_rot_err,
            area_coverage=_area_coverage(best_pos_err, best_rot_err),
            trial_count=TRIAL_COUNT,
            success_count=trial_successes,
        )
        results.append(result)
        test_cfgs_data.append({
            "start_x": cfg.main_start.x, "start_y": cfg.main_start.y,
            "goal_x": cfg.main_goal_x, "goal_y": cfg.main_goal_y,
            "goal_yaw": cfg.main_goal_yaw,
            "object_type": cfg.object_type,
        })
        status = "PASS" if trial_successes > 0 else "FAIL"
        print(f"  {status} | {trial_successes}/{TRIAL_COUNT} = {sr_pct:.0f}% | avg_pushes: {avg_pushes} | "
              f"pos_err: {best_pos_err:.4f} | rot_err: {best_rot_err:.4f} | cov: {_area_coverage(best_pos_err, best_rot_err):.1f}%")

    total_trials = sum(r.trial_count for r in results)
    total_successes = sum(r.success_count for r in results)
    sr = total_successes / total_trials * 100 if total_trials > 0 else 0
    n_tests_passed = sum(1 for r in results if r.success_count > 0)
    avg_pushes = np.mean([r.pushes_used for r in results]) if results else 0

    disc_tests = [r for r in results if r.test_type == "disc_pos"]
    pos_only = [r for r in results if r.test_type == "pos_only"]
    pos_rot  = [r for r in results if r.test_type == "pos_rot"]
    disc_trials = sum(r.trial_count for r in disc_tests)
    disc_successes = sum(r.success_count for r in disc_tests)
    po_trials = sum(r.trial_count for r in pos_only)
    po_successes = sum(r.success_count for r in pos_only)
    pr_trials = sum(r.trial_count for r in pos_rot)
    pr_successes = sum(r.success_count for r in pos_rot)
    sr_disc = disc_successes / disc_trials * 100 if disc_trials > 0 else 0
    sr_po = po_successes / po_trials * 100 if po_trials > 0 else 0
    sr_pr = pr_successes / pr_trials * 100 if pr_trials > 0 else 0

    avg_cov = np.mean([r.area_coverage for r in results]) if results else 0

    print(f"\n{'='*60}")
    print(f"ASP BOB VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total tests:   {len(results)}")
    print(f"  Tests passed:  {n_tests_passed}")
    print(f"  Success rate:  {sr:.1f}%")
    print(f"  Disc SR:       {sr_disc:.1f}% ({len(disc_tests)} tests)")
    print(f"  Pos-only SR:   {sr_po:.1f}% ({len(pos_only)} tests)")
    print(f"  Pos+rot SR:    {sr_pr:.1f}% ({len(pos_rot)} tests)")
    print(f"  Avg pushes:    {avg_pushes:.1f}")
    print(f"  Avg coverage:  {avg_cov:.1f}%")
    print(f"{'='*60}")

    for r in results:
        status = "PASS" if r.success_count > 0 else "FAIL"
        print(f"  {status:5s} | Test {r.test_index:2d} | {r.test_name:30s} | pushes={r.pushes_used:2d}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as _f:
            writer = _csv.writer(_f)
            writer.writerow(["test_index", "test_name", "test_type", "object_type", "success",
                             "pushes_used", "pos_err", "rot_err", "area_coverage",
                             "trial_count", "success_count"])
            for r in results:
                writer.writerow([r.test_index, r.test_name, r.test_type, r.object_type,
                                 int(r.success), r.pushes_used, r.final_pos_error,
                                 r.final_rot_error, r.area_coverage,
                                 r.trial_count, r.success_count])
        print(f"\n[CSV] Results saved to {args.csv}")

    if args.csv and results:
        try:
            from asyncDualPlayPPO.tests.plot_validation import generate_single_run_plot
            plot_data = [{"test_index": r.test_index, "test_name": r.test_name,
                          "success": r.success, "final_pos_error": r.final_pos_error,
                          "final_rot_error": r.final_rot_error} for r in results]
            plot_path = os.path.splitext(args.csv)[0] + ".png"
            generate_single_run_plot(plot_data, test_cfgs_data, plot_path,
                                    rot_threshold_rad=args.rot_threshold)
        except Exception as _e:
            print(f"[WARN] Plot generation failed: {_e}")

    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
