"""
Validate a trained Diffusion-Policy push model against test configurations.

Same stop rules as ``validate_push_asp.py`` (the 30 predefined scenes from
``validation_configs.py``, retries, physics/tipped/launched/oob termination,
CSV + plots), but the actor is a Diffusion Policy that outputs the **same 4D
push primitive** as Push-PPO.

Success logging: position-only OR orientation-only achievement is logged as a
success, but does NOT stop the episode early — the policy keeps pushing through
its full budget to try to satisfy both.  If the object leaves the workspace, the
environment is reset and the attempt is retried.

The diffusion policy is action-agnostic: it is trained to regress the continuous
push primitive ``[Xs, Ys, length, theta]`` (theta encoded as ``sin/cos`` ->
action_dim 5, or raw -> action_dim 4) conditioned on the 28D push observation.
At eval the sampled primitive is fed straight into ``compute_push_waypoints``,
reusing the entire cuRobo IK / waypoint / scoring machinery from validate_push.

Usage:
  python -m asyncDualPlayPPO.tests.validate_push_diffusion \
      --chkpt data/outputs/.../checkpoints/latest.ckpt \
      --num_tests 30 --headless --csv dp_results.csv
"""

import argparse
import os
import sys
import math
from collections import deque
from dataclasses import dataclass
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# cuRobo must be imported before AppLauncher
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
    get_test_config, get_test_count,
)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Primitive ranges (must match action_push.decode_push_action defaults that the
# demonstrations were generated with) — used only to clamp out-of-range samples.
_MAX_XS = 0.50
_MIN_YS, _MAX_YS = 0.25, 0.70
_MAX_LEN = 0.20


@dataclass
class ValidationResult:
    test_index: int
    test_name: str
    test_type: str
    success: bool
    pushes_used: int
    final_pos_error: float
    final_rot_error: float
    area_coverage: float


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def _area_coverage(pos_err, rot_err):
    pc = max(0.0, 1.0 - pos_err / 0.10)
    rc = max(0.0, 1.0 - rot_err / 0.40)
    return pc * rc * 100.0


def _load_diffusion_policy(chkpt_path, device, dp_repo):
    """Load a diffusion_policy checkpoint and return its (eval-mode) policy.

    Instantiates the policy directly from ``cfg.policy`` and loads the (EMA)
    weights from the checkpoint, avoiding the training Workspace import (which
    pulls in training-only deps that may be absent in the IsaacLab venv).
    """
    if dp_repo and dp_repo not in sys.path:
        sys.path.insert(0, dp_repo)
    try:
        import dill
        import hydra
    except ModuleNotFoundError as e:
        print(f"[ERROR] diffusion_policy deps missing ({e}). "
              f"pip install dill hydra-core diffusers, or pass --dp_repo.")
        sys.exit(1)

    payload = torch.load(open(chkpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    try:
        use_ema = bool(cfg.training.use_ema)
    except Exception:  # noqa: BLE001
        use_ema = True

    policy = hydra.utils.instantiate(cfg.policy)
    sds = payload.get("state_dicts", {})
    key = "ema_model" if (use_ema and "ema_model" in sds) else "model"
    if key not in sds:
        raise KeyError(f"checkpoint has no '{key}' state_dict (have: {list(sds.keys())})")
    policy.load_state_dict(sds[key])

    policy.to(device)
    policy.eval()
    return policy, cfg


def _decode_dp_action(act, theta_encoding, clamp):
    """Decode a (1, Da) primitive sample -> (Xs, Ys, length, theta)."""
    Xs = act[:, 0]
    Ys = act[:, 1]
    length = act[:, 2]
    if theta_encoding == "sincos":
        theta = torch.atan2(act[:, 3], act[:, 4])
    else:
        theta = act[:, 3]
    if clamp:
        Xs = Xs.clamp(-_MAX_XS, _MAX_XS)
        Ys = Ys.clamp(_MIN_YS, _MAX_YS)
    length = length.clamp(min=0.0, max=_MAX_LEN)
    return Xs, Ys, length, theta


def main():
    parser = argparse.ArgumentParser(description="Validate Diffusion-Policy Push Model")
    parser.add_argument("--chkpt", type=str, required=True, help="Path to DP .ckpt")
    parser.add_argument("--num_tests", type=int, default=30, help="Number of test scenes to run")
    parser.add_argument("--max_pushes", type=int, default=30, help="Max pushes per test")
    parser.add_argument("--rot_threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians (default 0.2)")
    parser.add_argument("--theta_encoding", type=str, default="auto",
                        choices=["auto", "sincos", "raw"],
                        help="How theta is encoded in the action (auto -> 5D=sincos, 4D=raw)")
    parser.add_argument("--no_clamp", action="store_true",
                        help="Disable clamping of decoded Xs/Ys to workspace ranges")
    parser.add_argument("--dp_repo", type=str, default=None,
                        help="Path to the diffusion_policy repo root (auto-detected if omitted)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save validation results to CSV file")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.dp_repo is None:
        # sibling of asyncDualPlayPPO: master_isaac/diffusion_policy
        _master = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        args.dp_repo = os.path.join(_master, "diffusion_policy")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    chkpt_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.chkpt)))
    if args.csv is None:
        args.csv = os.path.join(chkpt_run_dir, "validation_results_diffusion.csv")
    elif not os.path.isabs(args.csv):
        args.csv = os.path.join(chkpt_run_dir, args.csv)

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.utils import configclass
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import SceneEntityCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper, _euler_to_quat
    from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints
    from asyncDualPlayPPO.tasks.utils.observations import (
        ee_poses as _obs_ee_poses,
        object_states as _obs_object_states,
        goal_states as _obs_goal_states,
        goal_distance as _obs_goal_distance,
    )
    import numpy as np

    # ── Environment (T-block + disc, identical to validate_push_asp.py) ─────────
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

    env_cfg = ValidateEnvCfg()
    env_cfg.scene.num_envs = 1  # single env for validation

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES,
        scale=1.0, use_default_offset=False,
    )

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushEnvWrapper(
        env=base_env, device=device, num_objects=1,
        max_pushes_per_episode=args.max_pushes,
        rel_obs=False,
    )

    # ── Observation builder (active object: T-block or disc) ────────────────────
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

    # ── Visual markers ─────────────────────────────────────────────────────────
    _goal_viz = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkers",
            markers={
                "tblock": UsdFileCfg(
                    usd_path=os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), "assets/blocks/t_shape.usda"),
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

    def _update_goal_marker(gx, gy, gyaw=0.0, obj_type="tblock"):
        origins = env.env.scene.env_origins
        pos = torch.tensor([[gx, gy, 0.001]], device=device) + origins
        euler = torch.zeros(1, 3, device=device)
        euler[0, 2] = gyaw
        quat = _euler_to_quat(euler)
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
    ik_solver = IKSolver(_ik_config)

    _QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    _robot_scene = env.env.scene["robot"]
    _arm_jids, _ = _robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
    _lf_ids, _ = _robot_scene.find_bodies("left_inner_finger")
    _rf_ids, _ = _robot_scene.find_bodies("right_inner_finger")

    def _tcp_pos_local():
        lf_w = _robot_scene.data.body_pos_w[:, _lf_ids[0]]
        rf_w = _robot_scene.data.body_pos_w[:, _rf_ids[0]]
        return ((lf_w + rf_w) / 2.0 - env.env.scene.env_origins).clone()

    # ── IK→physics calibration ────────────────────────────────────────────────
    _WS_X = (-0.50, 0.50)
    _WS_Y = (0.25, 0.70)
    _WS_Z = (0.25, 0.55)
    # Object workspace bounds — if the object leaves these, reset the env.
    _OBJ_WS_X = (-0.55, 0.55)
    _OBJ_WS_Y = (0.20, 0.78)
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

    # ── Load diffusion policy ──────────────────────────────────────────────────
    policy, dp_cfg = _load_diffusion_policy(args.chkpt, device, args.dp_repo)
    To = int(getattr(policy, "n_obs_steps", 2))
    Da = int(getattr(policy, "action_dim", 5))
    policy_obs_dim = int(getattr(policy, "obs_dim", env.obs_dim))

    if args.theta_encoding == "auto":
        theta_encoding = "sincos" if Da >= 5 else "raw"
    else:
        theta_encoding = args.theta_encoding

    if policy_obs_dim != env.obs_dim:
        print(f"[WARN] Policy obs_dim={policy_obs_dim} != env obs_dim={env.obs_dim}. "
              f"Observation layout mismatch — results will be invalid.")

    print(f"[Validate] Loaded diffusion policy from {args.chkpt}")
    print(f"[Validate] To={To}  action_dim={Da}  theta_encoding={theta_encoding}  "
          f"obs_dim={policy_obs_dim}  rot_threshold={args.rot_threshold:.3f} rad  "
          f"max_pushes={args.max_pushes}")

    def _predict_primitive(obs_window):
        with torch.no_grad():
            out = policy.predict_action({"obs": obs_window})
        act = out["action"][:, 0]  # first action step -> (1, Da)
        return _decode_dp_action(act, theta_encoding, clamp=not args.no_clamp)

    # ── Run tests ─────────────────────────────────────────────────────────────
    results: List[ValidationResult] = []
    test_cfgs_data: List[dict] = []
    n_tests = min(args.num_tests, get_test_count())

    for test_idx in range(1, n_tests + 1):
        cfg = get_test_config(test_idx)
        if cfg is None:
            continue

        _obj_type = getattr(cfg, "object_type", "tblock")
        _active_obj_name = "disc_object" if _obj_type == "disc" else "target_object"
        _inactive_obj_name = "target_object" if _obj_type == "disc" else "disc_object"
        _spawn_z = 0.03 if _obj_type == "disc" else 0.05

        print(f"\n[Test {test_idx}/{n_tests}] {cfg.name} #{cfg.test_id}")

        test_success = False
        pushes_used = 0
        stop_reason = "max_pushes"
        pos_err = 0.0
        rot_err = 0.0
        retry_count = 0
        MAX_RETRIES = 3
        best_pos_err = float('inf')
        best_rot_err = float('inf')

        for retry in range(MAX_RETRIES):
            retry_count = retry + 1

            obs = env.reset()
            env.goal_pos_euler[0, 0] = cfg.main_goal_x
            env.goal_pos_euler[0, 1] = cfg.main_goal_y
            env.goal_pos_euler[0, 2] = 0.0
            env.goal_pos_euler[0, 3:5] = 0.0
            env.goal_pos_euler[0, 5] = cfg.main_goal_yaw
            env._update_goal_in_extras()
            _update_goal_marker(cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw, _obj_type)

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
            env._update_goal_in_extras()
            obs = _build_obs(_obj_type)
            env._capture_prev_obj(obs)

            _init_obj_pos = obs[0, env.robot_dim:env.robot_dim + 3]
            _init_obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
            _init_goal_pos = obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
            _init_goal_euler = obs[0, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]
            _init_pos_err = (_init_obj_pos - _init_goal_pos).norm().item()
            _init_rot_err = _rot_distance_rad(_init_obj_euler.unsqueeze(0), _init_goal_euler.unsqueeze(0)).item()
            if _obj_type == "disc":
                _init_rot_err = 0.0
            _init_oob_2d = float((_init_obj_pos[:2] - _init_goal_pos[:2]).norm().item())
            _rtag = f"[R{retry_count}] " if retry > 0 else ""
            print(f"  {_rtag}[{cfg.test_type}] goal=({cfg.main_goal_x:+.3f},{cfg.main_goal_y:+.3f}) yaw={cfg.main_goal_yaw:+.3f}  "
                  f"start=({cfg.main_start.x:+.3f},{cfg.main_start.y:+.3f})  "
                  f"init_pos_err={_init_pos_err:.4f}m  init_rot_err={_init_rot_err:.3f}rad")

            # Diffusion policies are stateless across calls; reset any internal buffers.
            if hasattr(policy, "reset"):
                policy.reset()

            # Rolling observation window (B=1, To, obs_dim).
            obs_deque = deque([obs.clone() for _ in range(To)], maxlen=To)

            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

            test_success = False
            pushes_used = 0
            stop_reason = "max_pushes"
            pos_err = 0.0
            rot_err = 0.0
            prev_pos_err = _init_pos_err
            prev_rot_err = _init_rot_err

            for push_i in range(args.max_pushes):
                obs_window = torch.stack(list(obs_deque), dim=1)  # (1, To, obs_dim)
                Xs, Ys, length, theta = _predict_primitive(obs_window)

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
                    env_full[:, 6] = -1.0  # always closed
                    obs, _, step_terminated, _, _ = env.step(env_full)
                    terminated |= step_terminated

                    _tcp_z_check = _tcp_pos_local()[:, 2]
                    terminated |= (_tcp_z_check < -0.01)

                env.push_count[0] += 1
                pushes_used = int(env.push_count[0].item())

                ee_pos_local = _tcp_pos_local()
                ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
                prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
                env._update_goal_in_extras()
                obs = _build_obs(_obj_type)
                obs_deque.append(obs.clone())

                cur_obj_pos = obs[0, env.robot_dim:env.robot_dim + 3]
                cur_obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
                goal_pos = obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
                goal_euler = obs[0, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]

                pos_err = (cur_obj_pos - goal_pos).norm().item()
                rot_err = _rot_distance_rad(cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)).item()
                if _obj_type == "disc":
                    rot_err = 0.0

                obj_z = float(obs[0, env.robot_dim + 2])
                tipped = (abs(float(cur_obj_euler[0])) > 0.3 or abs(float(cur_obj_euler[1])) > 0.3)
                oob_2d = float((cur_obj_pos[:2] - goal_pos[:2]).norm())

                _cov = _area_coverage(pos_err, rot_err)
                print(f"  push {push_i:2d}: Xs={float(Xs.item()):+.3f} Ys={float(Ys.item()):+.3f} "
                      f"len={float(length.item()):.3f} θ={math.degrees(float(theta.item())):.0f}°  "
                      f"pos={pos_err:.4f}m rot={rot_err:.3f}rad z={obj_z:.3f} cov={_cov:.1f}%")

                if terminated[0]:
                    stop_reason = "physics"
                    break

                if _obj_type == "disc":
                    _success_check = pos_err < 0.05
                else:
                    # Partial achievement (position-only OR orientation-only) is
                    # logged as success, but we DO NOT stop early — let the policy
                    # use its full push budget to try to satisfy both.
                    _pos_ok = pos_err < 0.05
                    _rot_ok = rot_err < args.rot_threshold
                    _success_check = _pos_ok or _rot_ok
                if _success_check:
                    test_success = True

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
                obj_x_w = float(cur_obj_pos[0])
                obj_y_w = float(cur_obj_pos[1])
                if not (_OBJ_WS_X[0] <= obj_x_w <= _OBJ_WS_X[1]
                        and _OBJ_WS_Y[0] <= obj_y_w <= _OBJ_WS_Y[1]):
                    stop_reason = "out_of_ws"
                    pos_err = prev_pos_err
                    rot_err = prev_rot_err
                    env.reset()
                    break

                if pos_err < best_pos_err:
                    best_pos_err = pos_err
                    best_rot_err = rot_err
                prev_pos_err = pos_err
                prev_rot_err = rot_err
                env.capture_pre_push(obs)

            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_rot_err = rot_err
            if test_success:
                stop_reason = "success"

            if test_success:
                break
            if stop_reason == "max_pushes":
                break
            if retry < MAX_RETRIES - 1:
                pass
            else:
                break

        if retry_count > 1:
            stop_reason += f"_r{retry_count}"

        result = ValidationResult(
            test_index=test_idx,
            test_name=f"{cfg.name} #{cfg.test_id}",
            test_type=cfg.test_type,
            success=test_success,
            pushes_used=pushes_used,
            final_pos_error=best_pos_err,
            final_rot_error=best_rot_err,
            area_coverage=_area_coverage(best_pos_err, best_rot_err),
        )
        results.append(result)
        test_cfgs_data.append({
            "start_x": cfg.main_start.x, "start_y": cfg.main_start.y,
            "goal_x": cfg.main_goal_x, "goal_y": cfg.main_goal_y,
            "goal_yaw": cfg.main_goal_yaw,
            "object_type": getattr(cfg, "object_type", "tblock"),
        })
        status = "PASS" if test_success else "FAIL"
        print(f"  {status} | pushes: {pushes_used} | reason: {stop_reason} | "
              f"pos_err: {best_pos_err:.4f} | rot_err: {best_rot_err:.4f} | cov: {_area_coverage(best_pos_err, best_rot_err):.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_success = sum(1 for r in results if r.success)
    sr = n_success / len(results) * 100 if results else 0
    avg_pushes = np.mean([r.pushes_used for r in results]) if results else 0

    pos_only = [r for r in results if r.test_type == "pos_only"]
    pos_rot  = [r for r in results if r.test_type == "pos_rot"]
    sr_po = sum(1 for r in pos_only if r.success) / len(pos_only) * 100 if pos_only else 0
    sr_pr = sum(1 for r in pos_rot if r.success) / len(pos_rot) * 100 if pos_rot else 0

    avg_cov = np.mean([r.area_coverage for r in results]) if results else 0

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS (Diffusion Policy)")
    print(f"{'='*60}")
    print(f"  Total tests:   {len(results)}")
    print(f"  Successes:     {n_success}")
    print(f"  Success rate:  {sr:.1f}%")
    print(f"  Pos-only SR:   {sr_po:.1f}% ({len(pos_only)} tests)")
    print(f"  Pos+rot SR:    {sr_pr:.1f}% ({len(pos_rot)} tests)")
    print(f"  Avg pushes:    {avg_pushes:.1f}")
    print(f"  Avg coverage:  {avg_cov:.1f}%")
    print(f"{'='*60}")

    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  {status:5s} | Test {r.test_index:2d} | {r.test_name:30s} | pushes={r.pushes_used:2d}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as _f:
            writer = _csv.writer(_f)
            writer.writerow(["test_index", "test_name", "test_type", "success", "pushes_used",
                             "pos_err", "rot_err", "area_coverage"])
            for r in results:
                writer.writerow([r.test_index, r.test_name, r.test_type, int(r.success),
                                 r.pushes_used, r.final_pos_error, r.final_rot_error, r.area_coverage])
        print(f"\n[CSV] Results saved to {args.csv}")

    if args.csv and results:
        try:
            from asyncDualPlayPPO.tests.plot_validation import generate_single_run_plot
            plot_data = [{"test_index": r.test_index, "test_name": r.test_name,
                          "success": r.success, "final_pos_error": r.final_pos_error,
                          "final_rot_error": r.final_rot_error} for r in results]
            plot_path = os.path.splitext(args.csv)[0] + ".png"
            generate_single_run_plot(plot_data, test_cfgs_data, plot_path,
                                     rot_threshold_rad=args.rot_threshold,
                                     policy_label="Diffusion Policy")
        except Exception as _e:
            print(f"[WARN] Plot generation failed: {_e}")

    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
