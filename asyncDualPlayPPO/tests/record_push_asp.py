"""
Record top-down videos of a trained Push-ASP (Model C/E/F/G/H) Bob solving validation
scenes.  This is a recording-oriented fork of ``validate_push_asp.py`` — same ASP Bob
loader (PPOABC + GoalEncoder), same T-block/disc scene, same cuRobo IK + push primitive
+ visual markers, but instead of measuring success rate it captures camera frames and
writes MP4 (+ 480p GIF + keyframe PNG) per scene.

Supports both the T-block and the disc object (``--scene-set disc``), and d_pose
observation mode (``--dpose-obs``).  Bob is recorded pushing to a *fixed* goal from the
validation scene set — Alice is not required (goal is predefined, not adversarial).

Usage (DISC-F = ASP-disc, seed s42):
  python -m asyncDualPlayPPO.tests.record_push_asp \
      --chkpt_bob final_results_thesis/discF_e528_i3000_s42/bob/model_best.pt \
      --scene-set disc --dpose-obs --char-length 0.0 --dpose-threshold 0.05 \
      --scenes 1,2 --headless --enable_cameras --max_attempts 1 --max_pushes 10 \
      --out-dir /tmp/discF_demo

Notes:
  * ``--enable_cameras`` is REQUIRED (camera sensor needs rendering, works headless).
  * ``--scene-set disc`` selects the 30 disc scenes (D_*); ``--scenes`` indexes into that set.
  * MP4 videos land in ``<out-dir>/videos/``; keyframe PNGs in ``<out-dir>/``; GIF (480p) next to the MP4.
"""

import argparse
import os
import sys
import math
import signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
    get_test_config, get_test_count, set_test_set,
)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25, 0.70)
_WS_Z = (0.25, 0.55)

# Table centre (world frame) — Fix P78: table centre moved to (0, 0.40).
_TABLE_CENTER_XY = (0.0, 0.40)


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


def main():
    parser = argparse.ArgumentParser(description="Record top-down push videos (ASP Bob)")
    parser.add_argument("--chkpt_bob", type=str, required=True, help="Path to Bob checkpoint (or run dir with bob/)")
    parser.add_argument("--scenes", type=str, default="1,2",
                        help="Comma-separated scene indices to record (into --scene-set)")
    parser.add_argument("--scene-set", choices=["all", "tblock", "disc"], default="disc",
                        help="Which scene list to use (disc = 30 disc scenes)")
    parser.add_argument("--max_pushes", type=int, default=10, help="Max pushes per scene")
    parser.add_argument("--max_attempts", type=int, default=1,
                        help="Re-run a scene up to N times; keep the first successful rollout")
    parser.add_argument("--rot_threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians")
    parser.add_argument("--dpose-obs", action="store_true", dest="dpose_obs",
                        help="Use d_pose observations (for dpose-trained models)")
    parser.add_argument("--char-length", type=float, default=0.07,
                        help="SE(2) characteristic length L for d_pose (disc: 0.0)")
    parser.add_argument("--dpose-threshold", type=float, default=0.055,
                        help="d_pose success threshold in metres (disc: 0.05)")
    parser.add_argument("--rel-obs", "--rel_obs", action="store_true", dest="rel_obs", default=None,
                        help="Use object-relative goal observations (default: auto)")
    parser.add_argument("--argmax", action="store_true", dest="argmax",
                        help="Use argmax (deterministic) actions instead of sampling")
    parser.add_argument("--fps", type=int, default=15, help="Output video frame rate")
    parser.add_argument("--capture-every", type=int, default=3, dest="capture_every",
                        help="Capture one frame every N waypoint substeps")
    parser.add_argument("--width", type=int, default=1920, help="Camera image width")
    parser.add_argument("--height", type=int, default=1080, help="Camera image height")
    parser.add_argument("--cam-margin", type=float, default=0.055, dest="cam_margin",
                        help="Extra height above the robot's highest link, in metres")
    parser.add_argument("--cam-height", type=float, default=3.5, dest="cam_height",
                        help="Manual camera Z height in metres (overrides auto-detection)")
    parser.add_argument("--gif", dest="gif", action="store_true", default=True,
                        help="Also save a 480p GIF version of each video (default: on)")
    parser.add_argument("--no-gif", dest="gif", action="store_false", help="Disable GIF output")
    parser.add_argument("--clean", action="store_true",
                        help="Hide debug markers (push spheres/arrow). Goal ghost always shown.")
    parser.add_argument("--out-dir", type=str, dest="out_dir", default=None,
                        help="Directory for output MP4/PNG (default: presentation/figures)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if not getattr(args, "enable_cameras", False):
        print("[ERROR] --enable_cameras is required (camera sensor needs rendering). "
              "Re-run with --enable_cameras.")
        sys.exit(1)

    set_test_set(args.scene_set)

    # Resolve checkpoint dir -> bob/model_best.pt
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

    if args.out_dir is None:
        args.out_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..",
            "literature", "paper-async", "presentation", "figures"))
    os.makedirs(args.out_dir, exist_ok=True)
    video_dir = os.path.join(args.out_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    scene_indices = [int(s) for s in args.scenes.split(",") if s.strip() != ""]

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    signal.signal(signal.SIGINT, lambda *_: (simulation_app.close(), os._exit(1)))

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import SceneEntityCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push_asp import PushASPEnvWrapper, _OBS_ROBOT_DIM
    from asyncDualPlayPPO.tasks.utils.reward_pbrs import dpose_and_zero_yaw
    from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative,
        TBLOCK_MIN_R, TBLOCK_MAX_R,
        DISC_MIN_R, DISC_MAX_R,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
    from asyncDualPlayPPO.algorithms.rl.ppo.module import MultiCategorical
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
    import imageio.v2 as imageio

    @configclass
    class RecordEnvCfg(PushTaskCuRoboEnvCfg):
        @configclass
        class RecordSceneCfg(PushTaskCuRoboEnvCfg.PushTaskSceneCfg):
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
        scene: RecordSceneCfg = RecordSceneCfg(num_envs=1, env_spacing=2.5)

    ppo_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "cfg/ppo/ppo_continuous.yaml")
    with open(ppo_cfg_path, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    num_cat_dims = 4
    num_bins = 21

    # Detect GoalEncoder from checkpoint
    _chkpt = torch.load(args.chkpt_bob, map_location="cpu", weights_only=False)
    _chkpt_state = _chkpt.get("model_state_dict", _chkpt)
    _pi_w0 = _chkpt_state.get("pi_encoder.obj_encoder.0.weight")
    _is_noge = _pi_w0 is not None and _pi_w0.shape[1] == 22
    _has_goal_encoder = not _is_noge
    print(f"[Detect] GoalEncoder={'ON' if _has_goal_encoder else 'OFF (Model D)'}")
    del _chkpt, _chkpt_state
    torch.cuda.empty_cache()

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

    env_cfg = RecordEnvCfg()
    env_cfg.scene.num_envs = 1

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES,
        scale=1.0, use_default_offset=False,
    )

    # ── Top-down camera (added before build) ───────────────────────────────────
    env_cfg.scene.top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCamera",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(_TABLE_CENTER_XY[0], _TABLE_CENTER_XY[1], 1.2),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushASPEnvWrapper(
        env=base_env, alice_pushes=5, bob_pushes=args.max_pushes,
        max_goals_per_episode=1, num_objects=1,
        rel_obs=(args.rel_obs if args.rel_obs is not None else (not args.dpose_obs)),
        dpose_obs=args.dpose_obs,
        char_length=args.char_length,
        dpose_threshold=args.dpose_threshold,
        device=device,
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
                    radius=0.05, height=0.001,
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
        if args.clean:
            return
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

    # ── Camera height = robot's highest link (or manual) ───────────────────────
    _robot_max_z = float(_robot_scene.data.body_pos_w[0, :, 2].max().item())
    _cam_height = args.cam_height if args.cam_height is not None else (_robot_max_z + args.cam_margin)
    _origin = env.env.scene.env_origins[0]
    _cam_pos = torch.tensor([[float(_origin[0]) + _TABLE_CENTER_XY[0],
                              float(_origin[1]) + _TABLE_CENTER_XY[1],
                              float(_origin[2]) + _cam_height]], device=device)
    _cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    top_cam = env.env.scene["top_camera"]
    top_cam.set_world_poses(positions=_cam_pos, orientations=_cam_quat, convention="opengl")
    print(f"[Setup] camera at z = {_cam_height:.3f} m over table centre {_TABLE_CENTER_XY}")

    def _grab_frame():
        out = top_cam.data.output["rgb"]
        if out is None:
            return None
        img = out[0].detach().cpu().numpy()
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            if float(np.nanmax(img)) <= 1.0 + 1e-3:
                img = img * 255.0
            img = np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    # warm up renderer
    for _ in range(5):
        env.step(_calib_act)

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
        obs = torch.cat([ee, obj, goal, dist], dim=-1)
        if args.dpose_obs:
            obs = dpose_and_zero_yaw(obs, _OBS_ROBOT_DIM, 14, 6, args.char_length)
        return obs

    # ── Load Bob ──────────────────────────────────────────────────────────────
    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(num_cat_dims,), dtype=np.float32,
    )

    bob_ppo = PPOABC(
        vec_env=env, cfg_train=bob_cfg, device=device,
        sampler="sequential", log_dir="/tmp/record_push_asp",
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
    print(f"[Record] Loaded Bob from {args.chkpt_bob}")
    print(f"[Record] scene_set={args.scene_set}, dpose_obs={args.dpose_obs}, "
          f"char_length={args.char_length}, scenes={scene_indices}")

    # ── Init ──────────────────────────────────────────────────────────────────
    obs = env.reset()
    env.env.sim.step()
    _close_act = torch.zeros(1, env.action_space.shape[0], device=device)
    _close_act[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
    _close_act[:, 6] = -1.0
    env.step(_close_act)

    # ── Rollout one scene, returning (frames, success) ─────────────────────────
    def _rollout_scene(cfg):
        frames = []
        substep = 0
        _obj_type = cfg.object_type
        _active_obj_name = "disc_object" if _obj_type == "disc" else "target_object"
        _inactive_obj_name = "target_object" if _obj_type == "disc" else "disc_object"
        _spawn_z = 0.03 if _obj_type == "disc" else 0.05
        _min_r = DISC_MIN_R if _obj_type == "disc" else TBLOCK_MIN_R
        _max_r = DISC_MAX_R if _obj_type == "disc" else TBLOCK_MAX_R

        top_cam.set_world_poses(positions=_cam_pos, orientations=_cam_quat, convention="opengl")

        _reset_jpos = _robot_scene.data.joint_pos.clone()
        _reset_jpos[:, _arm_jids] = _calib_cmd
        _robot_scene.write_joint_state_to_sim(_reset_jpos, torch.zeros_like(_reset_jpos))

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
        if _obj_type == "tblock":
            full_obs = env._get_push_obs()
        else:
            full_obs = _build_obs(_obj_type)
        bob_obs = full_obs.clone()

        f = _grab_frame()
        if f is not None:
            frames.append(f)

        ee_pos_local = _tcp_pos_local()
        ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
        prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
        bob_hidden[0].zero_()
        bob_hidden[1].zero_()

        success = False
        pos_err = 0.0
        for push_i in range(args.max_pushes):
            with torch.no_grad():
                h_in = (bob_hidden[0], bob_hidden[1])
                raw_logits, new_bh = bob_ppo.actor_critic._actor_forward(bob_obs, h_in)
                _dist = MultiCategorical(raw_logits.view(1, num_cat_dims, num_bins))
                b_acts = _dist.mode() if args.argmax else _dist.sample()
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
                min_r=_min_r, max_r=_max_r,
            )
            Xf = Xs + length * torch.cos(theta)
            Yf = Ys + length * torch.sin(theta)

            waypoints = compute_push_waypoints(
                Xs=Xs, Ys=Ys, length=length, theta=theta,
                current_ee_pos=ee_pos_local, current_ee_quat=ee_quat_w, device=device,
            )
            _update_push_markers(Xs, Ys, Xf, Yf, theta)

            terminated = torch.zeros(1, dtype=torch.bool, device=device)
            for (wp_pos, wp_quat, _wp_grip) in waypoints:
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
                _, _, step_terminated, _, _ = env.step(env_full)
                terminated |= step_terminated
                terminated |= (_tcp_pos_local()[:, 2] < -0.01)

                substep += 1
                if substep % args.capture_every == 0:
                    f = _grab_frame()
                    if f is not None:
                        frames.append(f)

            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
            env._update_goal_in_extras()
            if _obj_type == "tblock":
                full_obs = env._get_push_obs()
            else:
                full_obs = _build_obs(_obj_type)
            bob_obs = full_obs.clone()

            cur_obj_pos = full_obs[0, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]
            cur_obj_euler = full_obs[0, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]
            goal_pos = full_obs[0, _OBS_ROBOT_DIM + 14:_OBS_ROBOT_DIM + 14 + 3]
            goal_euler = full_obs[0, _OBS_ROBOT_DIM + 14 + 3:_OBS_ROBOT_DIM + 14 + 6]
            pos_err = (cur_obj_pos - goal_pos).norm().item()
            rot_err = _rot_distance_rad(cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)).item()
            if _obj_type == "disc":
                rot_err = 0.0

            obj_z = float(full_obs[0, _OBS_ROBOT_DIM + 2])
            print(f"  push {push_i:2d}: pos_err={pos_err:.4f}m rot_err={rot_err:.3f}rad z={obj_z:.3f}")

            f = _grab_frame()
            if f is not None:
                frames.append(f)

            if _obj_type == "disc":
                success = pos_err < 0.05
            else:
                success = pos_err < 0.05 and rot_err < args.rot_threshold

            if success or terminated[0]:
                break
            env.capture_pre_push(full_obs)

        # hold a few frames at the end
        hold_cmd = torch.zeros(1, env.action_space.shape[0], device=device)
        hold_cmd[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
        hold_cmd[:, 6] = -1.0
        for _ in range(max(1, args.fps // 2)):
            env.step(hold_cmd)
            f = _grab_frame()
            if f is not None:
                frames.append(f)

        return frames, success

    # ── GIF writer (480p) ─────────────────────────────────────────────────────
    def _save_gif(frames, gif_path):
        if not args.gif:
            return
        try:
            import cv2
            from PIL import Image
            imgs = []
            for f in frames:
                h, w = f.shape[:2]
                if h != 480:
                    scale = 480.0 / h
                    f = cv2.resize(f, (max(1, int(w * scale)), 480), interpolation=cv2.INTER_AREA)
                imgs.append(Image.fromarray(np.ascontiguousarray(f[..., :3]).copy()))
            duration = int(round(1000.0 / args.fps))
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], loop=0, duration=duration)
            print(f"  saved {gif_path} ({os.path.getsize(gif_path) / 1e6:.1f} MB, 480p)")
        except Exception as e:
            print(f"  [error] GIF encoding failed for {gif_path}: {e}")

    # ── Record each scene ─────────────────────────────────────────────────────
    n_tests = get_test_count()
    for idx in scene_indices:
        if idx < 1 or idx > n_tests:
            print(f"[skip] scene {idx} out of range (1..{n_tests})")
            continue
        cfg = get_test_config(idx)
        if cfg is None:
            continue
        print(f"\n[Scene {idx}] {cfg.name}  type={cfg.test_type}  "
              f"obj={cfg.object_type}  goal=({cfg.main_goal_x:+.2f},{cfg.main_goal_y:+.2f})")

        best_frames, got_success = None, False
        for attempt in range(args.max_attempts):
            frames, success = _rollout_scene(cfg)
            print(f"  attempt {attempt + 1}/{args.max_attempts}: "
                  f"{'SUCCESS' if success else 'fail'} ({len(frames)} frames)")
            if best_frames is None:
                best_frames = frames
            if success:
                best_frames, got_success = frames, True
                break

        if not best_frames:
            print(f"  [warn] no frames captured for scene {idx}")
            continue

        stem = f"rec_push_s{idx:02d}"
        mp4_path = os.path.join(video_dir, f"{stem}.mp4")
        gif_path = os.path.join(video_dir, f"{stem}.gif")
        key_path = os.path.join(args.out_dir, f"{stem}_key.png")
        try:
            imageio.mimsave(mp4_path, best_frames, fps=args.fps, macro_block_size=None)
            imageio.imwrite(key_path, best_frames[len(best_frames) // 2])
            print(f"  saved {mp4_path} ({'success' if got_success else 'best-effort'}) "
                  f"+ keyframe {key_path}")
            _save_gif(best_frames, gif_path)
        except Exception as e:
            print(f"  [error] encoding failed for scene {idx}: {e}")

    print("\n[Record] Done.")
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
