"""
Record top-down videos of a trained Push-PPO / PBRS model solving validation scenes.

A top-down RGB camera is placed above the centre of the table (world XY = (0, 0.40),
matching Fix P78) at the same height as the robot's highest link, looking straight
down (OpenGL convention).  Each scene is rolled out with the trained policy and the
camera frames are encoded to an MP4 (+ a keyframe PNG) using imageio's bundled ffmpeg.

This is a recording-oriented fork of ``validate_push.py`` — same cuRobo IK + push
primitive + visual markers, but instead of measuring success rate it captures video.

Usage (Model A — the simple PBRS single-agent checkpoint, object-relative obs+act):
  python -m asyncDualPlayPPO.tests.record_push_video \
      --chkpt runs/ppo_pbrs_reward/26.06.20/runs/hpc_pbrs_simp_528env/agent/model_best_simp.pt \
      --rel-obs --rel-act --headless --enable_cameras --scenes 11,14,21

Notes:
  * ``--enable_cameras`` is REQUIRED (camera sensors need rendering, works headless).
  * T-block only — the scene uses the single ``target_object`` (T-block), exactly like
    ``validate_push.py``.  A ``disc_pos`` config (scenes 1-10) would just run the T-block
    in position-only mode (rotation ignored).  The defaults use genuine T-block scenes:
    11 = E_Forward (pos-only), 13 = E_Left (pos-only, lateral), 21 = E_Diag (pos+rot).
  * Outputs land in ``literature/paper-async/presentation/figures`` by default.
"""

import argparse
import os
import sys
import math

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

# Table centre (world frame) — Fix P78: table centre moved to (0, 0.40).
_TABLE_CENTER_XY = (0.0, 0.40)


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


def main():
    parser = argparse.ArgumentParser(description="Record top-down push videos")
    parser.add_argument("--chkpt", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--scenes", type=str, default="11,13,21",
                        help="Comma-separated validation scene indices to record")
    parser.add_argument("--max_pushes", type=int, default=15, help="Max pushes per scene")
    parser.add_argument("--max_attempts", type=int, default=3,
                        help="Re-run a scene up to N times; keep the first successful rollout")
    parser.add_argument("--rot_threshold", type=float, default=0.2,
                        help="Rotation success threshold in radians")
    parser.add_argument("--rel-obs", action="store_true", dest="rel_obs",
                        help="Use object-relative observation (30D instead of 28D)")
    parser.add_argument("--rel-act", action="store_true", dest="rel_act",
                        help="Decode actions as object-relative (r, phi, len, theta)")
    parser.add_argument("--fps", type=int, default=15, help="Output video frame rate")
    parser.add_argument("--capture-every", type=int, default=3, dest="capture_every",
                        help="Capture one frame every N waypoint substeps")
    parser.add_argument("--width", type=int, default=1920, help="Camera image width")
    parser.add_argument("--height", type=int, default=1080, help="Camera image height")
    parser.add_argument("--cam-margin", type=float, default=0.055, dest="cam_margin",
                        help="Extra height above the robot's highest link, in metres")
    parser.add_argument("--clean", action="store_true",
                        help="Hide debug markers (push spheres/arrow). Goal ghost always shown.")
    parser.add_argument("--out-dir", type=str, dest="out_dir", default=None,
                        help="Directory for output MP4/PNG (default: presentation/figures)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if not getattr(args, "enable_cameras", False):
        print("[WARN] --enable_cameras was not passed; camera frames will be empty. "
              "Re-run with --enable_cameras (and --headless for offscreen render).")

    # default output dir → presentation figures
    if args.out_dir is None:
        args.out_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..",
            "literature", "paper-async", "presentation", "figures"))
    os.makedirs(args.out_dir, exist_ok=True)

    scene_indices = [int(s) for s in args.scenes.split(",") if s.strip() != ""]

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sensors import CameraCfg
    from asyncDualPlayPPO.tasks.push_task_curobo import PushTaskCuRoboEnvCfg
    from asyncDualPlayPPO.tasks.utils.wrapper_push import PushEnvWrapper, _euler_to_quat
    from asyncDualPlayPPO.tasks.utils.action_push import (
        decode_push_action, compute_push_waypoints,
    )
    from asyncDualPlayPPO.tasks.utils.action_push_relative import (
        decode_push_action_relative,
    )
    from asyncDualPlayPPO.algorithms.rl.ppo.ppo import PPO
    from asyncDualPlayPPO.algorithms.rl.ppo.module_push import ActorCriticPush
    import gymnasium as gym_mc
    import numpy as np
    import imageio.v2 as imageio

    # ── Environment (T-block only, exactly like validate_push.py) ────────────────
    env_cfg = PushTaskCuRoboEnvCfg()
    env_cfg.scene.num_envs = 1

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_ARM_JOINT_NAMES,
        scale=1.0, use_default_offset=False,
    )

    # ── Top-down camera (added to scene before build) ───────────────────────────
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
            convention="opengl",  # forward -Z, up +Y → straight down
        ),
    )

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    env = PushEnvWrapper(
        env=base_env, device=device, num_objects=1,
        max_pushes_per_episode=args.max_pushes,
        rel_obs=args.rel_obs,
    )

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

    def _update_goal_marker(gx, gy, gyaw=0.0):
        origins = env.env.scene.env_origins
        pos = torch.tensor([[gx, gy, 0.001]], device=device) + origins
        euler = torch.zeros(1, 3, device=device)
        euler[0, 2] = gyaw
        quat = _euler_to_quat(euler)
        _goal_viz.visualize(translations=pos, orientations=quat)

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

    # ── Camera height = robot's highest link ────────────────────────────────────
    _robot_max_z = float(_robot_scene.data.body_pos_w[0, :, 2].max().item())
    _cam_height = _robot_max_z + args.cam_margin
    _origin = env.env.scene.env_origins[0]
    _cam_pos = torch.tensor([[float(_origin[0]) + _TABLE_CENTER_XY[0],
                              float(_origin[1]) + _TABLE_CENTER_XY[1],
                              float(_origin[2]) + _cam_height]], device=device)
    _cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # OpenGL identity → looks down
    top_cam = env.env.scene["top_camera"]
    top_cam.set_world_poses(positions=_cam_pos, orientations=_cam_quat, convention="opengl")
    print(f"[Setup] Robot max height = {_robot_max_z:.3f} m → camera at z = {_cam_height:.3f} m "
          f"over table centre {_TABLE_CENTER_XY}")

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

    # warm up renderer so the first captured frame is valid
    for _ in range(5):
        env.step(_calib_act)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    num_bins = 21
    _mc_space = gym_mc.spaces.Box(
        low=0.0, high=float(num_bins - 1), shape=(4,), dtype=np.float32,
    )
    agent_cfg = {
        "learn": {
            "nsteps": 32, "noptepochs": 3, "nminibatches": 4,
            "cliprange": 0.2, "ent_coef": 0.01, "gamma": 0.998, "lam": 0.95,
            "optim_stepsize": 3e-4, "init_noise_std": 0.3,
            "value_loss_coef": 1.0, "max_grad_norm": 1.0,
        },
        "policy": {
            "use_multicategorical": True, "num_cat_dims": 4, "num_bins": 21,
            "use_lstm": True, "lstm_hidden_size": 256,
            "pi_hid_sizes": [512, 256, 128],
            "vf_hid_sizes": [512, 256, 128],
            "activation": "relu",
        },
    }
    agent = PPO(
        vec_env=env, cfg_train=agent_cfg, device=device,
        sampler="sequential", log_dir="/tmp/record_push",
        asymmetric=False,
    )
    agent.observation_space = env.observation_space
    agent.state_space = env.state_space
    agent.action_space = _mc_space
    agent.desired_kl = None
    agent.actor_critic = ActorCriticPush(
        agent.observation_space.shape, agent.state_space.shape,
        agent.action_space.shape, agent.init_noise_std, agent.model_cfg,
        asymmetric=False,
    ).to(device)
    agent.load(args.chkpt)
    agent.actor_critic.eval()
    print(f"[Record] Loaded model from {args.chkpt}")
    print(f"[Record] rel_obs={args.rel_obs}, rel_act={args.rel_act}, scenes={scene_indices}")

    # ── Rollout one scene, returning (frames, success) ──────────────────────────
    def _rollout_scene(cfg):
        _obj_type = getattr(cfg, "object_type", "tblock")
        frames = []
        substep = 0

        obs = env.reset()
        # keep the overhead camera locked at robot height over the table centre
        top_cam.set_world_poses(positions=_cam_pos, orientations=_cam_quat, convention="opengl")
        env.goal_pos_euler[0, 0] = cfg.main_goal_x
        env.goal_pos_euler[0, 1] = cfg.main_goal_y
        env.goal_pos_euler[0, 2] = 0.0
        env.goal_pos_euler[0, 3:5] = 0.0
        env.goal_pos_euler[0, 5] = cfg.main_goal_yaw
        env._update_goal_in_extras()
        _update_goal_marker(cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw)

        obj = env.env.scene["target_object"]
        obj.write_root_pose_to_sim(torch.tensor([[
            cfg.main_start.x, cfg.main_start.y, 0.02, 1.0, 0.0, 0.0, 0.0
        ]], device=device))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        env.env.sim.step()
        obs = env._get_push_obs()
        env._capture_prev_obj(obs)

        f = _grab_frame()
        if f is not None:
            frames.append(f)

        hidden = [
            torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
            torch.zeros(1, agent.actor_critic.lstm_hidden_size, device=device),
        ]
        ee_pos_local = _tcp_pos_local()
        ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
        prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()

        success = False
        for push_i in range(args.max_pushes):
            with torch.no_grad():
                actions, _, _, _, _, _, new_h = agent.actor_critic.act_with_hidden(
                    obs, None, (hidden[0], hidden[1]),
                )
                if new_h is not None:
                    hidden[0] = new_h[0]
                    hidden[1] = new_h[1]

            if args.rel_act:
                obj_x = obs[0, env.robot_dim]
                obj_y = obs[0, env.robot_dim + 1]
                obj_yaw = obs[0, env.robot_dim + 5]
                Xs, Ys, length, theta = decode_push_action_relative(
                    actions, torch.stack([obj_x, obj_y]).unsqueeze(0),
                    obj_yaw.unsqueeze(0), num_bins=num_bins,
                )
            else:
                Xs, Ys, length, theta = decode_push_action(actions, num_bins=num_bins)

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
                obs, _, step_terminated, _, _ = env.step(env_full)
                terminated |= step_terminated
                terminated |= (_tcp_pos_local()[:, 2] < -0.01)

                substep += 1
                if substep % args.capture_every == 0:
                    f = _grab_frame()
                    if f is not None:
                        frames.append(f)

            env.push_count[0] += 1
            ee_pos_local = _tcp_pos_local()
            ee_quat_w = _QUAT_TOOL_DOWN.expand(1, 4).clone()
            prev_joint_cmd = _robot_scene.data.joint_pos[:, _arm_jids].clone()
            obs = env._get_push_obs()

            cur_obj_pos = obs[0, env.robot_dim:env.robot_dim + 3]
            cur_obj_euler = obs[0, env.robot_dim + 3:env.robot_dim + 6]
            goal_pos = obs[0, env.robot_dim + env.obj_state_dim:env.robot_dim + env.obj_state_dim + 3]
            goal_euler = obs[0, env.robot_dim + env.obj_state_dim + 3:env.robot_dim + env.obj_state_dim + 6]
            pos_err = (cur_obj_pos - goal_pos).norm().item()
            rot_err = _rot_distance_rad(cur_obj_euler.unsqueeze(0), goal_euler.unsqueeze(0)).item()
            if _obj_type == "disc":
                rot_err = 0.0

            if _obj_type == "disc":
                success = pos_err < 0.05
            else:
                success = pos_err < 0.05 and rot_err < args.rot_threshold

            f = _grab_frame()
            if f is not None:
                frames.append(f)

            if success or terminated[0]:
                break
            env.capture_pre_push(obs)

        # hold a few frames at the end so the loop "rests" on the result
        hold_cmd = torch.zeros(1, env.action_space.shape[0], device=device)
        hold_cmd[:, :6] = _robot_scene.data.joint_pos[:, _arm_jids]
        hold_cmd[:, 6] = -1.0
        for _ in range(max(1, args.fps // 2)):
            env.step(hold_cmd)
            f = _grab_frame()
            if f is not None:
                frames.append(f)

        return frames, success

    # ── Record each scene ───────────────────────────────────────────────────────
    n_tests = get_test_count()
    for idx in scene_indices:
        if idx < 1 or idx > n_tests:
            print(f"[skip] scene {idx} out of range (1..{n_tests})")
            continue
        cfg = get_test_config(idx)
        if cfg is None:
            continue
        print(f"\n[Scene {idx}] {cfg.name}  type={cfg.test_type}  "
              f"goal=({cfg.main_goal_x:+.2f},{cfg.main_goal_y:+.2f}) yaw={cfg.main_goal_yaw:+.2f}")

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
        mp4_path = os.path.join(args.out_dir, f"{stem}.mp4")
        key_path = os.path.join(args.out_dir, f"{stem}_key.png")
        try:
            imageio.mimsave(mp4_path, best_frames, fps=args.fps, macro_block_size=None)
            imageio.imwrite(key_path, best_frames[len(best_frames) // 2])
            print(f"  saved {mp4_path} ({'success' if got_success else 'best-effort'}) "
                  f"+ keyframe {key_path}")
        except Exception as e:
            print(f"  [error] encoding failed for scene {idx}: {e}")

    print("\n[Record] Done.")
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
