"""
CuRobo Interactive Follow-Target Test (IsaacLab)
=================================================

Spawns a red visual sphere at the gripper midpoint that you can move with
the viewport gizmo. CuRobo computes IK in real-time; a velocity clamp makes
the robot chase the ball smoothly.

Usage:
    python tests/test_curobo_follow_target.py --max_vel 0.5

Controls:
    1. Click the Red Ball in the viewport.
    2. Press 'W' to activate the translation gizmo.
    3. Drag the ball — the arm follows in real-time.
"""

import argparse
import os
import sys
import torch

# CuRobo MUST be imported before AppLauncher to avoid library conflicts.
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
except ModuleNotFoundError:
    print("\n[ERROR] CuRobo not found in .master_venv. Ensure it is installed.")
    sys.exit(1)

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    parser = argparse.ArgumentParser(description="CuRobo Interactive Follow Target")
    parser.add_argument("--max_vel", type=float, default=2.0, help="Max EE velocity in m/s")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ── Post-launch imports ────────────────────────────────────────────────────
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    import isaaclab.sim as sim_utils
    import omni.usd
    import numpy as np
    from pxr import UsdGeom, Usd, Gf
    from isaaclab.devices import Se3Gamepad, Se3GamepadCfg
    import carb

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # ── Environment ────────────────────────────────────────────────────────────
    print("\nCreating Environment...")
    env_cfg = AsyncDualPlayEnvCfg()
    env_cfg.scene.num_envs = 1

    env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )
    env_cfg.terminations.robot_through_table = None
    env_cfg.terminations.objects_off_table = None
    try:
        env_cfg.scene.cube = None
    except AttributeError:
        pass
    try:
        env_cfg.scene.target_object = None
    except AttributeError:
        pass
    for _attr in ("object_state", "cube_state", "goal_state", "goal_distance",
                  "cube_goal_state", "cube_goal_distance"):
        for _group in (env_cfg.observations.alice_policy, env_cfg.observations.bob_policy):
            try:
                setattr(_group, _attr, None)
            except AttributeError:
                pass

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # ── Gamepad ────────────────────────────────────────────────────────────────
    carb.log_info("Initializing Se3Gamepad...")
    gamepad = Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.02, rot_sensitivity=0.0, gripper_term=False))
    gamepad.reset()

    # ── CuRobo IK solver ──────────────────────────────────────────────────────
    print("\nInitializing CuRobo IKSolver...")
    tensor_args = TensorDeviceType(device=torch.device(device), dtype=torch.float32)
    ur5e_yaml = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
    robot_cfg = RobotConfig.from_dict(ur5e_yaml["robot_cfg"], tensor_args)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg, world_model=None, tensor_args=tensor_args
    )
    ik_solver = IKSolver(ik_config)

    # ── Reset and warm-up ──────────────────────────────────────────────────────
    env.reset()
    robot = env.scene["robot"]
    robot.update(env.step_dt)

    wrist3_ids, _ = robot.find_bodies("wrist_3_link")
    wrist3_id = wrist3_ids[0]

    left_finger_ids, _  = robot.find_bodies("left_inner_finger")
    right_finger_ids, _ = robot.find_bodies("right_inner_finger")
    left_finger_id  = left_finger_ids[0]
    right_finger_id = right_finger_ids[0]

    joint_indices, found_names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    print(f"  Joint index mapping: {list(zip(found_names, joint_indices))}")

    reset_joints = robot.data.joint_pos[:, joint_indices].clone()  # (1, 6)
    print(f"  Init joints: {reset_joints[0].cpu().numpy().round(3)}")

    # Hold reset pose for 10 steps so PhysX transforms settle.
    hold_action = torch.zeros((1, env.action_space.shape[-1]), device=device)
    hold_action[0, :6] = reset_joints[0]
    for _ in range(10):
        env.step(hold_action)
    robot.update(env.step_dt)

    # ── Spawn red target sphere at gripper midpoint ───────────────────────────
    left_pos  = robot.data.body_pos_w[0, left_finger_id]
    right_pos = robot.data.body_pos_w[0, right_finger_id]
    ball_spawn = ((left_pos + right_pos) / 2.0).cpu().numpy()
    ball_spawn[2] += 0.05  # 5 cm above gripper midpoint

    target_path = "/World/interactive_target"
    sphere_cfg = sim_utils.SphereCfg(
        radius=0.03,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    sphere_cfg.func(target_path, sphere_cfg, translation=ball_spawn)

    stage = omni.usd.get_context().get_stage()
    target_prim = stage.GetPrimAtPath(target_path)
    _xformable = UsdGeom.Xformable(target_prim)
    _session_layer = stage.GetSessionLayer()
    # Find the existing translate op so we can write to it directly.
    # XformCommonAPI.SetTranslate fails on IsaacLab-spawned prims whose xform
    # op layout doesn't match the CommonAPI schema.
    _translate_op = next(
        (op for op in _xformable.GetOrderedXformOps()
         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )

    # Visual workspace border boxes
    _WS_X_MIN, _WS_X_MAX = -0.75,  0.75
    _WS_Y_MIN, _WS_Y_MAX =  0.20,  1.00
    _WS_Z_MIN, _WS_Z_MAX =  0.02,  1.05
    _vb = 0.03
    _border_cfg = sim_utils.CuboidCfg(
        size=[1.0, 1.0, 1.0],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0), opacity=0.25),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    _y_mid   = (_WS_Y_MIN + _WS_Y_MAX) / 2.0
    _z_mid   = (_WS_Z_MIN + _WS_Z_MAX) / 2.0
    _x_span  = (_WS_X_MAX - _WS_X_MIN) + 2 * _vb
    _y_span  = (_WS_Y_MAX - _WS_Y_MIN) + 2 * _vb
    _z_span  =  _WS_Z_MAX - _WS_Z_MIN
    for _name, _pos, _scale in [
        ("WsBorderTop",    [0.0,              _WS_Y_MAX + _vb, _z_mid], [_x_span, 0.01, _z_span]),
        ("WsBorderBottom", [0.0,              _WS_Y_MIN - _vb, _z_mid], [_x_span, 0.01, _z_span]),
        ("WsBorderLeft",   [_WS_X_MIN - _vb, _y_mid,          _z_mid], [0.01, _y_span, _z_span]),
        ("WsBorderRight",  [_WS_X_MAX + _vb, _y_mid,          _z_mid], [0.01, _y_span, _z_span]),
        ("WsBorderCeil",   [0.0,              _y_mid,  _WS_Z_MAX + _vb], [_x_span, _y_span, 0.01]),
    ]:
        _border_cfg.func(f"/World/{_name}", _border_cfg,
                         translation=np.array(_pos), scale=np.array(_scale))

    def _read_ball_pos() -> torch.Tensor:
        mat = _xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return torch.tensor([float(t[0]), float(t[1]), float(t[2])], device=device, dtype=torch.float32)

    # ── Control loop state ─────────────────────────────────────────────────────
    env_origin = env.scene.env_origins[0]  # (3,) — robot base in world frame

    # Fixed "tool pointing down" orientation — matches the working IK config.
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    current_ik_target_w = torch.tensor(ball_spawn, device=device, dtype=torch.float32)
    max_step_delta = args.max_vel * env.step_dt
    MAX_REACH = 0.78
    MIN_Z = 0.02

    def _clamp_ball(pos: torch.Tensor) -> torch.Tensor:
        """Clamp pos to the UR5e reachable workspace and write back to USD if moved."""
        local = pos - env_origin
        # Radial reach limit.
        reach = local.norm()
        if reach > MAX_REACH:
            local = local * (MAX_REACH / reach)
        # Z floor.
        if local[2] < MIN_Z:
            local[2] = MIN_Z
        clamped = local + env_origin
        if _translate_op is not None and not torch.allclose(pos, clamped, atol=1e-4):
            with Usd.EditContext(stage, _session_layer):
                _translate_op.Set(Gf.Vec3d(clamped[0].item(), clamped[1].item(), clamped[2].item()))
        return clamped

    action = torch.zeros((1, env.action_space.shape[-1]), device=device)
    action[0, :6] = reset_joints[0]

    print("\n" + "=" * 70)
    print("  [Interactive Mode Ready]")
    print("  1. Click the Red Ball in the viewport.")
    print("  2. Press 'W' to activate the translation gizmo.")
    print("  3. Drag the ball — the arm follows in real-time.")
    print(f"  Max velocity : {args.max_vel} m/s")
    print(f"  Env origin   : {env_origin.cpu().numpy().round(3)}")
    print(f"  Ball spawn   : {ball_spawn.round(3)}")
    print("=" * 70 + "\n")

    step_count = 0
    ik_ok_count = 0

    try:
        while simulation_app.is_running():
            simulation_app.update()

            # Apply gamepad XYZ delta to ball's USD position.
            gamepad_cmd = gamepad.advance()
            delta_pos = gamepad_cmd[:3].cpu().numpy()
            if delta_pos.any():
                mat = _xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = mat.ExtractTranslation()
                if _translate_op is not None:
                    with Usd.EditContext(stage, _session_layer):
                        _translate_op.Set(Gf.Vec3d(t[0] + float(delta_pos[0]),
                                                   t[1] + float(delta_pos[1]),
                                                   t[2] + float(delta_pos[2])))

            ball_w = _clamp_ball(_read_ball_pos())

            # Velocity-clamp: advance IK target toward ball.
            delta = ball_w - current_ik_target_w
            dist = delta.norm()
            if dist > max_step_delta:
                current_ik_target_w = current_ik_target_w + (delta / dist) * max_step_delta
            else:
                current_ik_target_w = ball_w.clone()

            # Workspace clamp in local (env-origin-relative) frame.
            local_target = current_ik_target_w - env_origin
            reach = local_target.norm()
            if reach > MAX_REACH:
                local_target = local_target * (MAX_REACH / reach)
                current_ik_target_w = local_target + env_origin
            if local_target[2] < MIN_Z:
                local_target[2] = MIN_Z
                current_ik_target_w[2] = env_origin[2] + MIN_Z

            # TCP offset: CuRobo targets wrist_3_link; subtract the live
            # wrist→gripper-midpoint offset so the fingers reach the ball.
            left_pos_w   = robot.data.body_pos_w[0, left_finger_id]
            right_pos_w  = robot.data.body_pos_w[0, right_finger_id]
            tcp_pos_w_cur = (left_pos_w + right_pos_w) / 2.0
            wrist3_pos_w  = robot.data.body_pos_w[0, wrist3_id]
            tcp_offset    = tcp_pos_w_cur - wrist3_pos_w

            ik_local_target = local_target - tcp_offset

            # Solve IK seeded from current joints.
            cur_joints = robot.data.joint_pos[:, joint_indices]  # (1, 6)
            goal_pose = Pose(
                position=ik_local_target.unsqueeze(0),
                quaternion=target_quat,
            )
            ik_result = ik_solver.solve_single(
                goal_pose,
                seed_config=cur_joints.unsqueeze(1),  # (1, 1, 6)
                retract_config=cur_joints,             # (1, 6)
            )

            if ik_result.success.any():
                ik_ok_count += 1
                action[0, :6] = ik_result.solution.view(-1)[:6]
            else:
                action[0, :6] = cur_joints.view(-1)

            env.step(action)
            robot.update(env.step_dt)
            step_count += 1

            if step_count % 10 == 0:
                tcp_local = tcp_pos_w_cur - env_origin
                err = (tcp_local - local_target).norm().item()
                ok = ik_result.success.any().item()
                print(
                    f"[{step_count:5d}] "
                    f"target=({local_target[0]:.3f},{local_target[1]:.3f},{local_target[2]:.3f}) "
                    f"tcp=({tcp_local[0]:.3f},{tcp_local[1]:.3f},{tcp_local[2]:.3f}) "
                    f"err={err:.4f}m  IK={'OK  ' if ok else 'FAIL'} "
                    f"({ik_ok_count}/{step_count})"
                )

    except KeyboardInterrupt:
        print("\nExiting.")

    simulation_app.close()


if __name__ == "__main__":
    main()
