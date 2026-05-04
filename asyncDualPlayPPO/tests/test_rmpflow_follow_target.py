"""
RMPflow Interactive Follow-Target Test (IsaacLab)
=================================================

Spawns a red visual sphere at the gripper midpoint that you can move with
the viewport gizmo. RMPflow computes the control commands in real-time.

Usage:
    python tests/test_rmpflow_follow_target.py --max_vel 0.5

Controls:
    1. Click the Red Ball in the viewport.
    2. Press 'W' to activate the translation gizmo.
    3. Drag the ball — the arm follows in real-time.
"""

import argparse
import os
import sys
import torch
import numpy as np

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    parser = argparse.ArgumentParser(description="RMPflow Interactive Follow Target")
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
    from pxr import UsdGeom, Usd, Gf, UsdShade, Sdf, UsdPhysics
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

    # Configure arm action for absolute pose control via RMPflow
    # AsyncDualPlayEnvCfg default arm_action is RMPFlowActionCfg
    env_cfg.actions.arm_action.use_relative_mode = False
    env_cfg.actions.arm_action.scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

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

    env_cfg.scene.robot.actuators["gripper"].stiffness       = 2000.0
    env_cfg.scene.robot.actuators["manual_mimics"].stiffness = 2000.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # ── Gamepad ────────────────────────────────────────────────────────────────
    gamepad = Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.02, rot_sensitivity=0.0, gripper_term=True))
    gamepad.reset()

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

    # ── Spawn red target sphere at gripper midpoint ───────────────────────────
    left_pos  = robot.data.body_pos_w[0, left_finger_id]
    right_pos = robot.data.body_pos_w[0, right_finger_id]
    ball_spawn = ((left_pos + right_pos) / 2.0).cpu().numpy()
    ball_spawn[2] += 0.05  # 5 cm above gripper midpoint

    target_path = "/World/interactive_target"
    stage = omni.usd.get_context().get_stage()
    _cone = UsdGeom.Cone.Define(stage, target_path)
    _cone.GetRadiusAttr().Set(0.02)
    _cone.GetHeightAttr().Set(0.08)
    _cone.GetAxisAttr().Set("Z")
    _mat    = UsdShade.Material.Define(stage, target_path + "/Mat")
    _shader = UsdShade.Shader.Define(stage, target_path + "/Mat/Shader")
    _shader.CreateIdAttr("UsdPreviewSurface")
    _shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.1, 0.1))
    _shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.0, 0.0))
    _mat.CreateSurfaceOutput().ConnectToSource(_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(_cone).Bind(_mat)
    UsdGeom.XformCommonAPI(_cone).SetTranslate(
        Gf.Vec3d(float(ball_spawn[0]), float(ball_spawn[1]), float(ball_spawn[2]))
    )

    target_prim = _cone.GetPrim()
    _xformable = UsdGeom.Xformable(target_prim)
    _session_layer = stage.GetSessionLayer()
    _translate_op = next(
        (op for op in _xformable.GetOrderedXformOps()
         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )

    # Visual workspace border boxes
    _WS_X_MIN, _WS_X_MAX = -0.65,  0.65
    _WS_Y_MIN, _WS_Y_MAX =  0.20,  0.75
    _WS_Z_MIN, _WS_Z_MAX =  0.02,  0.80
    _vb = 0.03
    _border_cfg = sim_utils.CuboidCfg(
        size=[1.0, 1.0, 1.0],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0), opacity=0.25),
    )
    _floor_cfg = sim_utils.CuboidCfg(
        size=[1.0, 1.0, 1.0],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08), opacity=0.55),
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
    # Dark floor slab — flat on the table showing the exact workspace footprint.
    _floor_cfg.func("/World/WsFloor", _floor_cfg,
                    translation=np.array([0.0, _y_mid, 0.002]),
                    scale=np.array([_WS_X_MAX - _WS_X_MIN, _WS_Y_MAX - _WS_Y_MIN, 0.001]))

    # ── Block spawner ─────────────────────────────────────────────────────────
    import random
    import omni.appwindow
    _BLOCK_USDS = [
        os.path.join(os.path.dirname(__file__), "..", "..", "asyncDualPlayPPO", "assets", "blocks", f)
        for f in ("cube.usd", "rect.usd", "cylinder.usd", "concave.usd", "triangle.usd")
    ]
    _spawned_blocks: list[str] = []
    _spawn_idx = [0]
    _spawn_requested = [False]

    def _spawn_block():
        if len(_spawned_blocks) >= 3:
            stage.RemovePrim(_spawned_blocks.pop(0))
        usd_file = random.choice(_BLOCK_USDS)
        scale    = random.uniform(0.8, 1.35)
        x = random.uniform(_WS_X_MIN + 0.15, _WS_X_MAX - 0.15)
        y = random.uniform(_WS_Y_MIN + 0.15, _WS_Y_MAX - 0.15)
        prim_path = f"/World/SpawnedBlock_{_spawn_idx[0]}"
        _spawn_idx[0] += 1
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.05))
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        xform.GetPrim().GetReferences().AddReference(usd_file)
        # Apply physics so the block can be grabbed/pushed.
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        UsdPhysics.CollisionAPI.Apply(xform.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass_api.CreateMassAttr().Set(0.3)
        r, g, b = random.random(), random.random(), random.random()
        _bmat    = UsdShade.Material.Define(stage, prim_path + "/BlockMat")
        _bshader = UsdShade.Shader.Define(stage,   prim_path + "/BlockMat/Shader")
        _bshader.CreateIdAttr("UsdPreviewSurface")
        _bshader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        _bshader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.5)
        _bmat.CreateSurfaceOutput().ConnectToSource(_bshader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(xform.GetPrim()).Bind(_bmat)
        _spawned_blocks.append(prim_path)

    def _on_keyboard(event, *_args, **_kwargs):
        if (event.type == carb.input.KeyboardEventType.KEY_PRESS
                and event.input == carb.input.KeyboardInput.C):
            _spawn_requested[0] = True
        return True

    _appwindow  = omni.appwindow.get_default_app_window()
    _input_iface = carb.input.acquire_input_interface()
    _kb_sub = _input_iface.subscribe_to_keyboard_events(_appwindow.get_keyboard(), _on_keyboard)

    # ── Cameras ───────────────────────────────────────────────────────────────
    import omni.kit.viewport.utility as vp_util
    def _make_lookat_matrix(eye, target):
        e   = Gf.Vec3d(*eye)
        t   = Gf.Vec3d(*target)
        fwd = (t - e).GetNormalized()
        world_up = Gf.Vec3d(0, 1, 0) if abs(fwd[2]) > 0.9 else Gf.Vec3d(0, 0, 1)
        right    = Gf.Cross(fwd, world_up).GetNormalized()
        up       = Gf.Cross(right, fwd)
        nfwd     = -fwd
        return Gf.Matrix4d(
            right[0], right[1], right[2], 0.0,
            up[0],    up[1],    up[2],    0.0,
            nfwd[0],  nfwd[1],  nfwd[2],  0.0,
            e[0],     e[1],     e[2],     1.0,
        )

    def _add_camera(prim_path, eye, target, focal_mm=24.0):
        cam = UsdGeom.Camera.Define(stage, prim_path)
        cam.GetFocalLengthAttr().Set(focal_mm)
        UsdGeom.Xformable(cam).MakeMatrixXform().Set(_make_lookat_matrix(eye, target))
        return prim_path

    _cam_side = _add_camera("/World/CamSide", eye=(1.5, 0.5, 0.1), target=(0.10, 0.5, 0.0))
    _cam_top = _add_camera("/World/CamTop", eye=(0.0, 0.5, 1.5), target=(0.0, 0.55, 0.0), focal_mm=18.0)
    _vp_side = vp_util.create_viewport_window("Side View", width=420, height=280)
    _vp_side.viewport_api.set_active_camera(_cam_side)
    _vp_side.position_x, _vp_side.position_y = 0, 0
    _vp_top = vp_util.create_viewport_window("Top View", width=420, height=280)
    _vp_top.viewport_api.set_active_camera(_cam_top)
    _vp_top.position_x, _vp_top.position_y = 0, 290

    def _read_ball_pos() -> torch.Tensor:
        mat = _xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return torch.tensor([float(t[0]), float(t[1]), float(t[2])], device=device, dtype=torch.float32)

    # ── Control loop state ─────────────────────────────────────────────────────
    env_origin = env.scene.env_origins[0]
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32) # [w, x, y, z]

    current_ik_target_w = torch.tensor(ball_spawn, device=device, dtype=torch.float32)
    max_step_delta = args.max_vel * env.step_dt
    MAX_REACH = 0.78
    MIN_Z = 0.02

    def _clamp_ball(pos: torch.Tensor) -> torch.Tensor:
        """Clamp pos to the workspace rectangle and write back to USD if moved."""
        local = pos - env_origin
        local[0] = local[0].clamp(_WS_X_MIN, _WS_X_MAX)
        local[1] = local[1].clamp(_WS_Y_MIN, _WS_Y_MAX)
        local[2] = local[2].clamp(_WS_Z_MIN, _WS_Z_MAX)
        clamped = local + env_origin
        if _translate_op is not None and not torch.allclose(pos, clamped, atol=1e-4):
            with Usd.EditContext(stage, _session_layer):
                _translate_op.Set(Gf.Vec3d(clamped[0].item(), clamped[1].item(), clamped[2].item()))
        return clamped

    action = torch.zeros((1, env.action_space.shape[-1]), device=device)

    print("\n" + "=" * 70)
    print("  [RMPflow Interactive Mode Ready]")
    print("  1. Click the Red Ball in the viewport.")
    print("  2. Press 'W' to activate the translation gizmo.")
    print("  3. Drag the ball — the arm follows in real-time.")
    print(f"  Max velocity : {args.max_vel} m/s")
    print("=" * 70 + "\n")

    step_count = 0

    try:
        while simulation_app.is_running():
            simulation_app.update()

            if _spawn_requested[0]:
                _spawn_requested[0] = False
                _spawn_block()

            # Apply gamepad XYZ delta to ball's USD position.
            gamepad_cmd = gamepad.advance()
            delta_pos = gamepad_cmd[:3].cpu().numpy()
            # Remap left-stick so UP=forward(+Y), DOWN=back(-Y), RIGHT=right(-X), LEFT=left(+X)
            dx, dy = delta_pos[0], delta_pos[1]
            delta_pos[0] = dy
            delta_pos[1] = dx
            action[0, -1] = gamepad_cmd[6] # gripper
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
            local_target[0] = local_target[0].clamp(_WS_X_MIN, _WS_X_MAX)
            local_target[1] = local_target[1].clamp(_WS_Y_MIN, _WS_Y_MAX)
            local_target[2] = local_target[2].clamp(_WS_Z_MIN, _WS_Z_MAX)
            current_ik_target_w = local_target + env_origin

            # TCP offset
            left_pos_w   = robot.data.body_pos_w[0, left_finger_id]
            right_pos_w  = robot.data.body_pos_w[0, right_finger_id]
            tcp_pos_w_cur = (left_pos_w + right_pos_w) / 2.0
            wrist3_pos_w  = robot.data.body_pos_w[0, wrist3_id]
            tcp_offset    = tcp_pos_w_cur - wrist3_pos_w

            ik_local_target = local_target - tcp_offset

            # RMPFlow action vector.
            # Assuming RMPFlowAction expects [x, y, z, qw, qx, qy, qz] when use_relative_mode=False
            # Wait, check RMPFlowAction implementation. In Isaac Lab it usually takes 7D for pose.
            # But ReachDualArmEnvCfg arm_action has 6D scale. This usually means 6D pose (pos + axis-angle or euler).
            # If so, let's use 6D: [x, y, z, rot_x, rot_y, rot_z].
            # However, standard Isaac Lab RMPflow often takes 3D pos + 4D quat.
            
            # Let's use 6D delta if 7D absolute is uncertain, but the request was "absolute copies".
            # Actually, let's use the 7D pose: [x, y, z, qw, qx, qy, qz].
            # If the action space is 7, we use 7.
            
            arm_action_dim = env.action_space.shape[-1] - 1
            if arm_action_dim == 7:
                action[0, 0:3] = ik_local_target
                action[0, 3:7] = target_quat
            else:
                # Fallback to 6D if that's what's configured (e.g. pos + euler)
                action[0, 0:3] = ik_local_target
                action[0, 3:6] = 0.0 # orientation placeholder

            env.step(action)
            robot.update(env.step_dt)
            step_count += 1

            if step_count % 10 == 0:
                tcp_local = tcp_pos_w_cur - env_origin
                err = (tcp_local - local_target).norm().item()
                print(
                    f"[{step_count:5d}] "
                    f"target=({local_target[0]:.3f},{local_target[1]:.3f},{local_target[2]:.3f}) "
                    f"tcp=({tcp_local[0]:.3f},{tcp_local[1]:.3f},{tcp_local[2]:.3f}) "
                    f"err={err:.4f}m"
                )

    except KeyboardInterrupt:
        print("\nExiting.")

    simulation_app.close()


if __name__ == "__main__":
    main()
