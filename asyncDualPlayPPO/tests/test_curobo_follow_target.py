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
    from pxr import UsdGeom, Usd, Gf, UsdShade, Sdf
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

    env_cfg.scene.robot.actuators["gripper"].stiffness       = 2000.0
    env_cfg.scene.robot.actuators["manual_mimics"].stiffness = 2000.0

    # Remove env-level zone border strips — we draw our own workspace floor slab.
    for _zb in ("zone_border_top", "zone_border_bottom",
                 "zone_border_left", "zone_border_right"):
        try:
            setattr(env_cfg.scene, _zb, None)
        except AttributeError:
            pass

    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # ── Gamepad ────────────────────────────────────────────────────────────────
    carb.log_info("Initializing Se3Gamepad...")
    # ── Button map (Xbox/standard layout) ─────────────────────────────────────
    # Left  stick  ↑↓     →  +X / -X   (forward/back)
    # Left  stick  ←→     →  +Y / -Y   (left/right)
    # Right stick  ↑↓     →  +Z / -Z   (up/down)
    # Right stick  ←→     →  (unused, rot_sensitivity=0)
    # D-pad        ↑↓←→   →  (unused, rot_sensitivity=0)
    # Left  trigger       →  -Z  (down, via add_callback)
    # Right trigger       →  +Z  (up,   via add_callback)
    # X button            →  toggle gripper open/close
    # A / B / Y           →  available for custom callbacks
    # Left  shoulder (LB) →  available
    # Right shoulder (RB) →  available
    # Left  stick click   →  available
    # Right stick click   →  available
    # MENU1 / MENU2       →  available
    # ──────────────────────────────────────────────────────────────────────────
    gamepad = Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.02, rot_sensitivity=0.0, gripper_term=True))
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

    # Create the pointer directly via UsdGeom — no sim_utils, no physics APIs
    # ever applied, so PhysX is completely unaware of this prim.
    stage = omni.usd.get_context().get_stage()

    # Disable all default stage lights so the scene starts unlit.
    from pxr import UsdLux
    for _prim in stage.Traverse():
        if _prim.IsA(UsdLux.BoundableLightBase) or _prim.IsA(UsdLux.NonboundableLightBase):
            _prim.SetActive(False)
    _cone = UsdGeom.Cone.Define(stage, target_path)
    _cone.GetRadiusAttr().Set(0.02)   # 2 cm base radius
    _cone.GetHeightAttr().Set(0.08)   # 8 cm tall, tip at +Z
    _cone.GetAxisAttr().Set("Z")
    # Red emissive material so it's visible against the scene.
    _mat    = UsdShade.Material.Define(stage, target_path + "/Mat")
    _shader = UsdShade.Shader.Define(stage, target_path + "/Mat/Shader")
    _shader.CreateIdAttr("UsdPreviewSurface")
    _shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.1, 0.1))
    _shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.0, 0.0))
    _mat.CreateSurfaceOutput().ConnectToSource(_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(_cone).Bind(_mat)
    # Place the cone: XformCommonAPI works cleanly on a fresh prim.
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
    _WS_Z_MIN, _WS_Z_MAX = -0.02,  0.80
    _vb = 0.03
    _border_cfg = sim_utils.CuboidCfg(
        size=[1.0, 1.0, 1.0],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0), opacity=0.25),
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
    # Floor slab is deferred until after env_origin is known (see below).

    # ── Block spawner (press C) ────────────────────────────────────────────────
    import random
    import omni.appwindow

    _BLOCK_USDS = [
        os.path.join(os.path.dirname(__file__), "..", "..", "asyncDualPlayPPO", "assets", "blocks", f)
        for f in ("cube.usd", "rect.usd", "cylinder.usd", "concave.usd", "triangle.usd")
    ]
    _spawned_blocks: list[str] = []   # prim paths, oldest first
    _spawn_idx = [0]
    _spawn_requested = [False]

    def _spawn_block():
        # Evict oldest when at capacity.
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
        # The USD files already carry RigidBodyAPI on their inner baseLink —
        # adding physics on the outer xform too creates an invalid hierarchy.
        # Unique random colour applied as a material override on the outer xform.
        r, g, b = random.random(), random.random(), random.random()
        _bmat    = UsdShade.Material.Define(stage, prim_path + "/BlockMat")
        _bshader = UsdShade.Shader.Define(stage,   prim_path + "/BlockMat/Shader")
        _bshader.CreateIdAttr("UsdPreviewSurface")
        _bshader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        _bshader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.5)
        _bmat.CreateSurfaceOutput().ConnectToSource(_bshader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(xform.GetPrim()).Bind(_bmat)
        _spawned_blocks.append(prim_path)
        print(f"  [C] Spawned {os.path.basename(usd_file)}  pos=({x:.2f},{y:.2f})  scale={scale:.2f}  total={len(_spawned_blocks)}")

    _reset_requested = [False]

    def _on_keyboard(event, *_args, **_kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.C:
                _spawn_requested[0] = True
            elif event.input == carb.input.KeyboardInput.R:
                _reset_requested[0] = True
        return True

    _appwindow  = omni.appwindow.get_default_app_window()
    _input_iface = carb.input.acquire_input_interface()
    _kb_sub = _input_iface.subscribe_to_keyboard_events(  # noqa: F841
        _appwindow.get_keyboard(), _on_keyboard
    )

    # ── Cameras & viewport panels ──────────────────────────────────────────────
    import omni.kit.viewport.utility as vp_util

    def _make_lookat_matrix(eye, target):
        """USD row-vector look-at matrix for a camera at eye pointing toward target.
        USD cameras look along local -Z with local +Y up."""
        e   = Gf.Vec3d(*eye)
        t   = Gf.Vec3d(*target)
        fwd = (t - e).GetNormalized()
        # Use Y-up when fwd is near-vertical, otherwise Z-up.
        world_up = Gf.Vec3d(0, 1, 0) if abs(fwd[2]) > 0.9 else Gf.Vec3d(0, 0, 1)
        right    = Gf.Cross(fwd, world_up).GetNormalized()
        up       = Gf.Cross(right, fwd)
        nfwd     = -fwd
        # Rows = local axes expressed in world space; row3 = translation.
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

    _ws_y_mid = (_WS_Y_MIN + _WS_Y_MAX) / 2.0

    # Side camera: right of table at ~45°, looking at arm + workspace.
    _cam_side = _add_camera(
        "/World/CamSide",
        eye    = (1.5,  _ws_y_mid, 0.15),
        target = (0.0,  _ws_y_mid, 0.0),
    )
    # Top camera: straight down over the workspace centre.
    _cam_top = _add_camera(
        "/World/CamTop",
        eye    = (0.0,  _ws_y_mid, 1.3),
        target = (0.0,  _ws_y_mid, 0.0),
        focal_mm=18.0,
    )

    # Viewport windows stacked on the left — pass positions to the constructor
    # so the underlying setPosition() call fires correctly.
    _vp_side = vp_util.create_viewport_window(
        "Side View", width=420, height=280, position_x=0, position_y=0
    )
    _vp_side.viewport_api.camera_path = _cam_side

    _vp_top = vp_util.create_viewport_window(
        "Top View", width=420, height=280, position_x=0, position_y=290
    )
    _vp_top.viewport_api.camera_path = _cam_top

    # Zoom the main viewport in by 50% — move its camera halfway toward the
    # robot working-area centre so the scene fills the screen more tightly.
    _active_vp = vp_util.get_active_viewport()
    _main_cam_prim = stage.GetPrimAtPath(_active_vp.camera_path)
    if _main_cam_prim.IsValid():
        _cam_xform = UsdGeom.Xformable(_main_cam_prim)
        _eye = _cam_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        _focus = Gf.Vec3d(0.0, 0.5, 0.4)
        _new_eye = _focus + (_eye - _focus) * 0.5
        _main_translate_op = next(
            (op for op in _cam_xform.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if _main_translate_op is not None:
            _main_translate_op.Set(_new_eye)
        else:
            UsdGeom.XformCommonAPI(_main_cam_prim).SetTranslate(_new_eye)

    def _read_ball_pos() -> torch.Tensor:
        mat = _xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return torch.tensor([float(t[0]), float(t[1]), float(t[2])], device=device, dtype=torch.float32)

    # ── Control loop state ─────────────────────────────────────────────────────
    env_origin = env.scene.env_origins[0]  # (3,) — robot base in world frame

    # Floor slab: placed now that env_origin is known so world coords are correct.
    _ox, _oy, _oz = env_origin[0].item(), env_origin[1].item(), env_origin[2].item()
    _ws_slab = UsdGeom.Cube.Define(stage, "/World/WsFloor")
    _ws_slab.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(_ws_slab).AddTranslateOp().Set(
        Gf.Vec3d(_ox, _oy + float(_y_mid), _oz + 0.003)
    )
    UsdGeom.Xformable(_ws_slab).AddScaleOp().Set(
        Gf.Vec3f(_WS_X_MAX - _WS_X_MIN, _WS_Y_MAX - _WS_Y_MIN, 0.001)
    )
    _ws_mat    = UsdShade.Material.Define(stage, "/World/WsFloor/Mat")
    _ws_shader = UsdShade.Shader.Define(stage,   "/World/WsFloor/Mat/Shader")
    _ws_shader.CreateIdAttr("UsdPreviewSurface")
    _ws_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15, 0.45, 0.15))
    _ws_shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.9)
    _ws_mat.CreateSurfaceOutput().ConnectToSource(_ws_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(_ws_slab.GetPrim()).Bind(_ws_mat)

    # Fixed "tool pointing down" orientation — matches the working IK config.
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    current_ik_target_w = torch.tensor(ball_spawn, device=device, dtype=torch.float32)
    last_good_joints     = reset_joints[0].clone()
    max_step_delta = args.max_vel * env.step_dt

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

            if _reset_requested[0]:
                _reset_requested[0] = False
                env.reset()
                # Hold reset joints for 20 steps so PhysX settles to default pose.
                _hold = torch.zeros((1, env.action_space.shape[-1]), device=device)
                _hold[0, :6] = reset_joints[0]
                for _ in range(20):
                    env.step(_hold)
                robot.update(env.step_dt)
                last_good_joints = reset_joints[0].clone()
                action[0, :6] = last_good_joints
                current_ik_target_w = torch.tensor(ball_spawn, device=device, dtype=torch.float32)
                if _translate_op is not None:
                    with Usd.EditContext(stage, _session_layer):
                        _translate_op.Set(Gf.Vec3d(
                            float(ball_spawn[0]), float(ball_spawn[1]), float(ball_spawn[2])
                        ))
                gamepad.reset()
                print("  [R] Reset — robot and target returned to start pose.")

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
            # gamepad_cmd[6]: +1.0 = open, -1.0 = close (X button toggles)
            action[0, 6] = gamepad_cmd[6]
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
                last_good_joints = ik_result.solution.view(-1)[:6].clone()
                action[0, :6] = last_good_joints
            else:
                # Fall back to last known-good joints so the robot holds pose.
                action[0, :6] = last_good_joints

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
