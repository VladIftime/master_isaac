"""
CuRobo Interactive Follow-Target Test (IsaacLab)
=================================================

Spawns a red visual cone at the gripper midpoint that you can move with the
viewport gizmo or an Xbox Bluetooth controller. CuRobo computes IK in
real-time; a velocity clamp makes the robot chase the target smoothly.

Usage:
    python tests/test_curobo_follow_target.py --max_vel 0.5

Xbox controller layout:
    Left  stick  ↑↓ / ←→  →  +X/-X  and  +Y/-Y   (forward/back, left/right)
    Right stick  ↑↓        →  +Rx/-Rx  (gripper tilt forward/back)
    Right stick  ←→        →  +Ry/-Ry  (gripper tilt left/right)
    Right trigger           →  +Z  (raise target)
    Left  trigger           →  -Z  (lower target)
    X button                →  toggle gripper open/close
    R key                   →  reset robot and target to start pose
    C key                   →  spawn a random block on the table
    N key / DPAD_UP         →  cycle to next validation test scene
    B button / R key        →  reset current test scene
"""

import argparse
import os
import sys
import torch
import torch._dynamo          # noqa: F401  lock venv torch before AppLauncher prepends Isaac Sim's pip_prebundle
import torch._C               # noqa: F401
import torch.optim            # noqa: F401

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# cuRobo MUST be imported before AppLauncher — importing after causes a native
# crash in librtx.scenedb.plugin.so because CuRobo's CUDA context init races
# with the RTX rendering background thread (carbOnPluginStartup).
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
except ModuleNotFoundError:
    print("\n[ERROR] CuRobo not found. Ensure it is installed.")
    sys.exit(1)

from isaaclab.app import AppLauncher

from asyncDualPlayPPO.tests.validation_configs import (
    get_test_count,
    get_test_config,
    get_test_label,
    PushTestConfig,
)


def main():
    parser = argparse.ArgumentParser(description="CuRobo Interactive Follow Target")
    parser.add_argument("--max_vel",     type=float, default=2.0, help="Max EE velocity in m/s")
    parser.add_argument("--gamepad_idx", type=int,   default=1,   help="Carb gamepad index (0=first, 1=second, …)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Force synchronous RTX rendering so SceneDB is fully initialized before
    # SetLightingMenuModeCommand fires — avoids a race-condition crash on
    # slower GPUs (RTX 3060 Ti + IOMMU enabled).
    for _flag in ("--/app/asyncRendering=false", "--/app/asyncRenderingLowLatency=false"):
        if _flag not in sys.argv:
            sys.argv.append(_flag)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ── Post-launch imports ────────────────────────────────────────────────────
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from asyncDualPlayPPO.tasks.async_dual_play import AsyncDualPlayEnvCfg
    import isaaclab.sim as sim_utils
    import omni.usd
    import numpy as np
    from pxr import UsdGeom, Usd, Gf, UsdShade, Sdf, UsdPhysics
    from scipy.spatial.transform import Rotation as ScipyRot
    import carb
    import struct, threading

    # ── Direct /dev/input joystick reader (bypasses carb gamepad API) ─────────
    class XboxJoystick:
        """Reads Xbox Bluetooth controller directly from /dev/input/jsN.

        Linux joystick protocol: 8-byte packets  (IhBB = ts, value, type, number)
        Xbox Wireless (xpad/bt) axis map:
            0  Left stick X   (-1=left,  +1=right)
            1  Left stick Y   (-1=up,    +1=down)   ← inverted vs. intuition
            2  Left  trigger  (0=rest,   +1=full)
            3  Right stick X  (-1=left,  +1=right)
            4  Right stick Y  (-1=up,    +1=down)   ← inverted
            5  Right trigger  (0=rest,   +1=full)
        Xbox button map:
            0=A  1=B  2=X  3=Y  4=LB  5=RB  6=Back  7=Start  8=Xbox  9=LS  10=RS
        """

        _JS_BTN  = 0x01
        _JS_AXIS = 0x02

        def __init__(self, device="/dev/input/js1", dead_zone=0.10,
                     pos_sensitivity=0.02, rot_sensitivity=0.02):
            self.pos_sensitivity = pos_sensitivity
            self.rot_sensitivity = rot_sensitivity
            self._dz = dead_zone
            self._axes    = [0.0] * 16
            self._buttons = [0]   * 16
            self._gripper_closed = False
            self._spawn_requested = False
            self._reset_requested = False
            self._next_test_requested = False
            self._lb_held = False
            self._rb_held = False
            self._lock = threading.Lock()
            self._fd = -1
            self._connected = False
            self._preferred_device = device  # try this first; fall back to scan
            self._try_connect()
            threading.Thread(target=self._loop, daemon=True).start()

        def _scan_device(self):
            """Return the first available /dev/input/jsN path, preferring _preferred_device."""
            if os.path.exists(self._preferred_device):
                return self._preferred_device
            for i in range(10):
                p = f"/dev/input/js{i}"
                if os.path.exists(p):
                    return p
            return None

        def _try_connect(self):
            dev = self._scan_device()
            if dev is None:
                return False
            try:
                fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
                with self._lock:
                    self._fd = fd
                    self._connected = True
                print(f"  [XboxJoystick] connected {dev}")
                return True
            except OSError:
                return False

        def _loop(self):
            import time
            while True:
                with self._lock:
                    fd = self._fd
                    connected = self._connected
                if not connected:
                    if self._try_connect():
                        pass
                    else:
                        time.sleep(2.0)
                    continue
                try:
                    data = os.read(fd, 8)
                    _ts, value, etype, number = struct.unpack("IhBB", data)
                    etype &= ~0x80  # strip init flag
                    with self._lock:
                        if etype == self._JS_AXIS and number < len(self._axes):
                            prev = self._axes[number]
                            self._axes[number] = value / 32767.0
                            # D-pad vertical axis: up = -1, center = 0, down = +1
                            if number == 6:
                                if prev >= -0.5 and self._axes[6] < -0.5:
                                    self._next_test_requested = True
                        elif etype == self._JS_BTN and number < len(self._buttons):
                            prev = self._buttons[number]
                            self._buttons[number] = value
                            if number == 1 and value == 1 and prev == 0:  # B btn
                                self._reset_requested = True
                            if number == 3 and value == 1 and prev == 0:  # X btn
                                self._gripper_closed = not self._gripper_closed
                            if number == 4 and value == 1 and prev == 0:  # Y btn
                                self._spawn_requested = True
                            if number == 6:  # LB — rotate CCW
                                self._lb_held = bool(value)
                            if number == 7:  # RB — rotate CW
                                self._rb_held = bool(value)
                            if number == 11 and value == 1 and prev == 0:
                                self._next_test_requested = True
                except BlockingIOError:
                    time.sleep(0.001)
                except OSError:
                    # Device disconnected — reset state and wait for reconnect.
                    with self._lock:
                        try: os.close(self._fd)
                        except OSError: pass
                        self._fd = -1
                        self._connected = False
                        self._axes    = [0.0] * 16
                        self._buttons = [0]   * 16
                        self._lb_held = False
                        self._rb_held = False
                    print("  [XboxJoystick] disconnected — reconnecting automatically…")
                except Exception:
                    break

        def _ax(self, n):
            v = self._axes[n]
            return 0.0 if abs(v) < self._dz else v

        def advance(self) -> torch.Tensor:
            """Return 7-element tensor [dx, dy, dz, drx, dry, drz, gripper]
            in the same convention as Se3Gamepad.advance()."""
            with self._lock:
                if not self._connected:
                    return torch.zeros(7, dtype=torch.float32)
                # Left stick: axis 0 = X (L/R), axis 1 = Y (up=neg, invert)
                # Se3Gamepad convention: cmd[0] = left-stick-up (+), cmd[1] = left-stick-right (+)
                cmd0 =   self._ax(0) * self.pos_sensitivity   # left stick X, right=+
                cmd1 =  -self._ax(1) * self.pos_sensitivity   # left stick Y, up=+
                # Triggers rest at -32767, full press = +32767.
                # Normalise to 0..1: (raw_norm + 1) / 2
                rt = (self._axes[5] + 1.0) / 2.0   # RT axis 5: up
                lt = (self._axes[2] + 1.0) / 2.0   # LT axis 2: down
                cmd2 = (rt - lt) * self.pos_sensitivity
                # Right stick: axis 3 = X (L/R), axis 4 = Y (up=neg, invert)
                cmd3 = -self._ax(4) * self.rot_sensitivity    # Rx: right-stick Y, up=+
                cmd4 =  self._ax(3) * self.rot_sensitivity    # Ry: right-stick X, right=+
                cmd5 = (self._lb_held - self._rb_held) * self.rot_sensitivity  # Rz: LB=CCW, RB=CW
                gripper = -1.0 if self._gripper_closed else 1.0
            return torch.tensor([cmd0, cmd1, cmd2, cmd3, cmd4, cmd5, gripper],
                                 dtype=torch.float32)

        def pop_reset(self) -> bool:
            """Returns True (and clears the flag) if B was pressed since last call."""
            with self._lock:
                if self._reset_requested:
                    self._reset_requested = False
                    return True
            return False

        def pop_spawn(self) -> bool:
            """Returns True (and clears the flag) if Y was pressed since last call."""
            with self._lock:
                if self._spawn_requested:
                    self._spawn_requested = False
                    return True
            return False

        def pop_next_test(self) -> bool:
            """Returns True (and clears the flag) if D-pad up was pressed since last call."""
            with self._lock:
                if self._next_test_requested:
                    self._next_test_requested = False
                    return True
            return False

        def reset(self):
            with self._lock:
                self._gripper_closed = False
                self._spawn_requested = False
                self._reset_requested = False
                self._next_test_requested = False
                self._lb_held = False
                self._rb_held = False

        def close(self):
            try: os.close(self._fd)
            except OSError: pass

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
    # Use XboxJoystick (direct /dev/input reader) instead of Se3Gamepad so the
    # Bluetooth Xbox controller works regardless of carb gamepad enumeration.
    gamepad = XboxJoystick(device=f"/dev/input/js{args.gamepad_idx}")
    if not gamepad._connected:
        print(f"  [XboxJoystick] js{args.gamepad_idx} not found — will connect automatically when available")

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
        # Disable the inner baseLink's rigid body so there is only one in the
        # hierarchy.  A scaled parent breaks PhysX unless the rigid body sits
        # on the same prim as the scale ops.
        _bl = stage.GetPrimAtPath(prim_path + "/baseLink")
        if _bl.IsValid() and _bl.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(_bl).CreateRigidBodyEnabledAttr(False)
        # physx_utils.setRigidBody adds RigidBodyAPI + PhysxRigidBodyAPI to the
        # outer xform and walks the subtree to apply convex-hull collision.
        import omni.physx.scripts.utils as physx_utils
        physx_utils.setRigidBody(xform.GetPrim(), "convexDecomposition", False)
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr().Set(0.3)
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
    _next_test_requested_kb = [False]

    def _on_keyboard(event, *_args, **_kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.C:
                _spawn_requested[0] = True
            elif event.input == carb.input.KeyboardInput.R:
                _reset_requested[0] = True
            elif event.input == carb.input.KeyboardInput.N:
                _next_test_requested_kb[0] = True
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

    # EE (wrist) camera — pose updated every sim step from wrist_3_link transform.
    _cam_ee_path = "/World/CamEE"
    _cam_ee = UsdGeom.Camera.Define(stage, _cam_ee_path)
    _cam_ee.GetFocalLengthAttr().Set(18.0)
    _cam_ee_xform = UsdGeom.Xformable(_cam_ee)
    _cam_ee_mat_op = _cam_ee_xform.MakeMatrixXform()
    _cam_ee_mat_op.Set(Gf.Matrix4d(1))  # identity until first step

    # Viewport windows stacked on the left — pass positions to the constructor
    # so the underlying setPosition() call fires correctly.
    if not args.headless:
        _vp_side = vp_util.create_viewport_window(
            "Side View", width=420, height=280, position_x=0, position_y=0
        )
        _vp_side.viewport_api.camera_path = _cam_side

        _vp_top = vp_util.create_viewport_window(
            "Top View", width=420, height=280, position_x=0, position_y=290
        )
        _vp_top.viewport_api.camera_path = _cam_top

        _vp_ee = vp_util.create_viewport_window(
            "EE View", width=420, height=280, position_x=0, position_y=580
        )
        _vp_ee.viewport_api.camera_path = _cam_ee_path

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

    # Base "tool pointing down" orientation [w=0, x=1, y=0, z=0] — 180° about X.
    # The right stick accumulates Rx/Ry deltas on top of this base each step.
    _BASE_ROT   = ScipyRot.from_quat([1.0, 0.0, 0.0, 0.0])  # scipy: [x,y,z,w]
    _acc_rot    = ScipyRot.identity()

    def _compute_target_quat() -> torch.Tensor:
        """Return CuRobo [w,x,y,z] quaternion from accumulated gripper orientation."""
        q = (_BASE_ROT * _acc_rot).as_quat()  # scipy gives [x,y,z,w]
        return torch.tensor([[q[3], q[0], q[1], q[2]]], device=device, dtype=torch.float32)

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

    # ── Validation test scene management ──────────────────────────────────────────
    test_scene_index = [0]  # 0 = free-play
    _test_objects: list[str] = []  # prim paths of spawned test objects (to clear)
    _ghost_prim = [None]
    _end_zone_prim = [None]
    _end_zone_highlight = [False]  # True when gripper is inside end-zone
    _hud_text = [""]

    _BLOCK_USD_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "asyncDualPlayPPO", "assets", "blocks"
    )
    _BLOCK_FILES_LIST = ["concave.usd", "cube.usd", "cylinder.usd", "rect.usd", "triangle.usd"]
    _T_BLOCK_USD = os.path.join(_BLOCK_USD_DIR, "t_shape.usda")
    _T_BLOCK_USD_BIN = os.path.join(_BLOCK_USD_DIR, "t_shape.usd")

    def _clear_test_objects():
        """Remove all spawned test objects from the stage."""
        nonlocal _test_objects, _ghost_prim, _end_zone_prim, _end_zone_highlight
        for path in _test_objects:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
        _test_objects = []
        if _ghost_prim[0] is not None and stage.GetPrimAtPath(_ghost_prim[0]).IsValid():
            stage.RemovePrim(_ghost_prim[0])
        _ghost_prim[0] = None
        if _end_zone_prim[0] is not None and stage.GetPrimAtPath(_end_zone_prim[0]).IsValid():
            stage.RemovePrim(_end_zone_prim[0])
        _end_zone_prim[0] = None
        _end_zone_highlight[0] = False

    def _spawn_goal_ghost(block_usd: str, x: float, y: float, yaw_deg: float):
        """Spawn a pink kinematic ghost at the goal pose (position + yaw)."""
        ghost_path = "/World/GoalGhost"
        _clear_test_objects()  # clears previous ghost too

        ghost_xform = UsdGeom.Xform.Define(stage, ghost_path)
        ghost_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.05))
        ghost_xform.AddRotateZOp().Set(float(yaw_deg))
        ghost_xform.AddScaleOp().Set(Gf.Vec3f(1.52, 1.52, 1.52))
        ghost_xform.GetPrim().GetReferences().AddReference(block_usd)

        # Pink emissive material
        _gmat = UsdShade.Material.Define(stage, ghost_path + "/GhostMat")
        _gshader = UsdShade.Shader.Define(stage, ghost_path + "/GhostMat/Shader")
        _gshader.CreateIdAttr("UsdPreviewSurface")
        _gshader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.4, 0.7))
        _gshader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.2, 0.4))
        _gshader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        _gmat.CreateSurfaceOutput().ConnectToSource(_gshader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(ghost_xform.GetPrim()).Bind(_gmat)

        _ghost_prim[0] = ghost_path

    def _spawn_end_zone(x: float, y: float):
        """Spawn a semi-transparent blue cube as the end-zone marker."""
        ez_path = "/World/EndZone"
        ez = UsdGeom.Cube.Define(stage, ez_path)
        ez.GetSizeAttr().Set(1.0)
        UsdGeom.Xformable(ez).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.02))
        UsdGeom.Xformable(ez).AddScaleOp().Set(Gf.Vec3f(0.08, 0.08, 0.04))

        _ezmat = UsdShade.Material.Define(stage, ez_path + "/EZMat")
        _ezshader = UsdShade.Shader.Define(stage, ez_path + "/EZMat/Shader")
        _ezshader.CreateIdAttr("UsdPreviewSurface")
        _ezshader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.6, 1.0))
        _ezshader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.4)
        _ezshader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        _ezmat.CreateSurfaceOutput().ConnectToSource(_ezshader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(ez.GetPrim()).Bind(_ezmat)

        _end_zone_prim[0] = ez_path

    def _spawn_block_at(path: str, usd_file: str, x: float, y: float, yaw_deg: float = 0.0, color=None):
        """Spawn a physics-enabled block at a given position."""
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.05))
        xform.AddRotateZOp().Set(float(yaw_deg))
        scale = random.uniform(0.8, 1.35)
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        xform.GetPrim().GetReferences().AddReference(usd_file)

        _bl = stage.GetPrimAtPath(path + "/baseLink")
        if _bl.IsValid() and _bl.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(_bl).CreateRigidBodyEnabledAttr(False)

        import omni.physx.scripts.utils as physx_utils
        physx_utils.setRigidBody(xform.GetPrim(), "convexDecomposition", False)
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr().Set(0.3)

        if color is None:
            r, g, b = random.random(), random.random(), random.random()
        else:
            r, g, b = color
        _bmat = UsdShade.Material.Define(stage, path + "/BlockMat")
        _bshader = UsdShade.Shader.Define(stage, path + "/BlockMat/Shader")
        _bshader.CreateIdAttr("UsdPreviewSurface")
        _bshader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        _bshader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        _bmat.CreateSurfaceOutput().ConnectToSource(_bshader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(xform.GetPrim()).Bind(_bmat)
        return path

    def _load_test_scene(index: int):
        """Load a validation test scene by index (0 = free-play, clears everything)."""
        nonlocal test_scene_index, _test_objects
        _clear_test_objects()

        if index == 0:
            _hud_text[0] = "Free-play mode"
            print(f"\n  [Test] Free-play mode")
            return

        cfg = get_test_config(index)
        if cfg is None:
            _hud_text[0] = "Free-play mode"
            return

        # Determine block USD
        if cfg.is_t_block:
            block_usd = _T_BLOCK_USD if os.path.exists(_T_BLOCK_USD) else _T_BLOCK_USD_BIN
            if not os.path.exists(block_usd):
                print(f"  [WARN] T-block USD not found at {block_usd}. Using rect.usd as fallback.")
                block_usd = os.path.join(_BLOCK_USD_DIR, "rect.usd")
        else:
            block_file = random.choice(_BLOCK_FILES_LIST)
            block_usd = os.path.join(_BLOCK_USD_DIR, block_file)

        # Spawn main block
        main_path = "/World/ValidationBlock"
        _spawn_block_at(
            main_path,
            block_usd,
            cfg.main_start.x,
            cfg.main_start.y,
            cfg.main_start.yaw_deg,
            color=(0.1, 0.8, 0.1),  # green for main block
        )
        _test_objects.append(main_path)

        # Spawn extra objects (distractors)
        for i, extra in enumerate(cfg.extra_starts):
            extra_file = random.choice([f for f in _BLOCK_FILES_LIST])
            extra_usd = os.path.join(_BLOCK_USD_DIR, extra_file)
            extra_path = f"/World/Distractor_{i}"
            distractor_colors = [
                (0.5, 0.5, 0.5),  # grey
                (0.6, 0.4, 0.2),  # brown
                (0.9, 0.5, 0.1),  # orange
                (0.6, 0.3, 0.7),  # purple
            ]
            c = distractor_colors[i % len(distractor_colors)]
            _spawn_block_at(
                extra_path, extra_usd, extra.x, extra.y, extra.yaw_deg, color=c
            )
            _test_objects.append(extra_path)

        # Spawn pink ghost at goal
        _spawn_goal_ghost(block_usd, cfg.main_goal_x, cfg.main_goal_y, cfg.main_goal_yaw_deg)

        # Spawn end-zone marker
        _spawn_end_zone(cfg.end_zone_x, cfg.end_zone_y)

        test_scene_index[0] = index
        label = get_test_label(index)
        _hud_text[0] = label
        print(f"\n  [Test] Loaded: {label}")
        print(f"    Main block start: ({cfg.main_start.x:.2f}, {cfg.main_start.y:.2f})")
        print(f"    Goal: ({cfg.main_goal_x:.2f}, {cfg.main_goal_y:.2f}) yaw={cfg.main_goal_yaw_deg}°")
        print(f"    Extra objects: {len(cfg.extra_starts)}")
        print(f"    End-zone: ({cfg.end_zone_x:.2f}, {cfg.end_zone_y:.2f})")

    def _check_end_zone() -> bool:
        """Check if the gripper midpoint is within the end-zone."""
        if _end_zone_prim[0] is None:
            return False
        if test_scene_index[0] == 0:
            return False  # no end-zone in free-play

        cfg = get_test_config(test_scene_index[0])
        if cfg is None:
            return False

        # gripper midpoint in world frame
        left_pos = robot.data.body_pos_w[0, left_finger_id]
        right_pos = robot.data.body_pos_w[0, right_finger_id]
        mid = (left_pos + right_pos) / 2.0

        ez_x, ez_y = cfg.end_zone_x, cfg.end_zone_y
        dx = mid[0].item() - ez_x
        dy = mid[1].item() - ez_y
        dist = (dx * dx + dy * dy) ** 0.5
        in_zone = dist < 0.06  # 6 cm radius from end-zone center

        # Visual feedback: change end-zone color when gripper is inside
        if in_zone != _end_zone_highlight[0]:
            _end_zone_highlight[0] = in_zone
            ez_prim = stage.GetPrimAtPath(_end_zone_prim[0])
            if ez_prim.IsValid():
                _ezmat = UsdShade.Material.Get(stage, _end_zone_prim[0] + "/EZMat")
                if _ezmat:
                    _ezshader = UsdShade.Shader.Get(stage, _end_zone_prim[0] + "/EZMat/Shader")
                    if _ezshader:
                        if in_zone:
                            _ezshader.GetInput("diffuseColor").Set(Gf.Vec3f(0.2, 1.0, 0.2))
                            _ezshader.GetInput("opacity").Set(1.0)
                        else:
                            _ezshader.GetInput("diffuseColor").Set(Gf.Vec3f(0.2, 0.6, 1.0))
                            _ezshader.GetInput("opacity").Set(0.4)

        return in_zone

    def _advance_test():
        """Cycle to the next test scene."""
        next_idx = (test_scene_index[0] + 1) % (get_test_count() + 1)
        _load_test_scene(next_idx)

    def _reset_test_scene():
        """Reload the current test scene from scratch."""
        _load_test_scene(test_scene_index[0])

    # Initialize HUD
    _hud_text[0] = "Free-play mode  |  DPAD_UP/N: next test  |  B/R: reset"

    # ── Print startup message ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [Validation Tests Mode Ready]")
    print("  DPAD_UP / N key  →  cycle to next validation test")
    print("  B button / R key →  reset current test")
    print("  Pressing N cycles through 50 test scenes + free-play")
    print(f"  Total tests: {get_test_count()} (Single, Multi-Object, T-Push)")
    print("=" * 70 + "\n")

    step_count = 0
    ik_ok_count = 0
    test_started = False
    end_zone_timer = 0

    try:
        while simulation_app.is_running():
            simulation_app.update()

            # ── Test scene cycling ──────────────────────────────────────────────
            if _next_test_requested_kb[0] or gamepad.pop_next_test():
                _next_test_requested_kb[0] = False
                _advance_test()

            # ── Reset / reload current test scene ───────────────────────────────
            if _reset_requested[0] or gamepad.pop_reset():
                _reset_requested[0] = False
                if test_scene_index[0] > 0:
                    _reset_test_scene()
                env.reset()
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
                _acc_rot = ScipyRot.identity()
                if test_scene_index[0] == 0:
                    print("  [R] Reset — robot and target returned to start pose.")
                else:
                    print(f"  [R] Test scene reset — {get_test_label(test_scene_index[0])}")

            if _spawn_requested[0] or gamepad.pop_spawn():
                _spawn_requested[0] = False
                _spawn_block()

            # Apply gamepad XYZ delta to ball's USD position.
            gamepad_cmd = gamepad.advance()
            delta_pos = gamepad_cmd[:3].cpu().numpy()
            delta_rot_vec = gamepad_cmd[3:6].cpu().numpy()
            if np.any(np.abs(delta_rot_vec) > 1e-6):
                _acc_rot = _acc_rot * ScipyRot.from_rotvec(delta_rot_vec)
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

            delta = ball_w - current_ik_target_w
            dist = delta.norm()
            if dist > max_step_delta:
                current_ik_target_w = current_ik_target_w + (delta / dist) * max_step_delta
            else:
                current_ik_target_w = ball_w.clone()

            local_target = current_ik_target_w - env_origin
            local_target[0] = local_target[0].clamp(_WS_X_MIN, _WS_X_MAX)
            local_target[1] = local_target[1].clamp(_WS_Y_MIN, _WS_Y_MAX)
            local_target[2] = local_target[2].clamp(_WS_Z_MIN, _WS_Z_MAX)
            current_ik_target_w = local_target + env_origin

            left_pos_w   = robot.data.body_pos_w[0, left_finger_id]
            right_pos_w  = robot.data.body_pos_w[0, right_finger_id]
            tcp_pos_w_cur = (left_pos_w + right_pos_w) / 2.0
            wrist3_pos_w  = robot.data.body_pos_w[0, wrist3_id]
            tcp_offset    = tcp_pos_w_cur - wrist3_pos_w

            ik_local_target = local_target - tcp_offset

            cur_joints = robot.data.joint_pos[:, joint_indices]
            goal_pose = Pose(
                position=ik_local_target.unsqueeze(0),
                quaternion=_compute_target_quat(),
            )
            ik_result = ik_solver.solve_single(
                goal_pose,
                seed_config=cur_joints.unsqueeze(1),
                retract_config=cur_joints,
            )

            if ik_result.success.any():
                ik_ok_count += 1
                last_good_joints = ik_result.solution.view(-1)[:6].clone()
                action[0, :6] = last_good_joints
            else:
                action[0, :6] = last_good_joints

            env.step(action)
            robot.update(env.step_dt)

            # ── End-zone check ──────────────────────────────────────────────────
            in_end_zone = _check_end_zone()
            if in_end_zone:
                end_zone_timer += 1
                if end_zone_timer == 1:
                    print(f"\n  [END-ZONE] Gripper reached end-zone! Test {get_test_label(test_scene_index[0])} complete.")
                    print(f"  [END-ZONE] Press N/DPAD_UP for next test or B/R to retry.\n")
            else:
                end_zone_timer = 0

            # ── Update EE camera pose ──────────────────────────────────────────
            _tcp      = tcp_pos_w_cur.cpu().numpy()
            _tool_rot = _BASE_ROT * _acc_rot
            _approach = _tool_rot.apply(np.array([0.0, 0.0,  1.0]))
            _cam_pos  = _tcp - _approach * 0.10
            _look_at  = _tcp + _approach * 0.30
            _cam_ee_mat_op.Set(_make_lookat_matrix(
                [float(v) for v in _cam_pos],
                [float(v) for v in _look_at],
            ))

            step_count += 1

            if step_count % 10 == 0:
                tcp_local = tcp_pos_w_cur - env_origin
                err = (tcp_local - local_target).norm().item()
                ok = ik_result.success.any().item()
                ez_str = "  [EZ:OK]" if in_end_zone else ""
                print(
                    f"[{step_count:5d}] {_hud_text[0]:40s} "
                    f"target=({local_target[0]:.3f},{local_target[1]:.3f},{local_target[2]:.3f}) "
                    f"tcp=({tcp_local[0]:.3f},{tcp_local[1]:.3f},{tcp_local[2]:.3f}) "
                    f"err={err:.4f}m  IK={'OK  ' if ok else 'FAIL'} "
                    f"({ik_ok_count}/{step_count}){ez_str}"
                )

    except KeyboardInterrupt:
        print("\nExiting.")

    simulation_app.close()


if __name__ == "__main__":
    main()
