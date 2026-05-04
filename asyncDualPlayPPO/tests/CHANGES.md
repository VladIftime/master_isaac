# Changes — CuRobo IK Test Files

Covers all modifications made after commit `d2a2a6af` ("working ball following") through
the current working tree (commits `732c2476`, `c2176175`, and unstaged edits).

---

## `test_curobo_follow_target.py`

*(committed in `732c2476`)*

### TCP offset — gripper centre tracking

**Before:** The interactive sphere spawned at `wrist_3_link` (inside the arm body). The IK
target was passed directly from the ball position with no offset, so CuRobo placed
`tool0` (= `wrist_3_link`) at the ball — the gripper overshoot the ball entirely.

**After:**

1. **TCP body lookup** — `robot.find_bodies(["grasp_convenient_link"])` is called after
   reset (fallback to `robotiq_arg2f_base_link`). This is the grasp centre between the
   Robotiq 140 fingers (~0.225 m from the base link along tool Z).

2. **Sphere spawns at TCP** — `sphere_cfg.func(target_path, sphere_cfg, translation=tcp_pos_w)`
   uses the TCP world position, so the ball appears between the fingers at startup.

3. **Dynamic TCP offset in the IK loop** — each step computes:
   ```python
   tcp_offset_w = robot.data.body_pos_w[0, tcp_id] - robot.data.body_pos_w[0, wrist3_id]
   ik_local_target = local_target - tcp_offset_w
   ```
   CuRobo receives the adjusted target so `wrist_3_link` lands at a position that puts
   the TCP (grasp centre) exactly on the sphere.

4. **Error reporting** now prints TCP position vs target instead of `wrist_3_link`.

### Scene cleanup

- `env_cfg.scene.cube = None` and `env_cfg.scene.target_object = None` strip the
  manipulation objects from the environment so they don't appear as "TargetCube" / extra
  clutter in the viewport.

### Action override

- `JointPositionActionCfg(use_default_offset=False)` sends raw CuRobo joint angles
  directly to the ImplicitActuator PD controller.

---

## `standalone_curobo_ik_framework.py`

### TCP offset — tool-frame approach  *(committed in `732c2476`)*

**Before:** A world-Z offset (`GRIPPER_TCP_Z_EXTRA`) was applied to the EE position before
calling `setup_tcp_offset`. This broke when the wrist rotated because world Z and tool Z
diverged.

**After:** The extra reach is added **inside `setup_tcp_offset`** along the **tool's local Z
axis** after converting to the local frame:

```python
FINGER_REACH = 0.15  # metres from robotiq_arg2f_base_link to grasp centre

offset_local = _quat_inv_rotate(ee_pos - tool0_pos, tool0_quat)
offset_local[2] += self.FINGER_REACH   # always along tool Z, regardless of wrist angle
self._tcp_offset_local = offset_local
```

The `_quat_rotate` helper was also updated from the deprecated `torch.cross` to
`torch.linalg.cross`.

### Ball spawn position  *(committed in `732c2476`)*

Changed from the previous ad-hoc value to `[0.0, 0.50, 0.15]`, centred in the
working environment (robot at origin, table at Y = 0.5, derived from
`cfg/task/AsyncDualPlay.yaml`).

---

### Scene setup matching `AsyncDualPlayEnvCfg`  *(committed in `c2176175`)*

`FollowTarget.set_up_scene` now calls `_add_env_scene()` which populates the stage to
match the visual layout of `test_abc_goal_encoder.py`:

| Element | Details |
|---------|---------|
| **Lights removed** | `SphereLight` / `DistantLight` under `/World/defaultGroundPlane` deleted at scene-build time for render performance |
| **Table** | `VisualCuboid` 2.0 × 2.0 × 0.1 m, dark gray `(0.2, 0.2, 0.2)`, at `[0, 0.5, -0.05]` |
| **Ground zone borders** | 4 thin black `VisualCuboid` bars matching `AsyncDualPlayEnvCfg` positions (top Y=1.0, bottom Y=0.2, left X=-0.75, right X=0.75) |
| **Blocks** | `concave.usd`, `cube.usd`, `cylinder.usd`, `rect.usd`, `triangle.usd` loaded from `asyncDualPlayPPO/assets/blocks/` via `add_reference_to_stage` at env-cfg positions |

Asset root resolved as:
```python
_PROJ_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # master_isaac/
_BLOCKS_DIR = _PROJ_ROOT / asyncDualPlayPPO/assets/blocks
```

---

### Workspace ceiling border  *(unstaged)*

Four additional `VisualCuboid` bars added at `Z = 0.70` forming a rectangular frame at
the top of the robot's reachable workspace:

```
ZoneCeilTop    [0.0,   1.0, 0.70]  scale [1.52, 0.02, 0.01]
ZoneCeilBottom [0.0,   0.2, 0.70]  scale [1.52, 0.02, 0.01]
ZoneCeilLeft   [-0.75, 0.6, 0.70]  scale [0.02, 0.82, 0.01]
ZoneCeilRight  [0.75,  0.6, 0.70]  scale [0.02, 0.82, 0.01]
```

Mirrors the ground-level zone borders and gives a clear visual boundary for the IK
solver's reachable volume.

### Physics-enabled target object  *(unstaged)*

`TargetObject` changed from a pure-visual USD ref (`concave.usd`) to a **`DynamicCuboid`**
(4 cm red cube). The block USD files contain no built-in physics API, so `DynamicCuboid`
is used as the World-API equivalent of `RigidObjectCfg` in `AsyncDualPlayEnvCfg`:

```python
DynamicCuboid(
    prim_path="/World/TargetObject", name="target_object",
    position=np.array([0.0, 0.7, 0.05]),
    size=0.04,
    color=np.array([1.0, 0.2, 0.2]),   # red — matches env cfg diffuse_color
)
```

The remaining four blocks (cube, cylinder, rect, triangle) stay as pure-visual USD refs.

### Non-selectable scene elements  *(unstaged)*

`_lock_non_target_prims(target_prim_path)` is called once after `world.reset()`. It
traverses the full USD stage and calls `omni.kit.commands.execute("LockPrims", ...)` on
every prim **except** `/`, `/World`, and the red target sphere. Locked prims cannot be
clicked or moved with the viewport gizmo.

### Z ceiling clamp  *(unstaged)*

The target sphere's USD `TranslateOp` is cached at startup:

```python
_xlate_ops = [op for op in UsdGeom.Xformable(target_prim).GetOrderedXformOps()
              if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
_Z_CEIL = 0.70
```

Each simulation step, if the dragged position exceeds the ceiling:

```python
t = _xlate_ops[0].Get()
if t[2] > _Z_CEIL:
    _xlate_ops[0].Set(Gf.Vec3d(t[0], t[1], _Z_CEIL))
```

Writing to the `TranslateOp` directly (same layer as the viewport gizmo) ensures the
sphere snaps back to the ceiling plane on the same frame, with no one-frame lag.
X and Y are **not** clamped — the IK solver naturally fails outside the robot's reach,
holding the last valid pose.
