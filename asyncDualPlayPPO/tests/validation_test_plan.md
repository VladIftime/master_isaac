# Validation Test Implementation Plan

## Overview

Create a validation test system integrated into `test_curobo_follow_target.py` that cycles through pre-defined push-test scenes via the Xbox D-pad up button. Each scene consists of: a robot, a table, objects to push, and a pink ghost target marking the goal pose (position + orientation).

Inspired by the **Real World Push-T Task** from Diffusion Policy (Chi et al., 2023):
> The robot must (1) precisely push a block into a target region, and (2) move the end-effector to an end-zone to terminate the episode.

---

## Test Categories

| # | Category | Objects | Tests | Description |
|---|----------|---------|-------|-------------|
| 1 | Single Push | 1 object (random shape) | 10 | Push one block to a goal pose |
| 2 | Multi-Object (2 extra) | 3 objects total | 10 | Push the main block with 2 distractors |
| 3 | Multi-Object (3 extra) | 4 objects total | 10 | Push the main block with 3 distractors |
| 4 | Multi-Object (4 extra) | 5 objects total | 10 | Push the main block with 4 distractors |
| 5 | T-Push | 1 T-shaped block | 10 | Diffusion Policy-style Push-T task |

**Total: 50 test scenes** (+ 1 free-play mode at index 0 = 51 states)

---

## Cycling Behavior (D-Pad Up)

Each press of D-pad **up** increments a `test_scene_index` (0..50, wraps around):

| Index | Mode |
|-------|------|
| 0 | **Free-play** — current behavior, no ghost, manual control |
| 1-10 | Single Push tests #1–#10 |
| 11-20 | Multi-Object (2 extra) tests #1–#10 |
| 21-30 | Multi-Object (3 extra) tests #1–#10 |
| 31-40 | Multi-Object (4 extra) tests #1–#10 |
| 41-50 | T-Push tests #1–#10 |

An HUD overlay (viewport text) shows: `Test N/50 — Single Push` etc.

---

## 1. Single Push Test

### Setup
- Pre-defined start position `S` and goal pose `G` (position + orientation) are hardcoded for each of the 10 tests
- A single randomly-selected block shape from `{cube, rect, cylinder, triangle, concave}` is spawned at `S`
- A pink kinematic ghost (the "goal_ghost") is placed at `G` (using the same `spawn_random_block` mechanism so it matches the object shape, scale 1.52x, diffuse_color=(1.0, 0.4, 0.7))
- 10 test configurations are spread across the workspace: top-left, top-right, bottom-left, bottom-right, center, and intermediate positions

### Execution
- User manually pushes the object until it aligns with the pink ghost
- User then moves the end-effector to the "end-zone" (a marked region, e.g. top-right corner)
- Pressing B resets the test

### Pre-defined Configs (example layout)

```
Test 1:  S=(-0.30, 0.70)  G=( 0.30, 0.60)
Test 2:  S=( 0.30, 0.60)  G=(-0.30, 0.70)
Test 3:  S=(-0.20, 0.50)  G=( 0.20, 0.75)
Test 4:  S=( 0.20, 0.75)  G=(-0.20, 0.50)
Test 5:  S=(-0.35, 0.65)  G=( 0.35, 0.55)
Test 6:  S=( 0.35, 0.55)  G=(-0.35, 0.65)
Test 7:  S=(-0.10, 0.55)  G=( 0.25, 0.70)
Test 8:  S=( 0.25, 0.70)  G=(-0.10, 0.55)
Test 9:  S=( 0.00, 0.50)  G=( 0.00, 0.70)
Test 10: S=( 0.00, 0.70)  G=( 0.00, 0.50)
```

All at Z=0.05m. Goal yaw rotations can be randomized (0, 90°, 180°, 270°) to test orientation alignment.

---

## 2. Multi-Object Tests

### Setup
- Main object (the one to push) and 2–4 distractor objects are spawned at fixed pre-defined positions
- Distractors use random block shapes different from the main object
- The pink ghost marks the goal for the **main object only**
- Distractors have distinct colors (e.g. grey, brown, orange, purple) to differentiate them
- 10 configs per extra-object count (2, 3, 4)

### Example layout (2 extra objects, Test 1)

```
Main:  S=(-0.20, 0.70)  G=( 0.25, 0.60)
Distractor 1: (-0.30, 0.55)
Distractor 2: ( 0.10, 0.72)
```

### Distractor behavior
- Distractors are dynamic rigid bodies with physics enabled
- They can be bumped/moved incidentally — the test is about pushing the main object to goal
- If a distractor falls off the table, it can be optionally ignored or the test can be reset

---

## 3. T-Push Test (Diffusion Policy Push-T)

### T-Shaped Block Asset

A new T-shaped block USD must be created:

```
asyncDualPlayPPO/assets/blocks/t_shape.usd
asyncDualPlayPPO/assets/blocks/configuration/t_shape_base.usd
asyncDualPlayPPO/assets/blocks/configuration/t_shape_physics.usd
```

**Geometry** (top-down view, dimensions in meters):
```
    0.12
  ┌──────┐
  │      │  0.04
  └──┬───┘
     │  0.10
     │
    0.04
```
- Top bar: 0.12m × 0.04m × 0.04m (L×W×H)
- Stem: 0.04m × 0.10m × 0.04m
- Total bounding box: ~0.12m × 0.14m × 0.04m
- Color: distinctive (e.g. bright red or orange)
- Mass: ~0.08 kg
- Friction: 0.5 (matching existing blocks)

Created as a combined convex hull collision mesh (or two-box compound collision) and a single visual mesh.

### Target Region

The goal is a **region** (not just a point) marked by the pink ghost T-block. The target region should be:
- A rectangular area drawn on the table surface (semi-transparent pink rectangle via UsdGeom.Cube)
- The pink ghost T-block is placed at the center of this region with the target orientation
- Region size: ~0.20m × 0.20m (generous enough to be achievable, tight enough to be meaningful)

### End-Zone

A visual-only end-zone marker is placed in a consistent location (e.g. top-right, X=0.05m, Y=0.30m):
- A small green/blue semi-transparent cube or sphere
- When the gripper midpoint enters the end-zone, the episode is complete
- Visual feedback: end-zone turns solid green when the gripper is inside

### Test Configurations

10 T-Push tests with varying S and G:

```
Test 1:  S=(-0.25, 0.70, 0°)  G=( 0.25, 0.55, 0°)
Test 2:  S=( 0.25, 0.60, 45°) G=(-0.25, 0.70, 90°)
Test 3:  S=(-0.30, 0.55, 90°) G=( 0.30, 0.65, 180°)
Test 4:  S=( 0.30, 0.70, 0°)  G=(-0.30, 0.55, -90°)
Test 5:  S=(-0.20, 0.50, 180°) G=( 0.20, 0.75, 45°)
Test 6:  S=( 0.20, 0.75, -45°) G=(-0.20, 0.50, 0°)
Test 7:  S=(-0.35, 0.65, 90°) G=( 0.35, 0.55, -90°)
Test 8:  S=( 0.35, 0.55, 0°)  G=(-0.35, 0.65, 180°)
Test 9:  S=( 0.00, 0.50, 0°)  G=( 0.00, 0.70, 0°)
Test 10: S=( 0.00, 0.70, 0°)  G=( 0.00, 0.50, 90°)
```

---

## Integration into `test_curobo_follow_target.py`

### New Data Structures

```python
@dataclass
class PushTestConfig:
    name: str                          # "Single Push", "Multi-Object", "T-Push"
    main_start: Tuple[float, ...]      # (x, y, z, yaw) for main object
    main_goal: Tuple[float, ...]       # (x, y, z, yaw) for goal pose
    extra_starts: List[Tuple[float, ...]]  # list of (x, y, z) for distractors
    block_type: Optional[str]          # specific shape or None = random
    end_zone_pos: Tuple[float, ...]    # (x, y, z) for end-zone marker

# Master list of 50 tests
VALIDATION_TESTS: List[PushTestConfig] = [ ... ]
```

### D-Pad Up Handler

In the `XboxJoystick._loop()` method, add detection for D-pad up events. On Xbox Bluetooth, D-pad is typically reported as **axis 6** (vertical) and **axis 7** (horizontal). A press of D-pad up = axis 6 value of +32767.

```python
# In XboxJoystick:
# D-pad reported as axis 6 (vertical: up=-1, down=+1) and axis 7 (horizontal)
# Or alternatively as HAT switch events.
# Detect rising edge: value transitions from 0 to -32767 for "up"

if number == 6 and value == -32767 and self._axes[6] not in (-32767,):
    self._next_test_requested = True
```

Or use keyboard **N** key as fallback.

### Scene Loading on Cycle

When `next_test` is requested:

1. Remove all previously spawned objects (blocks, ghosts, end-zone markers)
2. Read `PushTestConfig` for current index
3. Spawn the main object at `main_start` position with random or specified block type
4. Spawn extra objects at `extra_starts` positions (for multi-object tests)
5. Spawn pink ghost at `main_goal` position/orientation (kinematic, no collision, scale 1.52x)
6. Spawn end-zone marker (semi-transparent cube)
7. Update viewport text overlay with test info

### Goal Ghost Implementation

Following the pattern from `wrapper.py`, the ghost should be:
```python
# Pink ghost at goal pose
ghost_cfg = sim_utils.UsdFileCfg(
    usd_path=f".../blocks/{block_file}",
    scale=(1.52, 1.52, 1.52),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.7)),
)
```

Since we're in single-env mode with direct USD manipulation (as in `test_curobo_follow_target.py`), the ghost is created via `UsdGeom` with a pink emissive material:

```python
# Alternative direct USD approach (matching test_curobo_follow_target.py style)
ghost_prim = UsdGeom.Xform.Define(stage, path)
# Add reference to block USD
ghost_prim.GetPrim().GetReferences().AddReference(block_usd_path)
# Scale up slightly
ghost_prim.AddScaleOp().Set(Gf.Vec3f(1.52, 1.52, 1.52))
# Pink material
mat = UsdShade.Material.Define(stage, path + "/GhostMat")
shader = UsdShade.Shader.Define(stage, path + "/GhostMat/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.4, 0.7))
shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.2, 0.4))
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(ghost_prim.GetPrim()).Bind(mat)
```

### End-Zone Marker

```python
ez_prim = UsdGeom.Cube.Define(stage, "/World/EndZone")
ez_prim.GetSizeAttr().Set(0.08)
ez_mat = UsdShade.Material.Define(stage, "/World/EndZone/Mat")
ez_shader = UsdShade.Shader.Define(stage, "/World/EndZone/Mat/Shader")
ez_shader.CreateIdAttr("UsdPreviewSurface")
ez_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.6, 1.0))
ez_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.4)
ez_mat.CreateSurfaceOutput().ConnectToSource(ez_shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(ez_prim.GetPrim()).Bind(ez_mat)
# When gripper is inside: change opacity to 1.0, color to solid green
```

### HUD / Viewport Text

Use `omni.kit.widget.text` or `carb.gfx` to overlay text showing:
```
Test  3/50  |  Multi-Object (2 extra)  |  Press B to reset
```

---

## Implementation Steps

### Step 1: Create T-shaped block USD
- Create `asyncDualPlayPPO/assets/blocks/t_shape.usd` with a T-shaped collision and visual mesh
- Or as a compound of two boxes in a single USD with convex decomposition
- Register in `BLOCK_FILES` list and `spawn_random_block` logic

### Step 2: Define test configurations
- Create a new file `asyncDualPlayPPO/tests/validation_configs.py` with `PushTestConfig` dataclass and all 50 test configs
- Include `get_test_config(index: int) -> PushTestConfig` and `get_test_count() -> int`

### Step 3: Modify `test_curobo_follow_target.py`
- Import `ValidationTestManager` from the new config module
- Add DPAD_UP / keyboard N detection to `XboxJoystick._loop()`
- Add `pop_next_test()` method
- Implement `_load_test_scene(index)` function:
  - Remove all spawned entities (blocks, ghosts, end-zone)
  - Place objects per test config
  - Create pink ghost at goal
  - Create end-zone marker
- Implement `_check_end_zone()` in main loop:
  - When gripper midpoint is within end-zone → flash success, auto-reset or advance
- Display overlay text

### Step 4: Add logging/recording
- For each completed test, log: test_index, success (block at goal + gripper in end-zone), time taken, final position error
- Optionally save screenshots

### Step 5: Verify and iterate
- Load each test scene and verify visual correctness
- Test cycling forward and backward through all 51 states (0 free-play + 50 tests)

---

## Key Design Decisions

1. **Test definition format**: All configs hardcoded (not randomized) for reproducibility. Each test is deterministic and repeatable.

2. **Distractor spawning**: Extra objects use `physx_utils.setRigidBody` (same pattern as existing `_spawn_block()` in `test_curobo_follow_target.py`) to enable physics. They are spawned at precise positions per test config.

3. **Ghost matching**: The pink ghost uses the same block USD reference as the main object, scaled 1.52x, with pink diffuse material and collision disabled. This visually indicates the exact target pose.

4. **End-zone detection**: At each step, compute the gripper midpoint position. If it falls within the end-zone region (e.g., a 0.08m radius sphere), the episode terminates. The end-zone marker changes color for visual feedback.

5. **Keyboard fallback**: The **N** key (Next) provides DPAD-up functionality when no controller is connected.

6. **Reset behavior**: B button fully resets the current test scene to its initial state, moving the robot to home pose and respawning objects at their start positions.
