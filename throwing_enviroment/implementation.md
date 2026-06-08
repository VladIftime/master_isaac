# Throwing Environment — Implementation

Single-arm object throwing simulation in Isaac Lab / Isaac Sim. One dual-arm UR5e
robot system stands on a platform and uses a configurable IK solver to control a
single arm for throwing a drink bottle toward a shopping basket target on a table.

The design mirrors the pingpong_dual_arm environment and is directly inspired by
the Gazebo throwing implementation in `gazebo_impl/`. The action space, IK solver
integration, and RL training pipeline all follow the pingpong patterns.

## Project Structure

```
throwing_enviroment/
├── source/Throwing/                       # Gym environment registration package
│   ├── setup.py
│   ├── config/extension.toml
│   └── Throwing/
│       ├── __init__.py
│       └── tasks/
│           ├── __init__.py                # import_packages() for task discovery
│           └── throwing/
│               ├── __init__.py            # gym.register("Throwing-Direct-v0")
│               └── agents/
│                   └── skrl_ppo_cfg.yaml  # PPO agent configuration
├── tasks/                                 # Environment implementation
│   ├── __init__.py
│   ├── throwing_env_cfg.py                # Scene + MDP config
│   ├── throwing_env.py                    # Environment class (attachment, reward logic)
│   ├── observations.py                    # Observation functions (joints, EE, objects)
│   ├── rewards.py                         # Reward functions (distance, success, velocity)
│   ├── events.py                          # Reset events (robot reset, randomize target, attach)
│   └── terminations.py                    # Boundary + settled checks
├── scripts/
│   ├── skrl/
│   │   ├── train.py                       # Training launcher (skrl PPO)
│   │   └── play.py                        # Inference / playback
│   ├── test_env.py                        # Launch & step environment
│   ├── test_ik_throwing.py                # Multi-phase pick-and-throw IK benchmark (approach → grasp → lift → wrist-snap throw, --sweep for SNAP_RAD tuning)
│   ├── test_throw.py                      # Single-throw test (kinematic hold → release → land, --loop flag)
│   ├── convert_meshes.py                  # OBJ → USD with MeshConverter
│   ├── prebake_physics.py                 # Apply CollisionAPI + PhysicsMaterial to USD meshes
│   ├── prebake_drink.py                   # Pre-bake drink001: MassAPI, CollisionAPI, high friction
│   ├── prebake_basket.py                  # Pre-bake shopping basket: single rigid body, handles removed
│   ├── export_full_scene.py               # Export scene USD with FK-baked crane-pose transforms (--pose flag)
│   ├── inspect_usd.py                     # Inspect USD file prim hierarchy
│   └── inspect_usd_summary.py             # Inspect physics schemas + mesh extents
├── assets/
│   ├── milk/                              # Original Gazebo assets (legacy)
│   │   ├── model.config, model.sdf
│   │   └── meshes/model.dae, *.usd
│   ├── wooden_box/                        # Original Gazebo assets (legacy)
│   │   ├── model.config, model.sdf
│   │   └── meshes/model.dae, *.usd
│   ├── obstacle_box/                      # Original Gazebo assets (legacy)
│   │   ├── model.config, model.sdf
│   │   └── meshes/juice_box_pink.dae
│   └── new_usds/                          # Synthesis-Assets-Explorer USD imports
│       ├── drink001/
│       │   ├── model_drink001.usd          # Original Synthesis USD
│       │   └── drink_target.usd            # Pre-baked: MassAPI + CollisionAPI + high friction
│       ├── shopping basket002/
│       │   ├── model_basket_22.usd         # Original Synthesis USD (3 bodies, 2 joints)
│       │   └── basket_target.usd           # Pre-baked: single rigid body, handles removed
│       └── trash can002/                  # Articulated trash can (not used — basket replaced it)
├── meshes/                                # Intermediate mesh files (legacy)
├── generated_usd/                         # Scene exports + legacy converter output
│   ├── full_scene.usd                     # Reference-based full scene (robot FK-baked + drink + basket + table + stand)
│   ├── joint_pose.json                    # 24 joint angles for crane home pose
│   ├── body_poses.json                    # 25 body world positions/orientations (FK debug)
│   ├── pose_robot.py                      # Kit script: sets drive targets + auto-plays simulation (optional — FK baking supersedes this)
│   ├── robot_crane_pose/                  # Robot USD cache copy with FK-baked link + joint transforms
│   ├── milk.usd, wooden_box.usd           # Legacy MeshConverter output
│   └── config.yaml
├── cfg/
│   └── task/
│       └── throwing.yaml                  # Task hyperparameters
├── gazebo_impl/                           # Reference Gazebo implementation
│   ├── behaviour_change.cpp
│   ├── primitive_design.cpp
│   └── RL_tossing_object_with_obstacle_avoidance_v3.py
└── implementation.md
```

## Scene Layout

```
                         +Y →
       +---[BODY]---+                         z
       | left  right |                         ↑
       | arm   arm   |                         |
       | [BODY]      |                         |
       | [Stand]     |
       +---[BODY]---+
       Robot (Origin, on stand)           
           |
           |  [Drink]  (attached to right gripper, offset between fingers)
           |
           |  ====[Table]====  (y=1.0, z=0.6, kinematic, 1.0×1.2 m)
           |
           |       [Basket]    (x=±0.45, y=0.7–1.3, on table)
           |
```

- **Robot position**: global constant `ROBOT_POS = (0, 0, 0.6)` — body base at z=0.6 atop a kinematic stand
- **Stand**: Box cuboid (0.5×0.5×0.6 m) beneath the robot, kinematic, centered at half height (z=0.3)
- **Table**: Box cuboid (1.0×1.2×0.05 m) at `(0, 1.0, 0.575)`, kinematic, surface at z=0.6. Same height as the robot stand — the robot and table share a common work surface.
- **Drink object**: Dynamic rigid body, mass 0.5 kg. Visual mesh loaded via `UsdFileCfg` from `assets/new_usds/drink001/drink_target.usd` (Synthesis drink bottle with pre-baked MassAPI, CollisionAPI, and high-friction PhysxMaterial). In RL training mode, spawned at `right_wrist_3_link` at episode reset and kinematically attached via `write_root_pose_to_sim`. In the standalone IK benchmark (`test_ik_throwing.py`), spawned on the table at a fixed position `(0.40, 0.40, 0.72)` (settles to z≈0.60) and interacts purely through physics — no kinematic attachment. The EE targets the drink root position directly (zero offset).
  Bottle root offset `(-0.012, 0.129, -0.176)` in EE-local frame is used only for RL kinematic attachment (Option A).
- **Target (shopping basket)**: Kinematic shopping basket, randomized in XY on the table surface. Visual mesh loaded via `UsdFileCfg` from `assets/new_usds/shopping basket002/basket_target.usd` (Synthesis basket with handles removed, single rigid body). Mass 2.0 kg. Size ~0.52×0.38×0.26 m.
- **Arm initial pose**: Right arm starts at a home pose (`shoulder_lift=-1.57, elbow=1.57, wrist_1=-1.57, wrist_2=-1.57, wrist_3=0.0`). Left arm is idle. Gripper starts **open** (`finger_joint=0.0`).

## Running

```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
cd /home/vladi/IsaacLab/master_isaac/throwing_enviroment

# Test environment (verify scene loads)
python scripts/test_env.py --ik diffik --num_envs 4

# Export full scene for Isaac Sim GUI (default crane pose)
python scripts/export_full_scene.py

# Export with custom joint positions
python scripts/export_full_scene.py --pose '{"right_elbow_joint": 0.0, "right_wrist_2_joint": 0.0}'

# Export with a JSON pose file
python scripts/export_full_scene.py --pose my_pose.json

# Single throw test (visual mode)
python scripts/test_throw.py --ik diffik --amp 0.3

# Single throw (headless)
python scripts/test_throw.py --ik diffik --headless --amp 0.3

# Indefinite throws (Ctrl+C to stop)
python scripts/test_throw.py --ik diffik --loop

# Multi-phase pick-and-throw IK benchmark
python scripts/test_ik_throwing.py --ik diffik
python scripts/test_ik_throwing.py --compare diffik:osc:rmpflow:curobo --output metrics.csv

# Sweep SNAP_RAD to find best throw distance
python scripts/test_ik_throwing.py --ik diffik --sweep "1.0:8.0:0.5"

# Headless training
python scripts/skrl/train.py --task=Throwing-Direct-v0 --headless --num_envs=1024

# With specific IK solver
python scripts/skrl/train.py --task=Throwing-Direct-v0 --headless --ik_solver=osc --num_envs=512

# Playback
python scripts/skrl/play.py --task=Throwing-Direct-v0 --num_envs=1 --checkpoint=path/to/agent.pt

# Resume from checkpoint
python scripts/skrl/train.py --task=Throwing-Direct-v0 --headless --checkpoint=logs/skrl/throwing/.../agent_5000.pt
```

## Object Attachment — Three Options

The primary challenge is reliably holding the object during the throwing motion
before releasing it. Three approaches were considered, in order of preference:

### Option A: Kinematic Pose Write (Current Implementation)

At each physics step while `_holding[env_ids] == True`, the object's root pose is
overwritten to match the end-effector's world pose with a fixed offset placing the
bottle center between the finger pads. The offset `(-0.012, 0.129, -0.176)` is
defined in the **EE's local frame** and rotated into world coordinates via
`quat_rotate(ee_quat, offset_local)` so it stays correct regardless of arm orientation:

```python
bottle_offset_local = torch.tensor([-0.012, 0.129, -0.176], device=ee_pos.device)
bottle_offset_world = quat_rotate(ee_quat[still_holding], bottle_offset_local)
bottle_root = ee_pos[still_holding] + bottle_offset_world
ee_pose = torch.cat([bottle_root, ee_quat[still_holding]], dim=-1)
milk.write_root_pose_to_sim(ee_pose, env_ids=still_ids)
```

**Disabling attachment**: Setting `cfg.disable_attachment = True` causes
`_update_attachment()` to return immediately, giving the test script full
manual control over the object and gripper. The `test_ik_throwing.py` benchmark
uses this mode for pure physics-based pick-and-throw.

**Release condition — two modes**, selected by `cfg.release_at_step`:

1. **Step-count mode** (`release_at_step > 0`): Object is released at the
   exact specified step, regardless of velocity:
   ```python
   release_mask = (
       self._holding & ~self._released
       & (self._steps_in_episode >= self.cfg.release_at_step)
   )
   ```

2. **Velocity mode** (`release_at_step = 0`, default for RL): After
   `release_min_steps` warmup steps (default 10) AND when EE linear velocity
   magnitude exceeds `release_vel_threshold` (default 2.0 m/s):
   ```python
   release_mask = (
       self._holding & ~self._released
       & (self._steps_in_episode > self.cfg.release_min_steps)
       & (vel_norm > self.cfg.release_vel_threshold)
   )
   ```

On release, the gripper opens (all 6 revolute joints → 0.0) for visual feedback.

**Pros**: Deterministic, works at any speed, object velocity is always smooth,
no physics tuning required. **Cons**: Not "true" physics grasping — no contact
forces between gripper and object.

### Option B: Physics Gripper Actuation (Not Used)

Close gripper (`finger_joint = 0.7`, stiffness=5000) at reset with the object
spawned between the pads. The object is held purely by contact friction.
Open gripper at release time.

Attempted but failed: the bottle fell through the finger pads. The finger pad
collision geometry is too thin (7.5mm box primitives) and gets merged through
the URDF importer's link-merging pass. Physics contact between the thin pad
surfaces and the bottle mesh was unreliable. Even with high friction
(`static_friction=5.0, dynamic_friction=5.0`) baked into the bottle's
PhysxMaterial, the object slipped through under gravity.

### Option C: PhysX Fixed Joint Constraint (Last Resort)

Create a physics fixed joint between the gripper pad and object root at reset,
delete it at release. Equivalent to Gazebo's `graspObjectInGazebo()`. Requires
raw `omni.physx` interface calls. Use this only if both A and B fail for
acceptable throwing dynamics.

## Throwing Logic

### Object Release

The environment tracks three boolean states per environment:

| Tensor | Type | Description |
|--------|------|-------------|
| `_holding` | Bool (N,) | True: object follows EE via pose write; False: object is free |
| `_released` | Bool (N,) | Latched: object has been released this episode |
| `_object_landed` | Bool (N,) | Latched: object has landed within 0.15m of target |

The release is triggered automatically by EE velocity exceeding the threshold
after the minimum steps. The RL agent controls the EE trajectory via 6-D deltas
— the speed at which it moves the arm determines when release occurs.

### Reward Structure

| Term | Weight | Description |
|------|--------|-------------|
| `dist_reward` | 1.0 | Gaussian: `exp(-dist² / 0.1)` — continuous distance-based reward |
| `success_bonus` | 2.0 | One-time bonus when object lands within 0.15m of target |
| `ee_velocity_reward` | 0.5 | Proportional to EE velocity while holding (encourages throwing speed) |

All reward weights are centralized in `RewardsCfg`. Precomputed tensors in
`_compute_rewards()` are raw (unscaled) — the config weight multiplies them
to produce the final per-step reward.

### Reward Tuning Notes

The Gaussian distance reward `exp(-dist² / 0.1)` produces:
- `dist=0.0` → reward ≈ 1.0
- `dist=0.15` → reward ≈ 0.80
- `dist=0.5` → reward ≈ 0.08
- `dist=1.0` → reward ≈ 0.00005

The `ee_velocity_reward` incentivizes the agent to move the arm fast during the
throwing phase, which naturally leads to object release via the velocity threshold
mechanism.

### Termination

Episode ends (5s max) on:
- **Time limit**: 5 seconds (~600 steps at 120 Hz)
- **Object out of bounds**: x=±3.0, y=±3.0, or z<−0.2
- **Object settled**: Velocity < 0.05 m/s for 30 consecutive steps after release

## Environment Configuration (`tasks/throwing_env_cfg.py`)

### Robot Articulation

Robot spawns from `pingpong_dual_arm/urdf/dual_arm_robot.urdf` via `UrdfFileCfg`
with runtime URDF→USD conversion. The URDF includes:
- **Body**: `IRL_lab_robot_body.obj` + `robot_body_back.obj`
- **Head**: `blue_head_with_headphone.obj`
- **Two UR5e arms**: 6-DOF each (shoulder_pan/lift, elbow, wrist_1/2/3)
- **Two Robotiq 2F-140 grippers**: Left and right, attached to each arm's `tool0` frame

Global position constants:
```python
STAND_Z = 0.6          # robot base height
ROBOT_POS = (0, 0, STAND_Z)  # Robot at origin on stand

TABLE_Z = STAND_Z                  # table surface height (0.6)
TABLE_CENTER_POS = (0.0, 1.0, TABLE_Z)  # table center forward of robot
TABLE_SIZE = (1.0, 1.2, 0.05)      # tabletop dimensions (x, y, z)
```

Robot body base positioned at z=0.6 on top of a 0.5×0.5×0.6 m kinematic stand
cuboid. Stand is centered at half height (z=0.3), extending from z=0 to z=0.6.

Actuators: position-controlled implicit (stiffness=8000, damping=500).
Four groups: `arm_left`, `arm_right`, `gripper_left`, `gripper_right`.
Gripper actuators cover all 6 revolute joints per side (main `finger_joint`
+ 5 mimic joints) to explicitly handle PhysX mimic constraint propagation.

### Arm Initial Joint Positions (Home Pose)

```
Right arm (throwing, default):
  shoulder_pan:   0.0
  shoulder_lift: -1.57
  elbow:          1.57
  wrist_1:       -1.57
  wrist_2:       -1.57
  wrist_3:        0.0

Left arm (idle):
  shoulder_pan:   0.0
  shoulder_lift: -1.57
  elbow:         -1.57
  wrist_1:       -1.57
  wrist_2:        1.57
  wrist_3:        0.0

Grippers: finger_joint = 0.0 (open) at reset. Gripper actuator covers all
6 revolute joints per side (`rgripper_finger_joint`, `rgripper_.*_knuckle_joint$`,
`rgripper_.*_inner_finger_joint$`) to handle PhysX mimic joints explicitly.
```

### Physics

```
sim.dt         = 1/120 s (~0.00833 s)
decimation     = 1 (120 Hz control)
episode_length = 5 s (600 steps)
physics_material = friction 1.0, restitution 0.3, combine_mode "max"

GPU buffers:
  gpu_found_lost_pairs_capacity = 1M
  gpu_max_rigid_contact_count   = 1M
  gpu_max_rigid_patch_count     = 327K
```

### Actions

```
Action space: Box (6,) per env
  arm (0:6):   [dx, dy, dz, droll, dpitch, dyaw]  — throwing arm EE delta
```

All IK solvers use the same 6-D **relative delta** format. The delta is
accumulated onto the current end-effector pose by each solver's controller:

| Solver | Accumulation | Description |
|--------|-------------|-------------|
| `diffik`, `osc`, `rmpflow` | `use_relative_mode=True` with `scale=0.15` | Isaac Lab internally accumulates deltas |
| `curobo` | Manual accumulation in `apply_actions()` | `target_world = ee_pos_w + delta`, then frame-converted to arm base frame |

For `diffik`/`osc`/`rmpflow`, scale=0.15 (default, overridable). For
`curobo`, no scaling is applied — the raw delta is used directly.

### Observations

```
Observation space: Box (27,) per env
  joint_pos      (0:6):    Throwing arm 6 joint positions
  joint_vel      (6:12):   Throwing arm 6 joint velocities
  ee_pose        (12:18):  End-effector pos(3) + Euler ZYX(3)
  object_pos     (18:21):  Drink position in env-local coordinates
  target_pos     (21:24):  Basket position in env-local coordinates
  dist_to_target (24:27):  Vector from object to target (env-local)
```

Positions in environment-local coordinates. Rotations as ZYX Euler angles.

### Events

| Event | Mode | Function |
|-------|------|----------|
| `reset_all` | reset | `mdp.reset_scene_to_default` |
| `reset_robot` | reset | `reset_robot_joints` (robot to defaults, gripper open) |
| `randomize_target` | reset | `randomize_target_position` (target XY random on table surface) |
| `attach_object` | reset | `attach_milk_to_gripper` (spawn bottle between fingers, close gripper) |

### Target Randomization

Target position randomized on each reset, on the table surface:
- `target_x`: Uniform `[-0.45, 0.45]` m
- `target_y`: Uniform `[0.7, 1.3]` m
- `target_z`: Fixed at `TABLE_Z + 0.1` (on table surface)

### Terminations

| Term | Type | Function |
|------|------|----------|
| `time_limit` | truncated | `mdp.time_out` (5 s) |
| `object_out_of_bounds` | terminated | Object flies outside x=±3, y=±3, or z<−0.2 |
| `object_settled` | terminated | Object velocity < 0.05 m/s for 30 consecutive steps after release |

## ThrowingEnv Class (`tasks/throwing_env.py`)

Extends `ManagerBasedRLEnv`. Key additions:

### `step(action)` override

```python
def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    self._update_attachment()
    self._compute_rewards()
    return obs, reward, terminated, truncated, info
```

### Object Attachment (`_update_attachment`)

1. Reads EE world pose, velocity from `_ee_body` (e.g. `right_wrist_3_link`)
2. While `_holding` and not `_released`: writes object root pose to EE position plus a fixed bottle offset `(-0.012, 0.129, -0.176)` placing the bottle center between the gripper fingers
3. When `_steps > release_min_steps` AND EE velocity > threshold: releases object, opens gripper

### Reward Computation (`_compute_rewards`)

Computes three precomputed reward tensors from physics state:
- `_dist_reward`: Gaussian `exp(-dist² / 0.1)` from object to target
- `_success_bonus`: One-time +1 if dist < 0.15 after release
- `_ee_vel_reward`: EE velocity norm scaled by `_holding` flag

### Game State Tensors

| Tensor | Type | Description |
|--------|------|-------------|
| `_holding` | Bool (N,) | Object is attached to EE this episode |
| `_released` | Bool (N,) | Latched: object released this episode |
| `_object_settled_count` | Int (N,) | Consecutive low-velocity steps after release |
| `_steps_in_episode` | Int (N,) | Steps elapsed in current episode |
| `_object_landed` | Bool (N,) | Latched: object landed within 0.15m |
| `_dist_reward` | Float (N,) | Precomputed distance reward (unscaled) |
| `_success_bonus` | Float (N,) | Precomputed success bonus (unscaled) |
| `_ee_vel_reward` | Float (N,) | Precomputed velocity reward (unscaled) |

## Observations Module

| Function | Dims | Description |
|----------|------|-------------|
| `robot_joint_positions` | (N, 6) | Throwing arm joint positions |
| `robot_joint_velocities` | (N, 6) | Throwing arm joint velocities |
| `ee_pose` | (N, 6) | End-effector pos(3) + Euler ZYX(3) |
| `object_position` | (N, 3) | Drink position in env-local coordinates |
| `dist_to_target` | (N, 3) | Vector from object (drink) to target |

## IK Solvers (`ik_solvers/`)

IK solvers are imported directly from `pingpong_dual_arm/ik_solvers/` via
`sys.path` — no code duplication. The `build_ik_action()` dispatcher supports
all four solver types with the same 6-D relative delta action format.

| Solver | Isaac Lab Class | Key Param |
|--------|----------------|-----------|
| `diffik` | `DifferentialInverseKinematicsActionCfg` | `lambda_val=0.1` |
| `osc` | `OperationalSpaceControllerActionCfg` | `stiffness=[360]×6` |
| `rmpflow` | `RMPFlowActionCfg` | Lula GPU YAML configs |
| `curobo` | `CuroboIKActionCfg` (custom) | `num_seeds=10`, `newton_iters=30` |

For full solver details including cuRobo frame conversion, orientation
accumulation, and debugging, see `pingpong_dual_arm/implementation.md#ik-solvers-ik_solvers`.

Joint name patterns per side:
- Right arm: `right_shoulder_.*`, `right_elbow_.*`, `right_wrist_.*`
- Left arm:  `left_shoulder_.*`,  `left_elbow_.*`,  `left_wrist_.*`

End-effector body: `right_wrist_3_link` / `left_wrist_3_link`.

Swap the solver at config level:
```python
cfg = ThrowingEnvCfg()
cfg.ik_solver = "osc"
cfg.playing_arm_side = "left"
```

### IK Solver Testing (`scripts/test_ik_throwing.py`)

Multi-phase pick-and-throw benchmark: the drink is spawned on the table at
`(0.40, 0.50, 0.72)` (settles to z≈0.60) and the right arm executes a full
pick-and-throw sequence using pure physics interaction (no kinematic
attachment). The gripper approaches **from above** (top-down), grasps above
the drink's center, lifts, and throws via direct wrist_2 joint control.

| Phase | Steps | Description |
|-------|-------|-------------|
| **APPROACH** | 60 | EE moves from crane pose to XY directly above the drink at crane Z height. Position-only IK (no orientation change). |
| **DESCEND** | 100 | EE lowers to `GRASP_Z_OFFSET` above the drink center (default 0.3 m above). IK scale 0.8 for fast convergence. |
| **GRASP** | 20 | Gripper closes **gradually** (0.0 → 0.7 over 20 steps via ramped `_set_gripper_state`). |
| **LIFT** | 60 | EE returns to crane pose (position-only, no orientation change). |
| **THROW** | 40 | **Direct wrist_2 joint control** — bypasses IK entirely. All arm joint targets are held at LIFT-end positions; only `right_wrist_2_joint` follows the throw trajectory. Uses `robot.set_joint_position_target()` + `robot.write_data_to_sim()` + `env.sim.step()`. Gripper opens at `THROW_RELEASE_PROGRESS`. |
| **FLIGHT** | ~300 | Drink flies, auto-detect landing (velocity < 0.05 m/s for 30 steps), report 3D distance to target; cycle repeats. |

**Drink position**: The drink is placed at `(DRINK_WORLD_X, DRINK_WORLD_Y,
DRINK_WORLD_Z)` = `(0.40, 0.50, 0.72)`. After physics settling (60 steps),
the **actual** root position is read from `milk.data.root_pos_w` and used
for all subsequent target computations. `BOTTLE_OFFSET_LOCAL` is set to zero
— the EE targets the drink's root position directly. `GRASP_Z_OFFSET`
(default 0.05 m) raises the grasp point above the drink center so fingers
wrap around the upper portion.

**Throw power tuning** — two constants at the top of the script control the
throw velocity. The drink's release speed comes **entirely** from the angular
velocity of wrist_2 at the moment the gripper opens × the lever arm (no
artificial velocity injection):

```python
THROW_SNAP_RAD = 6.0          # forward snap angle (radians)
THROW_RELEASE_PROGRESS = 0.55 # when gripper opens (fraction through throw)
```

| Parameter | Effect on release velocity |
|-----------|---------------------------|
| `THROW_SNAP_RAD` | Larger → more total rotation → higher ω |
| `PHASE_STEPS["THROW"]` | Fewer → same angle in less time → higher ω |
| `THROW_RELEASE_PROGRESS` | Determines launch angle (velocity is constant throughout ramp) |

Approximate angular velocity (constant, linear ramp):
```
ω = SNAP_RAD / (THROW_STEPS × dt)
v_release = ω × lever_arm
```

Default: `6.0 / (40 × 1/120) = 18 rad/s`. With ~15 cm lever
arm: `18 × 0.15 = 2.7 m/s`.

To increase throw power (pick one or combine):
- Increase snap angle: `THROW_SNAP_RAD = 10.0`
- Halve throw steps: `PHASE_STEPS["THROW"] = 20`

**Throw trajectory** (`_throw_angle`): Linear wrist_2 angle ramp (radians):
- 0 → 1.0: linear from 0 to THROW_SNAP_RAD (constant angular velocity)
- Gripper opens at THROW_RELEASE_PROGRESS (default 0.55)

**Config overrides**: `disable_attachment=True`, `randomize_target=False`,
`release_vel_threshold=inf`, `release_at_step=0`. The script sets
`_holding=False` and `_released=False` to prevent the env's built-in
`object_settled` termination from triggering an unwanted auto-reset during
the approach/settle phases. The `attach_milk_to_gripper` reset event is
also skipped when `disable_attachment=True`.

Per-solver metrics (saved to CSV with `--output`):
- Mean/max position error (cm) during APPROACH/DESCEND/LIFT
- Mean/max orientation error (deg)
- Mean throw distance (m) — 3D distance from drink landing to target
- Best throw distance per solver

```bash
# Single solver
python scripts/test_ik_throwing.py --ik diffik

# Compare all solvers
python scripts/test_ik_throwing.py --compare diffik:osc:rmpflow:curobo --output metrics.csv

# Sweep THROW_SNAP_RAD values (comma-separated)
python scripts/test_ik_throwing.py --ik diffik --sweep "2.0,3.0,4.0,5.0,6.0"

# Sweep with range (start:stop:step)
python scripts/test_ik_throwing.py --ik diffik --sweep "1.0:8.0:1.0" --output sweep.csv
```

### Single-Throw Test (`scripts/test_throw.py`)

Executes one complete throw cycle: spawn bottle → hold (kinematic pose write)
→ throwing arc → release → fly → land → auto-terminate.

```bash
# Visual mode
python scripts/test_throw.py --ik diffik --amp 0.3 --release-at 30

# Headless
python scripts/test_throw.py --ik diffik --headless --amp 0.3

# Indefinite throws (Ctrl+C to stop)
python scripts/test_throw.py --ik diffik --amp 0.3 --loop
```

Output columns: `step`, `ee_y`, `ee_z`, `obj_y`, `obj_z`, `v_obj`, `dist3d`, `state`.
3D Euclidean distance to target reported at completion.

## RL Training Pipeline

### Registration

Environment registered in `source/Throwing/Throwing/tasks/throwing/__init__.py`:
```python
gym.register(
    id="Throwing-Direct-v0",
    entry_point="tasks.throwing_env:ThrowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.throwing_env_cfg:ThrowingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

### Agent Config (`skrl_ppo_cfg.yaml`)

| Parameter | Value |
|-----------|-------|
| Network | MLP [512, 256, 128], ELU activations |
| rollouts | 256 |
| learning_epochs | 8 |
| mini_batches | 8 |
| discount_factor | 0.999 |
| λ (GAE) | 0.95 |
| learning_rate | 5e-4, KLAdaptiveLR (kl_thresh=0.008) |
| state/value preprocessor | RunningStandardScaler |
| Total timesteps | 300,000 |

Smaller network and fewer rollouts than pingpong (512→256 rollouts,
[1024,512,256,128]→[512,256,128]) because the task is simpler (single robot,
single object, no game logic).

### Training CLI

```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
cd /home/vladi/IsaacLab/master_isaac/throwing_enviroment

python scripts/skrl/train.py \
    --task=Throwing-Direct-v0 \
    --headless \
    --num_envs=1024 \
    --ik_solver=diffik \
    --playing_arm_side=right \
    --max_iterations=500
```

## Plugging in RL

```python
from tasks.throwing_env_cfg import ThrowingEnvCfg
from tasks.throwing_env import ThrowingEnv

cfg = ThrowingEnvCfg()
cfg.scene.num_envs = 64
cfg.ik_solver = "diffik"
cfg.playing_arm_side = "right"
env = ThrowingEnv(cfg=cfg)
obs = env.reset()  # {"policy": (64, 27)}
action = policy(obs["policy"])  # (64, 6)
obs, reward, terminated, truncated, info = env.step(action)
```

## Mesh Assets & Conversion

### Synthesis Assets (Primary)

Objects are imported from the [Synthesis-Assets-Explorer](https://github.com/Extwin-Synthesis/Synthesis-Assets-Explorer)
repository as pre-built USD files designed for Isaac Sim:

| Object | Source USD | Pre-baked Output | Pre-bake Script |
|--------|-----------|-----------------|-----------------|
| Drink bottle | `assets/new_usds/drink001/model_drink001.usd` | `drink_target.usd` | `scripts/prebake_drink.py` |
| Shopping basket | `assets/new_usds/shopping basket002/model_basket_22.usd` | `basket_target.usd` | `scripts/prebake_basket.py` |

### Pre-baking Steps

**`scripts/prebake_drink.py`** — applies to the drink bottle:
- `MassAPI` to root prim (enables mass property overrides at spawn time)
- `CollisionAPI` + `PhysxCollisionAPI` with `convexDecomposition` to all 2 meshes
- `PhysxMaterialAPI` with `static_friction=5.0`, `dynamic_friction=5.0`, `restitution=0.1` on all collision meshes

**`scripts/prebake_basket.py`** — applies to the shopping basket:
- Removes `RigidBodyAPI` from handle prims (handles become static visual children)
- Removes `ArticulationRootAPI` from body prim (prevents articulation errors)
- Deactivates handle joints and handle prims (hides cosmetic thin bars)
- Keeps `RigidBodyAPI` only on `E_body_20` (single kinematic rigid body)
- `CollisionAPI` + `PhysxCollisionAPI` with `convexDecomposition` to all 3 meshes
- `MassAPI` on `E_body_20`: 2.0 kg

```bash
python scripts/prebake_drink.py
python scripts/prebake_basket.py
```

### Object Configuration in throwing_env_cfg.py

| Object | Spawner | USD Path | Mass | Kinematic |
|--------|---------|----------|------|-----------|
| drink | `UsdFileCfg` | `assets/new_usds/drink001/drink_target.usd` | 0.5 kg | No |
| basket | `UsdFileCfg` | `assets/new_usds/shopping basket002/basket_target.usd` | 2.0 kg | Yes |
| table | `CuboidCfg` | — (procedural) | — | Yes |
| stand | `CuboidCfg` | — (procedural) | — | Yes |

### Legacy Mesh Conversion Pipeline

Original Gazebo assets (DAE/OBJ) can be converted to USD via the legacy pipeline:

```bash
# DAE → OBJ (assimp CLI, one-time)
assimp export model.dae output.obj -f obj

# OBJ → USD (inside Isaac Sim)
python scripts/convert_meshes.py
```

Generates `generated_usd/milk.usd`, `wooden_box.usd`, `obstacle_box.usd`.
These are **not currently used** — the Synthesis assets have replaced them.

### Scene Export (`scripts/export_full_scene.py`)

Exports the complete throwing environment as a reference-based USD scene
for inspection and editing in the Isaac Sim GUI. **Robot link transforms are
baked from live simulation FK**, so the robot appears in the correct pose
when opened — no Script Editor workflow required.

```bash
# Default crane pose
python scripts/export_full_scene.py

# Custom pose via JSON string
python scripts/export_full_scene.py \
  --pose '{"right_elbow_joint": 0.0, "right_wrist_1_joint": 0.0}'

# Custom pose via JSON file
echo '{"right_shoulder_lift_joint": -2.0}' > my_pose.json
python scripts/export_full_scene.py --pose my_pose.json
```

**How FK baking works:**

1. Launches a headless simulation, creates the environment, resets to the
   configured pose, and steps 5 frames for physics to settle
2. Reads all 25 body world transforms (`body_pos_w`, `body_quat_w`) from the
   robot articulation's Fabric data buffers
3. Computes each link's local transform in the `/ur` frame via FK:
   `local_pos = quat_inverse(root_quat).rotate(body_pos - root_pos)`
4. Overwrites `XformOp:translate` and `XformOp:orient` on each link prim
   in the robot USD with the FK result
5. Also sets `drive:angular:physics:targetPosition` on all 24 joint prims
   to match the current joint angles
6. Builds `full_scene.usd` referencing the modified robot + drink, basket,
   table, stand, and ground plane

**Output files in `generated_usd/`:**

| File | Description |
|------|-------------|
| `full_scene.usd` | Reference-based scene — robot appears in pose at open |
| `joint_pose.json` | 24 joint angles captured from the simulation |
| `body_poses.json` | 25 body world positions/orientations (FK debug data) |
| `robot_crane_pose/` | Modified robot USD with FK-baked link transforms + joint targets |

**To export a fully flattened, single-file USD** (no external references):

1. Open Isaac Sim GUI
2. File > Open > `full_scene.usd`
3. File > Export > USD (check **Flatten**) to save the self-contained scene

**Editing finger grippers in the exported scene:**

1. In the Stage tree, find:  
   `World/Robot/.../rgripper_left_inner_finger` and `rgripper_right_inner_finger`
2. Select the collision child prim → Property panel
3. To enlarge pads: increase the collision box depth from `0.0075` to `0.03`
4. To add friction: Property > Physics Material:  
   `Static Friction = 5.0`, `Dynamic Friction = 5.0`, `Restitution = 0.1`
5. File > Save As > `robotiq_gripper_modified.usd`

**Adjusting the robot pose:**

Three options to change the arm pose before export:

1. **Edit config** — modify `DualArm_CFG.init_state.joint_pos` in
   `throwing_env_cfg.py:118-134`, then re-run the export script.
2. **`--pose` CLI** — pass JSON overrides for specific joints (see above).
3. **`--pose` with file** — load joint overrides from a JSON file.

Available joint names (24 total):

| Group | Joint names |
|-------|-------------|
| Left arm | `left_shoulder_pan_joint`, `left_shoulder_lift_joint`, `left_elbow_joint`, `left_wrist_1_joint`, `left_wrist_2_joint`, `left_wrist_3_joint` |
| Right arm | `right_shoulder_pan_joint`, `right_shoulder_lift_joint`, `right_elbow_joint`, `right_wrist_1_joint`, `right_wrist_2_joint`, `right_wrist_3_joint` |
| Grippers | `lgripper_finger_joint`, `rgripper_finger_joint`, plus 8 knuckle/finger joints |

## Key Differences from PingPong

| Aspect | PingPong | Throwing |
|--------|----------|----------|
| Robots | 2 dual-arm robots at y=±2.7 | 1 dual-arm robot at origin |
| URDF | `dual_arm_robot_rackets.urdf` | `dual_arm_robot.urdf` (with grippers) |
| End-effector | Rackets (fixed joints) | Robotiq grippers |
| Action dim | 12 (two 6-D arms) | 6 (one 6-D arm) |
| Obs dim | 69 | 27 |
| Physics objects | Table, ball (dynamic) | Table, drink (dynamic), basket (kinematic) |
| Core mechanic | Paddle-ball contact + table zones | Kinematic object attachment + velocity release |
| Rewards | Contact, velocity, table success/fail, floor | Distance, success bonus, EE velocity |
| RL network | MLP [1024, 512, 256, 128] | MLP [512, 256, 128] |
| Rollouts | 512 | 256 |
| Assets source | Custom USD from TableTennisRobot | Synthesis-Assets-Explorer |

## Known Issues

1. **IK solvers reside in pingpong_dual_arm**: The `ik_solvers/` module is imported
   from `pingpong_dual_arm/ik_solvers/` via `sys.path`. No separate copy is maintained.
   Changes to the pingpong IK solvers (e.g. new solver types, config changes) will
   automatically affect the throwing environment.

2. **Left arm is uncontrolled**: Only the configured `playing_arm_side` receives IK
   commands. The idle arm stays at its default pose.

3. **Object attachment is non-physical (Option A)**: Kinematic pose write means the
   object doesn't interact with the gripper through contact forces. Physics grip
   (Option B) was attempted but failed due to finger pad collision geometry being
   too thin (7.5mm). The `disable_attachment` config flag allows scripts to take
   full manual control and bypass the kinematic attachment system entirely.

4. **Robotiq 2F-140 mimic joints require explicit handling**: PhysX mimic constraints
   do not resolve instantly on `write_joint_state_to_sim`. All 6 revolute gripper
   joints per side must be explicitly set — not just `rgripper_finger_joint`.
   The `_set_gripper_state()` helper in `events.py` handles this by matching
   all joints via regex patterns and computing positions from the URDF mimic
   multipliers. The gripper actuator config in `DualArm_CFG` also covers all
   6 joints.

5. **Table requires explicit collision properties**: The table cuboid had
   `kinematic_enabled=True` but no `collision_props`, causing dynamic objects to
   pass through. Fixed by adding `collision_props=CollisionPropertiesCfg(collision_enabled=True)`.

6. **Observation side hardcoded**: Observation terms currently reference
   `right_wrist_3_link` and `ARM_JOINTS_RIGHT`. Switching `playing_arm_side`
   to `"left"` only changes the action arm.

7. **`write_root_velocity_to_sim` unreliable**: Attempts to set explicit release
   velocity on the bottle had limited/no effect. The object's post-release velocity
   is dominated by the pose-teleport artifacts from the holding phase.

8. **cuRobo CUDA graph**: The cuRobo solver uses `use_cuda_graph=True` which
    requires CUDA ≥ 12.0 for graph resets. See pingpong implementation.md for
    full details on the cuRobo IK solver integration.

9. **Robot joint baking uses FK from simulation**: `export_full_scene.py` now
   reads live body transforms from the Fabric stage, computes forward kinematics,
   and bakes the resulting link transforms + joint targets directly into the robot
   USD. The Scene Editor workflow is no longer needed. Programmatic flattening
   (single-file USD) still requires the GUI's File > Export > Flatten option.

10. **Fabric/USDRT stage cannot be exported programmatically**: `save_as_stage()`
     and `UsdUtils.FlattenLayerStack` both fail on Fabric-backed stages. The
     scene export works around this by building a reference-based USD from
     Isaac Lab data buffers (positions) and cached USD files (robot, assets).

11. **`object_settled` termination triggers unwanted auto-reset in test scripts**:
    Setting `env._released=True` (needed to disable kinematic attachment) also
    enables the `object_settled` termination condition. If the drink sits still on
    the table for 30 steps, the env auto-resets mid-approach. **Fix**: set
    `env._released=False` in test scripts — `disable_attachment=True` already
    prevents attachment without needing the `_released` flag.

12. **`attach_milk_to_gripper` closes gripper on every reset**: The reset event
    unconditionally closes the gripper to 0.7 and teleports the drink to the EE.
    This interfered with the IK benchmark where the drink should stay on the table.
    **Fix**: `attach_milk_to_gripper` now returns immediately when
    `cfg.disable_attachment=True`.

13. **DiffIK singularity jumps at workspace boundary**: The DLS method with λ=0.1
    amplifies motion in near-singular directions by 1/λ²=100×. When the EE
    approaches the workspace edge, the IK produces catastrophic joint-space jumps.
    The test script mitigates this by using `IK_DEFAULT_SCALE=0.8` and sufficient
    phase step counts. The auto-reset fix (#11) was the primary cause of the
    observed "teleport" behavior.

14. **`set_joint_position_target` only writes to buffer**: The function does NOT
    sync to PhysX immediately — it only fills `_data.joint_pos_target`.
    `write_data_to_sim()` must be called to sync targets to the simulation.
    During `env.step()`, this sync happens automatically after `apply_actions()`.
    For manual joint control (e.g. THROW phase), call
    `robot.write_data_to_sim()` + `env.sim.step()` explicitly.

## Stack

| Component | Version | Location |
|-----------|---------|----------|
| Isaac Sim | 5.1.0-rc.19 | `/home/vladi/isaac-sim-5.1.0/` |
| Isaac Lab | 0.54.3 (2.3.2) | `/home/vladi/IsaacLab/source/isaaclab/` |
| Python | 3.11 | `/home/vladi/IsaacLab/master_isaac/.master_venv` |
| PyTorch | 2.7.0+cu128 | GPU |
| skrl | >= 1.4.2 | `.master_venv` |
