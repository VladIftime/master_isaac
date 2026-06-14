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
│                   ├── skrl_ppo_cfg.yaml  # PPO agent configuration
│                   └── skrl_sac_cfg.yaml  # SAC agent configuration (macro-action)
├── tasks/                                 # Environment implementation
│   ├── __init__.py
│   ├── throwing_env_cfg.py                # Scene + MDP config (ManagerBasedRLEnv)
│   ├── throwing_env.py                    # ManagerBasedRLEnv class (attachment, reward logic)
│   ├── throwing_direct_env_cfg.py         # DirectRLEnvCfg — fast training (no managers)
│   ├── throwing_direct_env.py             # DirectRLEnv — state-machine throw (replicate_physics)
│   ├── throwing_primitive_env_cfg.py      # SAC macro-action env config (legacy gymnasium wrapper)
│   ├── throwing_primitive_env.py          # SAC one-shot gymnasium.Env wrapper (legacy)
│   ├── throw_primitive.py                 # Throw primitive execution (batched + single)
│   ├── throw_validation_configs.py        # 10 predefined target configs for validation
│   ├── observations.py                    # Observation functions (joints, EE, objects)
│   ├── rewards.py                         # Reward functions (distance, success, velocity)
│   ├── events.py                          # Reset events (robot reset, randomize target, attach)
│   └── terminations.py                    # Boundary + settled checks
├── scripts/
│   ├── train_sac.py                       # SAC training launcher (DirectRLEnv + skrl)
│   ├── validate_throw.py                  # Validation: 10 test targets × 3 attempts + plots (--fast for DirectRLEnv)
│   ├── prebake_robot_usd.py               # One-time URDF→USD conversion for faster startup
│   ├── skrl/
│   │   ├── train.py                       # Training launcher (skrl PPO)
│   │   └── play.py                        # Inference / playback
│   ├── test_env.py                        # Launch & step environment
│   ├── test_ik_throwing.py                # Multi-phase pick-and-throw IK benchmark (approach → descend → grasp → lift → raise → extend → settle → wrist-snap throw)
│   ├── test_joint_throwing.py             # Gazebo-style joint-space throw (IK reach → bang-bang all-joint catapult)
│   ├── test_throw.py                      # Throw primitive test (same pipeline as SAC, --loop uses pre-position cache)
│   ├── plot_tb_logs.py                    # Plot TensorBoard logs (reward, distance, entropy, Q-values) + CSV export
│   ├── convert_meshes.py                  # OBJ → USD with MeshConverter
│   ├── prebake_physics.py                 # Apply CollisionAPI + PhysicsMaterial to USD meshes
│   ├── prebake_drink.py                   # Pre-bake drink001: MassAPI, CollisionAPI, high-friction physics material (properly bound)
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
├── ik_solvers/                            # Local IK solver module (from pingpong_dual_arm)
│   ├── __init__.py
│   └── curobo_ik.py
├── urdf/
│   └── dual_arm_robot.urdf               # Dual-arm robot URDF (local copy)
├── hpc/
│   ├── train_sac.slurm                   # SLURM job script — ManagerBased path (legacy)
│   └── train_sac_direct.slurm            # SLURM job script — DirectRLEnv (fast, 4096 envs)
├── gazebo_impl/                           # Reference Gazebo implementation
│   ├── behaviour_change.cpp
│   ├── new_impl.cpp
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
            |  ====[Table]====  (y=1.0, z=0.5, kinematic, 2.0×1.7 m)
           |
           |       [Basket]    (x=±0.45, y=0.7–1.3, on table)
           |
```

- **Robot position**: global constant `ROBOT_POS = (0, 0, 0.6)` — body base at z=0.6 atop a kinematic stand
- **Stand**: Box cuboid (0.5×0.5×0.6 m) beneath the robot, kinematic, centered at half height (z=0.3)
- **Table**: Box cuboid (2.0×1.7×0.05 m) at `(0, 1.0, TABLE_Z - 0.025)`, kinematic, surface at TABLE_Z. Same height as the robot stand — the robot and table share a common work surface.
- **Drink object**: Dynamic rigid body, mass 0.5 kg. Visual mesh loaded via `UsdFileCfg` from `assets/new_usds/drink001/drink_target.usd` (Synthesis drink bottle with pre-baked MassAPI, CollisionAPI, and high-friction PhysxMaterial). In RL training mode, spawned at `right_wrist_3_link` at episode reset and kinematically attached via `write_root_pose_to_sim`. In the standalone IK benchmark (`test_ik_throwing.py`), spawned on the table at a fixed position `(0.40, 0.40, 0.72)` (settles to z≈0.60) and interacts purely through physics — no kinematic attachment. The EE targets the drink root position directly (zero offset).
  Bottle root offset `(-0.012, 0.129, -0.176)` in EE-local frame is used only for RL kinematic attachment (Option A).
- **Target (basket)**: Kinematic basket, randomized in XY on the table surface. Visual mesh loaded via `UsdFileCfg` from `assets/new_usds/basket_02/model_basket1.usd` with `scale=(0.4, 0.4, 0.4)` (raw USD is ~0.92×1.28×0.40 m, scaled to ~0.37×0.51×0.16 m). ArticulationRoot disabled via `articulation_enabled=False`. Mass 2.0 kg.
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

# Throw primitive test — DirectRLEnv (fast, no IK)
python scripts/test_throw.py --direct --loop
python scripts/test_throw.py --direct --loop --initial_jv 0.5 --final_jv -0.3

# Throw primitive test — ManagerBased (IK grasping, slower)
python scripts/test_throw.py --ik diffik --loop

# Pre-bake robot URDF to USD (one-time optimization)
python scripts/prebake_robot_usd.py

# Plot training logs (auto-finds latest run)
python scripts/plot_tb_logs.py

# Multi-phase pick-and-throw IK benchmark
python scripts/test_ik_throwing.py --ik diffik
python scripts/test_ik_throwing.py --compare diffik:osc:rmpflow:curobo --output metrics.csv

# Gazebo-style joint-space throw (bang-bang all-joint catapult)
python scripts/test_joint_throwing.py
python scripts/test_joint_throwing.py --headless --num_throws 10
python scripts/test_joint_throwing.py --target_x 0.0 --target_y 1.0 --throw_steps 60

# Re-bake drink friction (run after modifying prebake_drink.py)
python scripts/prebake_drink.py

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

### Option C: PhysX Fixed Joint Constraint (Attempted — Failed with Fabric)

Create a physics fixed joint between the gripper pad and object root at reset,
delete it at release. Equivalent to Gazebo's `graspObjectInGazebo()`. Requires
`UsdPhysics.FixedJoint.Define()` + `stage.RemovePrim()` at runtime.

**Attempted in `test_joint_throwing.py`** but failed: PhysX with Fabric-backed
simulation does not pick up USD-level topology changes (new joints) at runtime.
The joint is created on the USD stage but the GPU physics cache ignores it.
PhysX warns "disjointed body transforms" and the constraint has no effect.
Would require a full scene rebuild or a non-Fabric physics pipeline to work.

**Current recommendation**: Use physics gripper friction (Option B) with
high-friction material (5.0/5.0) properly bound via `UsdShade.MaterialBindingAPI`.
This holds reliably for moderate accelerations.

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
TABLE_SIZE = (2.0, 1.7, 0.05)      # tabletop dimensions (x, y, z)
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
  wrist_3:        1.5708  (π/2 — EE rotated 90° for grasp alignment)

Left arm (idle):
  shoulder_pan:   0.0
  shoulder_lift: -1.57
  elbow:         -1.57
  wrist_1:       -1.57
  wrist_2:        1.57
  wrist_3:        1.5708  (π/2 — EE rotated 90° for grasp alignment)

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

IK solvers are now **local** to the throwing_enviroment project (copied from
`pingpong_dual_arm/ik_solvers/`). The `build_ik_action()` dispatcher supports
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
`(0.65, 0.50, 0.72)` (settles to z≈0.60) and the right arm executes a full
pick-and-throw sequence using pure physics interaction (no kinematic
attachment). The gripper approaches **from above** (top-down), grasps above
the drink's center, lifts, raises Z to throw height, extends toward the
target, pauses to settle, and throws via direct wrist_2 joint control.

Orientation preservation is achieved by splitting the old EXTEND phase (XY+Z
simultaneously) into separate Z-only and XY-only phases. Each phase
constrains the IK solver to fewer DOFs, preventing it from finding joint
configurations that sacrifice orientation to reach the position target.

| Phase | Steps | Description |
|-------|-------|-------------|
| **APPROACH** | 60 | EE moves from crane pose to XY directly above the drink at crane Z height. Position-only IK (no orientation change). |
| **DESCEND** | 100 | EE lowers to `GRASP_Z_OFFSET` above the drink center (default 0.3 m above). IK scale 0.8 for fast convergence. |
| **GRASP** | 20 | Gripper closes **gradually** (0.0 → 0.7 over 20 steps via ramped `_set_gripper_state`). |
| **LIFT** | 60 | EE returns to crane pose (position-only, no orientation change). |
| **RAISE** | 40 | Z-only lift by `THROW_EXTEND_Z_OFFSET` from crane height. Preserves orientation by constraining IK to a single DOF. |
| **EXTEND** | 65 | XY-only reach toward the target at raised height, moving `EXTEND_RATIO` of the EE-to-target distance (default 0.5 = 50%). Preserves orientation by constraining IK to XY only. |
| **SETTLE** | 15 | Pause with zero action to let the object settle in the gripper before the rapid wrist snap. |
| **THROW** | 40 | **Direct wrist_2 joint control** — bypasses IK entirely. All arm joint targets are held at EXTEND-end positions; only `right_wrist_2_joint` follows the throw trajectory. Uses `robot.set_joint_position_target()` + `robot.write_data_to_sim()` + `env.sim.step()`. Gripper opens at `THROW_RELEASE_PROGRESS`. |
| **FLIGHT** | ~300 | Drink flies, auto-detect landing (velocity < 0.05 m/s for 30 steps), report 3D distance to target; cycle repeats. |

**Drink position**: The drink is placed at `(DRINK_WORLD_X, DRINK_WORLD_Y,
DRINK_WORLD_Z)` = `(0.65, 0.50, 0.72)`. After physics settling (60 steps),
the **actual** root position is read from `milk.data.root_pos_w` and used
for all subsequent target computations. `BOTTLE_OFFSET_LOCAL` is set to zero
— the EE targets the drink's root position directly. `GRASP_Z_OFFSET`
(default 0.30 m) raises the grasp point above the drink center so fingers
wrap around the upper portion.

**Throw power tuning** — three constants at the top of the script control the
throw velocity. The drink's release speed comes **entirely** from the angular
velocity of wrist_2 at the moment the gripper opens × the lever arm (no
artificial velocity injection):

```python
THROW_SNAP_RAD = 10.0         # forward snap angle (radians)
THROW_RELEASE_PROGRESS = 0.55 # when gripper opens (fraction through throw)
THROW_EXTEND_Z_OFFSET = 0.20  # Z raise during RAISE phase for extra height
EXTEND_RATIO = 0.5             # fraction of EE-to-target distance to extend (0.5 = 50%)
```

Target dimensions for the shopping basket (used by spawn area visualization):

```python
TARGET_WIDTH = 0.38
TARGET_LENGTH = 0.51
TARGET_HEIGHT = 0.27
```

| Parameter | Effect on release velocity |
|-----------|---------------------------|
| `THROW_SNAP_RAD` | Larger → more total rotation → higher ω |
| `PHASE_STEPS["THROW"]` | Fewer → same angle in less time → higher ω |
| `THROW_RELEASE_PROGRESS` | Determines launch angle (velocity is constant throughout ramp) |
| `EXTEND_RATIO` | Larger → EE closer to target at release → better aim, but may strain IK |

Approximate angular velocity (constant, linear ramp):
```
ω = SNAP_RAD / (THROW_STEPS × dt)
v_release = ω × lever_arm
```

Default: `10.0 / (40 × 1/120) = 30 rad/s`. With ~15 cm lever
arm: `30 × 0.15 = 4.5 m/s`.

To increase throw power (pick one or combine):
- Increase snap angle: `THROW_SNAP_RAD = 12.0`
- Halve throw steps: `PHASE_STEPS["THROW"] = 20`
- Increase Z raise: `THROW_EXTEND_Z_OFFSET = 0.35` (more height for downward arc)
- Increase EXTEND reach: `EXTEND_RATIO = 0.7` (arm reaches further forward)
- Increase EXTEND steps: `PHASE_STEPS["EXTEND"] = 100` (more build-up time)

**Throw trajectory** (`_throw_angle`): Linear wrist_2 angle ramp (radians):
- 0 → 1.0: linear from 0 to THROW_SNAP_RAD (constant angular velocity)
- Gripper opens at THROW_RELEASE_PROGRESS (default 0.55)

**Spawn area visualization**: On startup, a semi-transparent blue cuboid
(`opacity=0.3`) and four bright-blue corner spheres are drawn on the table
surface at `TABLE_Z + 0.1`, covering the XY range defined by
`cfg.target_x_range` × `cfg.target_y_range`. This shows the region where
target baskets will be randomized.

**Config overrides**: `disable_attachment=True`, `randomize_target=True`,
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
```

### Joint-Space Throwing (`scripts/test_joint_throwing.py`)

Gazebo-style joint-space pick-and-throw benchmark. Inspired directly by the
`gazebo_impl/primitive_design.cpp` and `behaviour_change.cpp` patterns:
pre-defined joint waypoints, `directlySetAllJoints`-style interpolation, and
timed gripper release. Object held by physics gripper friction (high-friction
material on drink, combine_mode="max").

**Architecture — hybrid approach:**
- **APPROACH/DESCEND/LIFT**: Task-space IK via `env.step(action)` with
  `compute_pose_error()` — proven reliable for reaching the drink.
- **THROW**: Pure joint-space bang-bang control — all 6 arm joints target
  `throw_end_joints` from step 0, PD controller drives maximum acceleration.
  Gripper opens at `RELEASE_PROGRESS` fraction.

This hybrid is necessary because:
1. Standalone `DifferentialIKController.compute()` fails with the dual-arm
   robot (Jacobian indexing issue — see Known Issue #16).
2. Direct `robot.set_joint_position_target()` + `env.sim.step()` only works
   AFTER `env.step()` has been called (to initialize the actuator PD
   controller state). The IK phases provide this initialization.
3. PhysX FixedJoint creation via USD API is unreliable with Fabric-backed
   simulation (see Known Issue #17).

| Phase | Steps | Method | Description |
|-------|-------|--------|-------------|
| **SETTLE** | 60 | `env.step(zeros)` | Spawn drink on table, settle physics |
| **APPROACH** | 60 | `env.step(pos_err)` | EE moves XY above drink at crane Z |
| **DESCEND** | 100 | `env.step(pos_err)` | EE lowers to `GRASP_Z_OFFSET` above drink |
| **GRASP** | 20 | `env.step(zeros)` + gripper ramp | Gripper closes 0.0 → 0.48 (or force-feedback with `--force_grasp`) |
| **LIFT** | 60 | `env.step(pos_err)` | EE returns to crane pose |
| **THROW** | 40 | `set_joint_position_target` (bang-bang) | All 6 joints target throw_end simultaneously |
| **FLIGHT** | ~300 | `env.step(zeros)` | Detect landing, report distance |
| **RETURN** | 60 | `env.step(pos_err)` | EE returns to crane pose |

**Bang-bang control** (Gazebo's `directlySetAllJoints` equivalent):
Instead of linearly interpolating targets (which gives the PD controller only
small errors to track → slow motion), the THROW phase sets `throw_end_joints`
as the target from step 0 every step. The full position error drives maximum
PD force, accelerating the arm as fast as actuator stiffness/damping allows.
Release happens at peak velocity (before the arm decelerates approaching the
target).

**Throw waypoint computation** (aim at target):
```python
aim_angle = atan2(target_x, target_y)
throw_end[0] = aim_angle - ARM_THROW_DIRECTION_OFFSET  # shoulder_pan aims throw
throw_end[1] = start[1] + SHOULDER_LIFT_DELTA * power  # catapult UP
throw_end[2] = start[2] + ELBOW_DELTA * power          # extend FORWARD
throw_end[3:6] = start[3:6]                            # wrists hold steady
```

Where:
- `ARM_THROW_DIRECTION_OFFSET = -π/2` compensates for the arm's tangential
  velocity direction. In crane pose the arm points in +X; a positive
  shoulder_pan rotation (counter-clockwise from above) produces +Y tangential
  velocity at the EE. The offset converts the target aim angle into the
  correct shoulder_pan value so the tangent of the arm's rotation arc points
  toward the target.
- `SHOULDER_LIFT_DELTA = 0.44` and `ELBOW_DELTA = -1.47` from Gazebo's
  `primitive_design.cpp` (init→end joint differences)
- `power = clamp(dist / NOMINAL_DIST, 0.6, 1.5)` scales throw intensity by
  target distance

**Key constants:**
```python
GRASP_Z_OFFSET = 0.33          # EE height above drink
GRASP_STR = 0.48               # fixed grip position (empirically determined via force feedback)
RELEASE_PROGRESS = 0.40        # release at 40% through throw (peak velocity)
ARM_THROW_DIRECTION_OFFSET = -math.pi / 2  # aim correction for arm kinematics
SHOULDER_LIFT_DELTA = 0.44     # from Gazebo primitive_design.cpp
ELBOW_DELTA = -1.47            # from Gazebo primitive_design.cpp
NOMINAL_DIST = 1.0             # reference distance for power scaling
```

**Gripper control — two modes** (selected by `--force_grasp` flag):

1. **Fixed ramp** (default): Linearly ramps gripper from 0 → `GRASP_STR=0.48`
   over `PHASE_STEPS["GRASP"]` steps. Empirically calibrated via force-feedback
   experiments — 0.48 is the position where the finger force first exceeds 65N
   on the drink bottle (deterministic across all tested throws).

2. **Force feedback** (`--force_grasp`): Incrementally closes by `GRASP_INCREMENT=0.02`
   per step, reading `robot.data.body_incoming_joint_wrench_b` on the inner finger
   bodies. Stops when max force exceeds `GRASP_FORCE_THRESHOLD=65.0N`. Logs per-finger
   forces every 2 steps. Used for calibration/debugging only.

Training (`throw_primitive.py`) always uses the fixed ramp to 0.48 — no force
sensing overhead in the batched training loop.

**Lessons learned during development:**

1. **Standalone `DifferentialIKController` fails with dual-arm**: The Jacobian
   from `robot.root_physx_view.get_jacobians()` indexed by `arm_ids` does not
   produce correct IK solutions for the UR5e in dual-arm configuration. The
   `compute()` function returns the input joint positions unchanged (zero delta).
   Root cause unresolved — likely a body/joint index mapping issue.

2. **PhysX FixedJoint unreliable with Fabric**: Creating a `UsdPhysics.FixedJoint`
   via USD API at runtime does not reliably constrain bodies when Fabric is
   enabled. PhysX warns "disjointed body transforms" and the constraint has no
   effect on simulation.

3. **PD controller cannot generate fast catapult motion**: With implicit actuator
   stiffness=8000 and damping=500, shoulder and elbow joints (high inertia)
   cannot accelerate fast enough for a Gazebo-style catapult in 40 steps (0.33s).
   Linear interpolation of targets makes this worse (small error → small force).
   Bang-bang partially mitigates but the arm still moves slowly. The wrist_2-only
   snap in `test_ik_throwing.py` works because wrist inertia is much lower.

4. **GRASP_Z_OFFSET must be ≤ gripper finger reach**: With offset=0.3m, the EE
   is 30cm above the drink but Robotiq 2F-140 fingers are only ~14cm long. The
   gripper closes around air, not the drink. Reducing to 0.10-0.12m places the
   finger pads at drink height.

5. **Aim direction is determined by tangential velocity, not radial**: The
   drink's release velocity comes from the **tangent** to the shoulder_pan
   rotation circle at the arm's current position — NOT from the direction the
   arm points. In crane pose the EE is at (+X, +0.18Y) relative to the base.
   A positive (counter-clockwise) shoulder_pan rotation produces +Y tangential
   velocity; negative (clockwise) produces -Y. Initial attempts with
   `ARM_THROW_DIRECTION_OFFSET = +π/2` threw backward (-Y); correcting to
   `-π/2` flipped the rotation direction and now throws toward the target (+Y).
   This is the correct kinematic reasoning — not a numerical hack.

6. **Drink friction was not properly baked**: The original `prebake_drink.py`
   applied `PhysxSchema.PhysxMaterialAPI` directly on mesh prims, but this does
   not create a valid physics material binding. The drink retained its source
   friction of 0.5/0.4 instead of the intended 5.0/5.0. Fixed by creating a
   proper `UsdShade.Material` + `UsdPhysics.MaterialAPI` prim and binding it via
   `UsdShade.MaterialBindingAPI`.

**Visual markers** (identical to `test_ik_throwing.py`):
- Yellow sphere at EE position (radius 0.015)
- Green sphere at target basket (radius 0.04)
- Blue semi-transparent cuboid showing the target spawn area on the table
- Blue corner spheres at spawn area extents

**Target randomization** uses the same `ThrowingEnvCfg` default ranges as
`test_ik_throwing.py` (no custom overrides). When `--target_x` / `--target_y`
CLI args are provided, the target is fixed at that position instead.

```bash
# Basic usage (fixed-ramp grasp)
python scripts/test_joint_throwing.py

# Force-feedback grasp (with per-finger force logging)
python scripts/test_joint_throwing.py --force_grasp

# Headless with fixed target
python scripts/test_joint_throwing.py --headless --target_x 0.0 --target_y 1.0

# Tune throw parameters
python scripts/test_joint_throwing.py --throw_steps 60 --release_progress 0.4

# Multiple throws
python scripts/test_joint_throwing.py --num_throws 10
```

### Throw Primitive Test (`scripts/test_throw.py`)

Tests the exact same throw pipeline as SAC training via `execute_primitive_batched()`.
Runs the full IK grasping + joint-space throw sequence with configurable action
parameters. On `--loop`, the **pre-position cache** skips SETTLE+APPROACH+DESCEND
(~220 steps) after the first throw.

A **green sphere marker** at the target center shows the point from which the
landing distance is measured.

```bash
# Single throw (full IK grasping sequence)
python scripts/test_throw.py --ik diffik

# Loop mode — [PRE-POSITIONED] tag on throw #2+
python scripts/test_throw.py --ik diffik --loop

# Custom throw parameters
python scripts/test_throw.py --ik diffik --loop --initial_jv 0.5 --final_jv -0.3

# Multiple parallel envs
python scripts/test_throw.py --ik diffik --loop --num_envs 4
```

Output per throw:
```
[Throw #1] [FULL SEQUENCE]  target=(0.186, 1.231, 0.501)
  env[0]: milk=(+0.281,+0.452,+0.538)  target=(+0.186,+1.231,+0.501)  dist=0.785m
  Mean distance: 0.785m

[Throw #2] [PRE-POSITIONED]  target=(0.300, 1.068, 0.501)
  env[0]: ...  dist=0.598m
```

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

## SAC Training Pipeline — Throw Primitive

### Overview

The SAC (Soft Actor-Critic) pipeline learns a **macro-action** throwing policy.
Instead of outputting per-step EE deltas (like PPO), the SAC agent outputs 4
scalar parameters that define an entire throw trajectory. Each RL "step"
executes the full multi-phase throw primitive (IK grasping + joint-space throw +
flight + landing measurement). The episode is **one-shot**: 1 outer step = 1
complete throw attempt, always terminated after.

This architecture directly mirrors the Gazebo RL implementation in
`gazebo_impl/RL_tossing_object_with_obstacle_avoidance_v3.py`, adapted to
Isaac Lab with skrl as the RL framework.

### Code Structure

```
throwing_enviroment/
├── tasks/
│   ├── throwing_primitive_env.py       # gymnasium.Env wrapper (one-shot episodic)
│   ├── throwing_primitive_env_cfg.py   # Simple dataclass config
│   └── throw_primitive.py             # Core throw execution logic (batched + single)
├── scripts/
│   └── train_sac.py                   # SAC training launcher (skrl Runner)
└── source/Throwing/Throwing/tasks/throwing/agents/
    └── skrl_sac_cfg.yaml              # SAC agent hyperparameters
```

### Action Space (4D Macro-Action)

The agent outputs 4 continuous parameters per throw:

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| 0 | `initial_joint_value` | [-1, 1] | Shoulder_pan angle for wind-up pose (mapped to [0.001, 2.401] rad for right arm) |
| 1 | `final_joint_value` | [-1, 1] | Shoulder_pan angle for throw end pose (mapped to [0.001, 2.401] rad for right arm) |
| 2 | `releasing_time` | [0.05, 1.0] | Fraction of throw duration at which gripper opens |
| 3 | `duration` | [0.1, 1.0] | Total throw trajectory time in seconds |

Action mapping (`map_action_to_params`):
```python
initial_jv = (0.5 * (1 + action[0]) * 2.4) + 0.001   # right arm
final_jv   = (0.5 * (1 + action[1]) * 2.4) + 0.001   # right arm
releasing_time = action[2]                              # direct
duration       = action[3]                              # direct
```

For the left arm, the joint values are negated.

### Observation Space (8D)

| Index | Dimension | Description |
|-------|-----------|-------------|
| 0 | robot_indicator | +1.0 (right arm) or -1.0 (left arm) |
| 1 | basket_x | Target X position / 3.0 (normalized) |
| 2 | basket_y | Target Y position / 3.0 (normalized) |
| 3 | obj_x | Drink X position / 3.0 (normalized) |
| 4 | obj_y | Drink Y position / 3.0 (normalized) |
| 5 | dist | 3D distance (object → target) / 3.0 |
| 6 | dist_x | |X distance| / 3.0 |
| 7 | dist_y | |Y distance| / 3.0 |

All positions are in environment-local coordinates.

### Reward Function

Widened exponential + linear distance reward (original Gazebo sigmas 0.01/0.05
were too narrow, producing near-zero reward for throws >30cm off):

```python
reward = 0.9 * exp(-d² / 0.1) + 0.1 * exp(-d² / 0.5) + 0.5 * max(0, 1 - d)
if d < 0.15:
    reward = 2.0  # success override
```

| Distance (m) | Reward |
|--------------|--------|
| 0.0 | 2.0 (success) |
| 0.15 | 2.0 (success) |
| 0.30 | ~0.80 |
| 0.50 | ~0.39 |
| 1.00 | ~0.01 |

If the drink is dropped before release (detected by falling below
`DRINK_BELOW_TABLE_Z = 0.45` in env-local Z), a penalty distance of
`DROP_PENALTY_DISTANCE = 10.0` m is assigned. Dropped drinks are despawned
during the episode and restored to their last valid position before
observation computation to avoid poisoning the replay buffer.

### Throw Primitive Execution (`tasks/throw_primitive.py`)

Each SAC step internally executes 10 phases using the underlying `ThrowingEnv`:

| Phase | Steps | Method | Description |
|-------|-------|--------|-------------|
| **SETTLE** | 60 | `env.step(zeros)` | Spawn drink on table at `(0.65, 0.50, 0.62)`, settle physics |
| **APPROACH** | 60 | `env.step(ik_action)` | EE moves XY above drink at crane Z height |
| **DESCEND** | 100 | `env.step(ik_action)` | EE lowers to `GRASP_Z_OFFSET=0.30` above drink center |
| **GRASP** | 20 | `env.step(zeros)` + gripper ramp | Gripper closes 0.0 → 0.48 over 20 steps |
| **LIFT** | 60 | `env.step(ik_action)` | EE returns to crane pose (kinematic hold active) |
| **GO_TO_INIT** | 40 | `set_joint_position_target` | Move arm to Gazebo `init_joints_pose` preset |
| **GO_TO_INITIAL** | 40 | `set_joint_position_target` | Move to wind-up pose (`shoulder_pan = initial_joint_value`) |
| **THROW** | 12–120 | Joint interpolation | Linear interpolation from initial→end joints over `duration` |
| **RELEASE** | (within THROW) | Gripper open | Gripper opens at `releasing_time` fraction of throw |
| **FLIGHT** | ≤200 | `env.sim.step()` | Wait for drink to settle (velocity < 0.05 m/s for 30 steps) |

Total inner simulation steps per outer RL step: ~300–700 (varies with `duration`).

#### Pre-Position Cache

On the **first** call to `execute_primitive_batched()`, the full SETTLE+APPROACH+DESCEND
sequence (220 steps) runs and caches: arm joint positions after DESCEND, drink settled
position, and crane EE position. On **subsequent** calls, the drink is spawned at the
cached settled position and the arm joints are written directly to the grasp-ready
configuration, followed by 10 stabilization steps. This skips ~220 steps per episode
(~1.8 s at 120 Hz), giving approximately 2–3× speedup after the first episode.

Pass `grasp_cache={}` to enable; `grasp_cache=None` to disable.

#### Drop Detection

The drink's env-local Z position is checked against `DRINK_BELOW_TABLE_Z = 0.45`
after every sim step in **all** phases (GRASP, LIFT, GO_TO_INIT, GO_TO_INITIAL,
THROW, FLIGHT). When a drop is detected:

1. Last valid position is recorded in `drop_pos`
2. Drink is despawned (teleported 10 m below scene) to prevent physics interference
3. Kinematic hold is skipped for that env
4. If all envs drop, the current phase loop exits early
5. After FLIGHT, dropped drinks are restored to `drop_pos` so observations see
   a physically plausible position (not the despawned location)
6. `distances[dropped] = DROP_PENALTY_DISTANCE (10.0)` assigns the penalty

This prevents dropped drinks from poisoning the SAC replay buffer with extreme
outlier positions that would corrupt the `RunningStandardScaler` and critic Q-values.

#### Gripper Consistency (`GRASP_STR`)

The gripper hold strength is `GRASP_STR = 0.48` (calibrated at 65 N via force
feedback — see `test_joint_throwing.py`). This value is used consistently across
**all** phases: GRASP ramp (0→0.48), LIFT, GO_TO_INIT, GO_TO_INITIAL, and THROW
(before release). Previously, post-GRASP phases used 0.7 (46% over-tightening),
which caused the gripper to squeeze the bottle harder than intended.

#### Kinematic Hold During Throw

From GRASP through THROW (until release), the drink is kinematically attached
to the EE via pose write. The offset between EE and drink is recorded at the
moment of grasping and maintained until the gripper opens:

```python
_grasp_offset = milk_pos - ee_pos  # recorded at grasp moment
# Each step while holding:
milk.write_root_pose_to_sim(ee_pos + _grasp_offset, ...)
```

#### Joint Presets (from Gazebo)

```python
# Right arm
RIGHT_INIT_JOINTS = [1.6, -1.7236, 2.3313, -2.0629, -1.5987, 0.0]
RIGHT_END_JOINTS  = [1.6, -1.2774, 0.8647, -2.1966, -1.5744, 0.0]

# Left arm (mirrored)
LEFT_INIT_JOINTS = [-1.6, -1.435, -2.3313, -1.0, 1.5987, 0.0]
LEFT_END_JOINTS  = [-1.6, -1.881, -0.8647, -0.9, 1.5744, 0.0]
```

The SAC agent overrides `shoulder_pan` (index 0) via `initial_joint_value`
and `final_joint_value`. For the left arm, `initial_joints_pose[1]` is also
overridden to `-1.3` (matching Gazebo's `new_impl.cpp` hardcoded value).
All other joints follow the Gazebo presets, which encode the catapult motion
(shoulder lift up + elbow extend).

#### EE 90° Rotation (Both Arms)

Both arms' end-effectors are rotated 90° around the EE's local Z-axis
(wrist_3 rotation) throughout all phases. This aligns the gripper fingers
perpendicular to the drink bottle for reliable grasping.

**Initial config** (`throwing_env_cfg.py`): Both arms start with
`wrist_3_joint = 1.5708` (π/2) in the robot's `init_state.joint_pos`.
The robot resets to this pose every episode, so the EE is already rotated
from the start. The IK phases use position-only control (zero orientation
delta), which naturally preserves the initial orientation.

**Joint-space phases** (GO_TO_INIT, GO_TO_INITIAL, THROW): The Gazebo joint
presets have `wrist_3 = 0.0`. The constant `EE_YAW_OFFSET = π/2` is added
to `wrist_3` (index 5) in all joint targets to maintain the rotation:

```python
EE_YAW_OFFSET = math.pi / 2
init_joints_rotated[5] += EE_YAW_OFFSET
initial_joints_pose[:, 5] += EE_YAW_OFFSET
end_joints_pose[:, 5] += EE_YAW_OFFSET
```

This ensures the gripper maintains the 90° rotation continuously from
approach through throw and return.

#### Joint Limit Clamping

Joint targets are clamped to hardware position limits before execution,
matching Gazebo's implicit clamping by the ROS trajectory controller.
`build_joint_targets()` accepts an optional `joint_pos_limits` tensor
(shape `(6, 2)` or `(N, 6, 2)`) and applies `torch.clamp(targets, lower, upper)`
to both `initial_joints_pose` and `end_joints_pose`:

```python
arm_limits = robot.data.joint_pos_limits[0, arm_joint_ids, :]  # (6, 2)
initial_joints_pose, end_joints_pose = build_joint_targets(
    init_joints, initial_jv, end_joints_base, final_jv,
    side=side, joint_pos_limits=arm_limits,
)
```

This ensures that RL-generated shoulder_pan values outside the UR5e's
physical range (typically ±2π) are silently clamped rather than producing
undefined behavior in the PD controller.

### ThrowingPrimitiveEnv (`tasks/throwing_primitive_env.py`)

A `gymnasium.Env` subclass that wraps `ThrowingEnv`:

- **Action space**: `Box(4)` — 4 macro throw parameters
- **Observation space**: `Box(8)` — target/object positions + distances
- **Episode**: Always 1 step (terminated=True after every `step()`)
- **Internal env**: `ThrowingEnv` with `disable_attachment=True`, DiffIK solver,
  60s timeout, no terminations except time_limit

Key design choices:
- `release_vel_threshold=inf` prevents the base env's auto-release logic
- `_holding=False` and `_released=False` prevent unwanted settled termination
- IK scale set to 0.8 for faster convergence during approach phases
- Target randomization: X ∈ [0.0, 0.45], Y ∈ [1.0, 1.4] (right-side forward)

### Batched Execution (`execute_primitive_batched`)

All N environments execute the same phase simultaneously. The throw phase handles
per-env variations in `duration` and `releasing_time`:

- `throw_steps = clamp(duration / sim_dt, 12, 120)` — per-env throw step count
- `release_step = releasing_time × throw_steps` — per-env release timing
- The loop runs for `max(throw_steps)` across all envs
- Envs that finish their throw early hold their final joint targets
- Release is per-env: `should_release = (step >= release_step) & active & ~released`

### SAC Agent Configuration (`skrl_sac_cfg.yaml`)

| Component | Architecture |
|-----------|-------------|
| **Policy** | GaussianMixin, MLP [256, 256], ReLU, clipped log_std ∈ [-20, 2] |
| **Critic 1 & 2** | DeterministicMixin, MLP [256, 256], ReLU, input = [STATES, ACTIONS] |
| **Target Critic 1 & 2** | Same architecture as critics, Polyak-averaged |

| Hyperparameter | Value |
|----------------|-------|
| Replay buffer | RandomMemory, size 100,000 |
| Batch size | 256 |
| Discount (γ) | 0.99 |
| Polyak (τ) | 0.005 |
| Learning rate | 3.0e-4 |
| Random timesteps | 1,000 (pure exploration before learning) |
| Learning starts | 1,000 |
| Gradient norm clip | 1.0 |
| Learn entropy (α) | True (auto-tuned) |
| Initial entropy | 1.0 |
| Target entropy | -2.0 |
| Observation preprocessor | RunningStandardScaler |
| Total timesteps | 35,000 |
| Trainer | SequentialTrainer |
| Checkpoint interval | 1,000 steps |

### SAC vs PPO Comparison

| Aspect | PPO (per-step) | SAC (macro-action) |
|--------|----------------|-------------------|
| Action dim | 6 (EE deltas) | 4 (throw params) |
| Obs dim | 27 | 8 |
| Episode length | 600 steps (5s) | 1 outer step |
| Inner sim steps/action | 1 | ~400–800 |
| Release mechanism | Velocity threshold | Explicit `releasing_time` parameter |
| Agent controls | Continuous EE trajectory | Throw shape (wind-up, release, power) |
| Training timesteps | 300,000 | 35,000 |
| Network size | [512, 256, 128] | [256, 256] |
| Off-policy | No | Yes (replay buffer) |
| Exploration | PPO clipping | Entropy-regularized (auto α) |

### Why SAC for This Task

1. **Sample efficiency**: SAC is off-policy — each throw experience is stored
   in a replay buffer and reused many times. With only 35K timesteps (throws),
   the agent can learn an effective policy. PPO would discard experiences after
   each update.

2. **Low-dimensional structured action**: The 4-parameter action space is small
   and continuous — ideal for SAC's Gaussian policy. The agent doesn't need to
   discover the throw trajectory shape (that's hardcoded from Gazebo); it only
   learns the optimal shoulder_pan angles, timing, and power.

3. **One-shot episodes**: Each episode is a single throw. There's no sequential
   decision-making or temporal credit assignment — the reward is immediate.
   SAC's Q-function only needs to learn Q(state, throw_params) → expected reward.

4. **Entropy regularization**: The auto-tuned entropy coefficient encourages
   exploration of different throw parameters early in training, then gradually
   exploits the best combinations as learning progresses.

### Training CLI

```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
cd /home/vladi/IsaacLab/master_isaac/throwing_enviroment

# Basic SAC training (64 parallel envs)
python scripts/train_sac.py --headless --num_envs=64

# More envs, full training
python scripts/train_sac.py --headless --num_envs=128 --max_iterations=35000

# Resume from checkpoint
python scripts/train_sac.py --headless --num_envs=64 \
    --checkpoint=logs/skrl/throwing_primitive/.../agent_5000.pt

# Custom seed
python scripts/train_sac.py --headless --num_envs=64 --seed=123
```

Logs are saved to `logs/skrl/throwing_primitive/<timestamp>_sac_torch/` with
TensorBoard events and agent checkpoints every 1000 steps.

### TensorBoard Custom Metrics

In addition to skrl's built-in metrics (reward, loss, Q-values, entropy),
`ThrowingPrimitiveEnv` logs custom metrics via its own `SummaryWriter`
(set up by `env.set_log_dir(log_dir)` in `train_sac.py`):

| Tag | Description |
|-----|-------------|
| `Throw / Mean Distance` | Mean landing distance to target (non-dropped envs) |
| `Throw / Min Distance` | Best throw distance in batch |
| `Throw / Max Distance` | Worst throw distance in batch |
| `Throw / Success Rate` | Fraction of envs with distance < 0.15 m |
| `Throw / Drop Rate` | Fraction of envs where drink was dropped |
| `Throw / Mean Reward` | Mean reward across all envs |
| `Action / Mean Initial JV` | Mean initial shoulder_pan (radians) |
| `Action / Mean Final JV` | Mean final shoulder_pan (radians) |
| `Action / Mean Release Time` | Mean release fraction |
| `Action / Mean Duration` | Mean throw duration (seconds) |

### Plotting Training Logs (`scripts/plot_tb_logs.py`)

Reads TensorBoard event files and generates a 6-panel diagnostic figure
(reward, distance, entropy coefficient, policy loss, critic loss, Q-values)
plus CSV export:

```bash
python scripts/plot_tb_logs.py                    # auto-finds latest run
python scripts/plot_tb_logs.py logs/skrl/throwing_primitive/<run>/
```

For runs with the custom `Throw / Mean Distance` tag, the distance panel
shows the actual logged values. For older runs, distance is numerically
inverted from the reward function.

### Training Log Output

Each outer step (= one throw per env) prints:
```
[Ep    128] reward=0.0312  dist=0.847m  success=0/64  dropped=3/64
  env[0]: target=(0.221,1.156,0.700) action=[ijv=1.423 fjv=0.891 rel=0.450 dur=0.532] obj=(0.412,0.823,0.600) dist=0.391 rew=0.0001
```

## DirectRLEnv — Fast Training (`tasks/throwing_direct_env.py`)

### Motivation

The original SAC pipeline used a `gymnasium.Env` wrapping a `ManagerBasedRLEnv`.
Each throw required ~300-700 calls to `env.step()`, each triggering 5 managers
(Action, Observation, Reward, Termination, Event). This Python overhead dominated
training time, especially at high env counts.

The DirectRLEnv refactor eliminates all manager overhead by implementing the
throw as a state machine inside `_apply_action()` — called once per physics
sub-step with no Python dispatch.

### Architecture

```
ThrowingDirectEnv(DirectRLEnv)
  ├── decimation = 320 (one outer step = full throw)
  ├── replicate_physics = True (GPU-batched vectorized physics)
  ├── _pre_physics_step(actions): decode 4D macro-action
  ├── _apply_action() × 320: state machine (STABILIZE→THROW→FLIGHT)
  ├── _get_rewards(): distance-based (computed before reset)
  ├── _get_dones(): always terminated (one-shot)
  └── _reset_idx(): randomize target, place drink at EE
```

### Phase State Machine (inside `_apply_action`)

| Phase | Sub-steps | Description |
|-------|-----------|-------------|
| **STABILIZE** | 0-9 | Robot at crane, drink placed 25cm below EE, gripper held |
| **GO_TO_INIT** | 10-29 | Move arm to Gazebo `init_joints_pose`, kinematic hold |
| **GO_TO_INITIAL** | 30-49 | Move to wind-up pose (agent's `initial_jv`), kinematic hold |
| **THROW** | 50-170 | Linear joint interpolation, release at `releasing_time` |
| **FLIGHT** | 170-319 | Zero actions, drink flies free |

Total: 320 physics sub-steps per outer step (~2.67s sim time at 120 Hz).

### Key Differences from ManagerBased Path

| Aspect | ManagerBased (legacy) | DirectRLEnv (fast) |
|--------|----------------------|-------------------|
| Base class | `gymnasium.Env` → `ThrowingEnv(ManagerBasedRLEnv)` | `DirectRLEnv` |
| IK solver | DiffIK action term (live IK every step) | None (drink kinematically attached from start) |
| Managers | 5 managers × 300-700 calls/throw | 0 (bare `sim.step()`) |
| `replicate_physics` | False | True |
| Pickup phases | SETTLE+APPROACH+DESCEND+GRASP+LIFT (300 steps) | Skipped (drink placed at EE at reset) |
| Env creation | ~3-5s (URDF→USD conversion) | ~0.6s (pre-built USD or cached) |
| GPU parallelism | 2048 envs max | 4096+ envs |
| Per-throw time | ~3-5s (2048 envs) | ~0.8s (4096 envs) |

### Release Velocity Injection

At the moment of release, the drink inherits the EE's linear and angular
velocity, plus a 10cm offset in the velocity direction to clear arm geometry:

```python
ee_lin_vel = robot.data.body_lin_vel_w[release_ids, ee_body_ids[0]]
vel_dir = ee_lin_vel / vel_norm.clamp(min=0.01)
release_pos = milk_pos + vel_dir * 0.10  # clear arm
milk.write_root_pose_to_sim(release_pos + quat, ...)
milk.write_root_velocity_to_sim([ee_lin_vel, ee_ang_vel], ...)
```

### Pre-baked Robot USD (`scripts/prebake_robot_usd.py`)

One-time conversion of `urdf/dual_arm_robot.urdf` → `assets/robot/dual_arm_robot.usd`.
The DirectRLEnv auto-detects the USD file; if absent, falls back to runtime URDF
conversion. Pre-baking saves ~3-5s per env creation.

```bash
python scripts/prebake_robot_usd.py  # one-time, saves to assets/robot/
```

### Training CLI (DirectRLEnv)

```bash
# Local training (DirectRLEnv, fast)
python scripts/train_sac.py --headless --num_envs=4096 --max_iterations=100000

# Visual test (DirectRLEnv)
python scripts/test_throw.py --direct --loop

# HPC (4096 envs, auto-resume)
cd throwing_enviroment && sbatch hpc/train_sac_direct.slurm
```

### `test_throw.py --direct`

Uses the DirectRLEnv for visual debugging. Each throw:
1. `env.reset()` → robot to crane, drink at EE, target randomized
2. `env.step(action)` → full 320-step throw executes
3. Results read from `env._last_distances`, `env._last_milk_pos`

Renders at 60 FPS (`render_interval=2`) so the throw animation is visible.

## Validation (`scripts/validate_throw.py`)

Deterministic evaluation of a trained SAC checkpoint against 10 predefined
target positions. Runs each target 3 times (configurable) and reports per-test
pass/fail plus overall success rate.

Two modes:
- **Default** (ManagerBased): Full IK grasping + throw primitive, slower but visual
- **`--fast`** (DirectRLEnv): No IK, drink at EE from start, 3-5× faster

```bash
# Standard validation (ManagerBased path)
python scripts/validate_throw.py \
    --checkpoint logs/skrl/throwing_primitive/.../checkpoints/agent_6000.pt \
    --num_tests 10 --attempts 3 --headless

# Fast validation (DirectRLEnv — recommended for sweeps)
python scripts/validate_throw.py --fast \
    --checkpoint .../agent_6000.pt --num_tests 10 --attempts 3 --headless

# Visual mode (watch throws)
python scripts/validate_throw.py \
    --checkpoint .../agent_6000.pt --num_tests 5 --attempts 3

# Custom threshold
python scripts/validate_throw.py --checkpoint .../agent_6000.pt --success_threshold 0.20
```

**Test configurations** (`tasks/throw_validation_configs.py`):

| # | Target | Description |
|---|--------|-------------|
| 1 | (0.00, 1.10) | Center near |
| 2 | (0.00, 1.50) | Center far |
| 3 | (0.40, 1.10) | Right near |
| 4 | (0.40, 1.50) | Right far |
| 5 | (0.20, 1.30) | Center-right mid |
| 6 | (0.10, 1.10) | Slight-right near |
| 7 | (0.30, 1.40) | Right mid-far |
| 8 | (0.45, 1.20) | Far-right near |
| 9 | (0.05, 1.55) | Center far edge |
| 10 | (0.25, 1.00) | Center-right closest |

**Output**:
- Per-test results with action params, landing distance, pass/fail
- Summary table with total success rate and average distances
- **Plot** saved to `logs/validation_results.png`:
  - Left panel: bird's-eye scatter (targets + landings, color-coded)
  - Right panel: bar chart of best distance per test with threshold line

**Model loading**: Uses skrl's Runner to recreate the exact model architecture
from `skrl_sac_cfg.yaml`, then loads weights from the checkpoint file. Policy
is called with `agent.policy.act({"states": obs})` in eval mode.

## HPC Deployment (`hpc/train_sac_direct.slurm`)

SLURM job scripts for training on the Hábrók HPC cluster with Apptainer (Singularity):

```bash
cd /path/to/master_isaac/throwing_enviroment

# DirectRLEnv (fast, 4096 envs — recommended)
sbatch hpc/train_sac_direct.slurm

# ManagerBased (legacy, 2048 envs)
sbatch hpc/train_sac.slurm
```

**Features**:
- Apptainer container (`isaac-lab.sif`) with all dependencies
- SIGUSR1 trap for automatic resubmission on time-limit
- Local scratch (`$TMPDIR`) for fast I/O, synced to NFS on exit
- Auto-resume from latest checkpoint via `RESUME_CHAIN=1`
- Warp/matplotlib cache redirected to writable dirs
- Isaac Sim kit data/cache/logs bound to project-local dirs

**Key configuration** (edit at top of slurm file):
```bash
NUM_ENVS=2048
MAX_ITERATIONS=100000
SIF_IMAGE="$(dirname "$PROJECT_ROOT")/asyncDualPlayPPO/isaac-lab.sif"
```

**Container bind mounts**:
```
$PROJECT_ROOT          → /workspace/isaaclab/user_project/throwing_enviroment
$LOCAL_LOGS            → .../logs/skrl/throwing_primitive (output)
$PROJECT_ROOT/.cache   → /root/.cache (warp kernel cache)
.isaac_cache/kit/*     → /isaac-sim/kit/* (Isaac Sim caches)
```

**Ground plane**: Uses a procedural `CuboidCfg` (10×10×0.01m) instead of
`GroundPlaneCfg` which downloads a USD from Omniverse's cloud CDN — HPC
compute nodes have no internet access.

## Self-Contained Project

The throwing_enviroment directory is fully self-contained — no external
dependencies except the `.sif` container image:

```
throwing_enviroment/
├── ik_solvers/              # Local copy (from pingpong_dual_arm)
├── urdf/
│   └── dual_arm_robot.urdf  # Dual-arm robot URDF
├── assets/urdf/ur_robotics/ur5e/
│   └── ur5e_robotiq_140.urdf # Single-arm URDF
└── meshes/                   # Robot mesh files
    ├── ur5e/                 # UR5e visual + collision meshes
    ├── intuition_body/       # Robot body visuals
    ├── dual_arm_head/        # Head with headphone
    └── robotiq_2f_gripper_description/  # Gripper meshes
```

Previously, `throwing_env_cfg.py` imported from `pingpong_dual_arm/` via
`sys.path`. Now all paths use `_PKG_ROOT` (the throwing_enviroment directory).

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
- Creates a `UsdShade.Material` prim (`/root/HighFrictionMaterial`) with
  `UsdPhysics.MaterialAPI` (`static_friction=5.0`, `dynamic_friction=5.0`,
  `restitution=0.1`) and binds it to all meshes via `UsdShade.MaterialBindingAPI`
  with purpose `"physics"`. This is the correct USD pattern for physics materials
  — applying `PhysxSchema.PhysxMaterialAPI` directly on mesh prims does NOT work.

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
| basket | `UsdFileCfg` | `assets/new_usds/basket_02/model_basket1.usd` (scale 0.4) | 2.0 kg | Yes |
| table | `CuboidCfg` | — (procedural, 2.0×1.7×0.05 m) | — | Yes |
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

1. **IK solvers are now local**: The `ik_solvers/` module was copied from
    `pingpong_dual_arm/ik_solvers/` into the throwing_enviroment directory.
    No external dependency on pingpong_dual_arm remains. URDFs and meshes
    are also local copies.

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

15. **Sweep functionality removed**: The `--sweep` CLI argument, `run_sweep()`,
    `_parse_sweep()` and the sweep dispatch block were removed from
    `test_ik_throwing.py`. Use manual parameter tuning or the `--compare`
    multi-solver benchmark instead.

16. **Standalone `DifferentialIKController` fails with dual-arm robot**: Using
    `robot.root_physx_view.get_jacobians()[:, ee_jac_idx, :, arm_ids]` and
    calling `DifferentialIKController.compute()` returns zero-delta joint
    positions (output == input). The Jacobian column indexing likely doesn't
    correctly map to the right arm's DOFs in the full 24-DOF articulation.
    **Workaround**: Use the env's built-in DiffIK action term via `env.step()`
    which handles the Jacobian mapping internally.

17. **PhysX FixedJoint cannot be created at runtime with Fabric**: Calling
    `UsdPhysics.FixedJoint.Define(stage, path)` and setting body targets via
    `GetBody0Rel().SetTargets(...)` during an active simulation produces a
    PhysX warning "disjointed body transforms" and the constraint has no
    physical effect. The Fabric backend caches physics state in GPU memory
    and does not pick up USD-level topology changes (new joints/bodies)
    without a full scene reset. **Workaround**: Use physics gripper friction
    (with high-friction material) or kinematic pose write instead of fixed
    joints.

18. **PD actuators cannot generate catapult-speed motion for high-inertia
    joints**: With implicit actuator stiffness=8000 and damping=500, the
    shoulder_lift and elbow joints (which drive heavy links) have PD response
    times of ~0.5-1.0s. A Gazebo-style catapult (0.44 rad shoulder + 1.47 rad
    elbow in 0.33s) is physically unreachable — the arm barely moves. The
    wrist_2 snap in `test_ik_throwing.py` works because wrist inertia is
    orders of magnitude lower. Options: (a) increase stiffness to 50000+ for
    throw phase, (b) use effort/velocity control mode during throw, (c) accept
    longer throw durations (200+ steps), (d) use wrist-only snap with arm
    pre-positioning (proven approach).

19. **Drink friction requires proper USD material binding**: Applying
    `PhysxSchema.PhysxMaterialAPI` directly to mesh prims does NOT set
    effective contact friction. PhysX requires a `UsdShade.Material` prim
    with `UsdPhysics.MaterialAPI` attributes, bound to the mesh via
    `UsdShade.MaterialBindingAPI` with purpose `"physics"`. Without this,
    the drink retains its source model's low friction (0.5/0.4) despite the
    prebake script appearing to set 5.0/5.0. The scene's
    `friction_combine_mode="max"` then yields only `max(1.0, 0.5) = 1.0`
    effective friction — insufficient for reliable grip during fast motions.

## Stack

| Component | Version | Location |
|-----------|---------|----------|
| Isaac Sim | 5.1.0-rc.19 | `/home/vladi/isaac-sim-5.1.0/` |
| Isaac Lab | 0.54.3 (2.3.2) | `/home/vladi/IsaacLab/source/isaaclab/` |
| Python | 3.11 | `/home/vladi/IsaacLab/master_isaac/.master_venv` |
| PyTorch | 2.7.0+cu128 | GPU |
| skrl | >= 1.4.2 | `.master_venv` |
