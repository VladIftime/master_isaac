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
│   ├── test_ik_throwing.py                # IK benchmark (YZ arc trajectories, visual markers, indefinite loop)
│   ├── test_throw.py                      # Single-throw test (kinematic hold → release → land, --loop flag)
│   ├── convert_meshes.py                  # OBJ → USD with MeshConverter
│   ├── prebake_physics.py                 # Apply CollisionAPI + PhysicsMaterial to USD meshes
│   ├── prebake_drink.py                   # Pre-bake drink001: MassAPI, CollisionAPI, high friction
│   ├── prebake_basket.py                  # Pre-bake shopping basket: single rigid body, handles removed
│   ├── export_full_scene.py               # Export scene USD + joint pose script for Isaac Sim GUI
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
│   ├── full_scene.usd                     # Reference-based full scene (robot + drink + basket + table + stand)
│   ├── joint_pose.json                    # 24 joint angles for crane home pose
│   ├── pose_robot.py                      # Kit script: sets drive targets + auto-plays simulation
│   ├── robot_crane_pose/                  # Robot USD cache copy (referenced by full_scene.usd)
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
- **Drink object**: Spawned at the right wrist_3_link at episode reset, kinematically attached until release via `write_root_pose_to_sim`. Visual mesh loaded via `UsdFileCfg` from `assets/new_usds/drink001/drink_target.usd` (Synthesis drink bottle with pre-baked MassAPI, CollisionAPI, and high-friction PhysxMaterial). Dynamic, mass 0.5 kg. Bottle root offset `(-0.012, 0.129, -0.176)` from wrist positions the bottle center between the finger pads.
- **Target (shopping basket)**: Kinematic shopping basket, randomized in XY on the table surface. Visual mesh loaded via `UsdFileCfg` from `assets/new_usds/shopping basket002/basket_target.usd` (Synthesis basket with handles removed, single rigid body). Mass 2.0 kg. Size ~0.52×0.38×0.26 m.
- **Arm initial pose**: Right arm starts at a home pose (`shoulder_lift=-1.57, elbow=1.57, wrist_1=-1.57, wrist_2=-1.57, wrist_3=0.0`). Left arm is idle. Gripper starts **open** (`finger_joint=0.0`).

## Running

```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
cd /home/vladi/IsaacLab/master_isaac/throwing_enviroment

# Test environment (verify scene loads)
python scripts/test_env.py --ik diffik --num_envs 4

# Export full scene for Isaac Sim GUI
python scripts/export_full_scene.py

# Single throw test (visual mode)
python scripts/test_throw.py --ik diffik --amp 0.3

# Single throw (headless)
python scripts/test_throw.py --ik diffik --headless --amp 0.3

# Indefinite throws (Ctrl+C to stop)
python scripts/test_throw.py --ik diffik --loop

# IK solver benchmark with visual path markers
python scripts/test_ik_throwing.py --ik diffik --trajectory arc

# Compare all 4 IK solvers, save to CSV
python scripts/test_ik_throwing.py --compare diffik:osc:rmpflow:curobo --output metrics.csv

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
bottle center between the finger pads:

```python
bottle_root_offset = torch.tensor([-0.012, 0.129, -0.176], device=ee_pos.device)
bottle_root = ee_pos[still_holding] + bottle_root_offset.unsqueeze(0)
ee_pose = torch.cat([bottle_root, ee_quat[still_holding]], dim=-1)
milk.write_root_pose_to_sim(ee_pose, env_ids=still_ids)
```

The bottle appears between the gripper fingers visually. When the release
condition is met, the pose write stops. The object inherits its last-frame
velocity from PhysX's position-derivative velocity computation and flies free.

**Release condition**: After `release_min_steps` warmup steps (default 10) AND
when EE linear velocity magnitude exceeds `release_vel_threshold` (default 2.0 m/s):

```python
release_mask = (
    self._holding & ~self._released
    & (self._steps_in_episode > self.cfg.release_min_steps)
    & (vel_norm > self.cfg.release_vel_threshold)
)
```

On release, the gripper opens (`finger_joint = 0.0`) for visual feedback.

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

Grippers: finger_joint = 0.0 (open) at reset, closed to 0.7 during attach
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

Three benchmark trajectories in the **YZ plane** (forward + upward throwing motion):

| Trajectory | Description | Parameters |
|-----------|-------------|-----------|
| **Arc** | YZ parabolic arc: Y sweeps forward, Z peaks mid-arc | `amp=0.15` m, `period=60` steps |
| **Linear punch** | Straight line forward-upward with parabolic Z | `amp=0.15` m, `period=60` steps |
| **Sinusoidal lob** | Y linear forward, Z sine arc | `amp=0.15` m, `period=60` steps |

X stays at initial EE position (no lateral swing). Visual markers:
- **Path spheres** (cyan, 31 markers): Full planned EE trajectory
- **Release point** (red, r=0.03): Midpoint of trajectory
- **Target marker** (green, r=0.04): Current target position from scene
- **EE current** (yellow, r=0.015): Actual EE position (updates every 10 steps)
- **Desired EE** (orange, r=0.012): Target EE position (updates every 10 steps)

Markers use `isaaclab.markers.VisualizationMarkers` (same pattern as pingpong's `test_ik_swing.py`).

Per-solver metrics (saved to CSV with `--output`):
- Mean/max position error (cm)
- Mean/max orientation error (deg)
- Mean joint jerk (rad/s³) — smoothness proxy
- Total steps completed

```bash
# Compare all solvers
python scripts/test_ik_throwing.py --compare diffik:osc:rmpflow:curobo --output metrics.csv

# Single solver with visual markers
python scripts/test_ik_throwing.py --ik curobo --trajectory lob --period 40
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
for inspection and editing in the Isaac Sim GUI:

```bash
python scripts/export_full_scene.py
```

**Output files in `generated_usd/`:**

| File | Description |
|------|-------------|
| `full_scene.usd` | Reference-based scene with robot (URDF cache), drink, basket, table (box), stand (box), ground plane. All entities at live simulation positions. |
| `joint_pose.json` | 24 joint angles captured from the crane home pose |
| `pose_robot.py` | Kit script that sets joint drive targets and auto-plays simulation |
| `robot_crane_pose/` | Copy of the URDF-converted robot USD with all `configuration/` sub-files |

**To export a fully flattened, posed scene:**

1. Open Isaac Sim GUI
2. File > Open > `full_scene.usd`
3. Window > Script Editor > File > Open Script > `pose_robot.py` > Run
4. The script sets all 24 joint drive targets then auto-plays the simulation.  
   Joints animate to crane pose. Press Stop when settled.
5. File > Export > USD (check **Flatten**) to save the posed single-file scene

**Editing finger grippers in the exported scene:**

1. In the Stage tree, find:  
   `World/Robot/.../rgripper_left_inner_finger` and `rgripper_right_inner_finger`
2. Select the collision child prim → Property panel
3. To enlarge pads: increase the collision box depth from `0.0075` to `0.03`
4. To add friction: Property > Physics Material:  
   `Static Friction = 5.0`, `Dynamic Friction = 5.0`, `Restitution = 0.1`
5. File > Save As > `robotiq_gripper_modified.usd`

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
   too thin (7.5mm) and getting merged through the URDF importer.

4. **Bottle root offset is world-frame**: The offset `(-0.012, 0.129, -0.176)`
   placing the bottle between the finger pads is computed in world coordinates at
   the home pose. If the arm pose changes significantly, this offset would need to
   be recomputed in the EE's local frame using quaternion rotation.

5. **Table is kinematic**: The table is a kinematic cuboid at the same height as
   the robot stand (z=0.6). The basket target sits on the table surface. The
   drink bottle can land on or bounce off the table surface.

6. **Observation side hardcoded**: Observation terms currently reference
   `right_wrist_3_link` and `ARM_JOINTS_RIGHT`. Switching `playing_arm_side`
   to `"left"` only changes the action arm.

7. **`write_root_velocity_to_sim` unreliable**: Attempts to set explicit release
   velocity on the bottle had limited/no effect. The object's post-release velocity
   is dominated by the pose-teleport artifacts from the holding phase. The
   `test_throw.py` script has been stripped of all velocity writes.

8. **cuRobo CUDA graph**: The cuRobo solver uses `use_cuda_graph=True` which
    requires CUDA ≥ 12.0 for graph resets. See pingpong implementation.md for
    full details on the cuRobo IK solver integration.

9. **Robot joint baking is not direct**: The exported `full_scene.usd` references
    the robot in its URDF default pose. Joint positions from the Fabric stage
    cannot be physically baked into the USD without a full FK chain computation.
    Use `pose_robot.py` in the Isaac Sim GUI to drive joints to the crane pose,
    then export a flattened USD with the posed state.

10. **Fabric/USDRT stage cannot be exported programmatically**: `save_as_stage()`
    and `UsdUtils.FlattenLayerStack` both fail on Fabric-backed stages. The
    scene export works around this by building a reference-based USD from
    Isaac Lab data buffers (positions) and cached USD files (robot, assets).

## Stack

| Component | Version | Location |
|-----------|---------|----------|
| Isaac Sim | 5.1.0-rc.19 | `/home/vladi/isaac-sim-5.1.0/` |
| Isaac Lab | 0.54.3 (2.3.2) | `/home/vladi/IsaacLab/source/isaaclab/` |
| Python | 3.11 | `/home/vladi/IsaacLab/master_isaac/.master_venv` |
| PyTorch | 2.7.0+cu128 | GPU |
| skrl | >= 1.4.2 | `.master_venv` |
