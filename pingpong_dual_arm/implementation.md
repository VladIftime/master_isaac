# Ping Pong Dual-Arm Environment — Implementation

Competitive ping pong simulation with two identical dual-arm UR5e robot systems in
Isaac Lab / Isaac Sim. Each robot has a rendered body, head, two arms with Robotiq
grippers, and a racket attached to the playing arm's end-effector.

## Project Structure

```
pingpong_dual_arm/
├── ik_solvers/                          # Swappable IK solver builders
│   └── __init__.py
├── tasks/                               # Environment implementation
│   ├── __init__.py
│   ├── pingpong_env_cfg.py              # Scene + MDP config
│   ├── pingpong_env.py                  # Environment class (PingPongEnv)
│   ├── observations.py                  # Observation functions
│   ├── rewards.py                       # Reward functions
│   ├── events.py                        # Reset events (serve, joint reset)
│   └── terminations.py                  # Boundary checks
├── scripts/
│   ├── test_env.py                      # Launch & step environment
│   ├── pingpong_swing.py                # Swing demo with macro primitives
│   ├── test_ik.py                       # Compare IK solvers on reach tasks
│   ├── random_policy.py                 # Random action rollout
│   └── convert_meshes.py                # STL → USD with MeshConverter
├── meshes/                              # All 3D assets
│   ├── dual_arm_head/                   # Robot head (OBJ)
│   ├── intuition_body/                  # Robot body (OBJ/DAE/STL)
│   ├── pingpong/                        # Table STL, racket STL, ball
│   ├── robotiq_2f_85_gripper_visualization/
│   ├── robotiq_2f_gripper_description/
│   └── ur5e/                            # UR5e arm collision/visual
├── assets/
│   ├── urdf/ur_robotics/ur5e/           # Single-arm URDF files
│   └── pingpong/                        # Generated USD files
├── urdf/
│   ├── dual_arm_robot.urdf              # Full dual-arm + body + head
│   ├── dual_arm_robot_no_gripper_col.urdf
│   ├── dual_arm_robot_no_gripper_col.usd
│   ├── config.yaml                      # URDF→USD converter config
│   ├── configuration/                   # Generated USD configs
│   └── cuMotion/                        # RMPflow + Lula IK configs
├── cfg/task/
│   └── pingpong_dual_arm.yaml           # Task hyperparameters
└── implementation.md
```

## Scene Layout

```
                    Robot B (Y=+1.87)
                +---[BODY]---+
                | left  right |
                | arm   arm   |
                |             |
                |  [RacketB]  |
                |             |
          =======|=====|==========  Table (1.525 × 2.74 m, green)
                |             |
                |  [RacketA]  |
                |             |
                | arm   arm   |
                | left  right |
                +---[BODY]---+
                    Robot A (Y=-1.87)
                        +Y →
```

- **Table**: Mesh from `table.stl` via MeshConverter, at origin, kinematic
- **Robot A**: at Y=-1.87, facing +Y (toward table), right arm is playing arm
- **Robot B**: at Y=+1.87, rotated 180° about Z (facing -Y), right arm is playing arm
- **Rackets**: Kinematic meshes from `racket.stl`, tracked to `right_wrist_3_link` each step
- **Ball**: 0.02 m radius sphere, 0.0027 kg, orange, restitution 0.95

## Running

```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
cd pingpong_dual_arm

# Test environment
python scripts/test_env.py --ik diffik --num_envs 4 --steps 500 --headless

# Swing demo
python scripts/pingpong_swing.py --ik diffik --steps 2000

# Compare IK solvers
python scripts/test_ik.py --solver osc
```

The venv provides `torch` and `isaaclab`. Three torch submodules (`_dynamo`,
`_C`, `optim`) must be imported BEFORE `isaaclab.app` to lock against Isaac
Sim's incompatible pip_prebundle.

## IK Solvers (`ik_solvers/`)

Builder functions that return `ActionTermCfg` for a specific arm on a specific
robot. Each solver uses `use_relative_mode=True` with `command_type="pose"`,
producing 6D relative delta actions [dx, dy, dz, droll, dpitch, dyaw].

| Solver | Isaac Lab Class | Controller | Key Param |
|--------|----------------|------------|-----------|
| `diffik` | `DifferentialInverseKinematicsActionCfg` | Damped Least Squares | `lambda_val=0.1` |
| `osc` | `OperationalSpaceControllerActionCfg` | Variable-KP impedance | `stiffness=[360]×6` |
| `rmpflow` | `RMPFlowActionCfg` | Lula GPU motion policies | YAML configs |
| `curobo` | Falls back to DiffIK | GPU nonlinear IK | Requires `curobo` |

Joint name patterns per side:
- Right arm: `right_shoulder_.*`, `right_elbow_.*`, `right_wrist_.*`
- Left arm:  `left_shoulder_.*`,  `left_elbow_.*`,  `left_wrist_.*`

End-effector body: `right_wrist_3_link` / `left_wrist_3_link`.

Swap the solver at config level:
```python
cfg = PingPongDualArmEnvCfg()
cfg.ik_solver = "osc"
```

## Environment Configuration (`tasks/pingpong_env_cfg.py`)

### Robot Articulation

Each robot spawns from `dual_arm_robot_no_gripper_col.urdf` via `UrdfFileCfg`
with runtime URDF→USD conversion. The URDF includes:
- **Body**: `IRL_lab_robot_body.obj` + `robot_body_back.obj`
- **Head**: `blue_head_with_headphone.obj`
- **Two UR5e arms**: 6-DOF each (shoulder_pan/lift, elbow, wrist_1/2/3)
- **Fixed joints** for body, head, and arm base connections

Actuators: position-controlled implicit (stiffness=5000, damping=200).
Two groups: `arm_left` and `arm_right`.

### Asset Scaling

Object position, rotation, and scale are applied directly in the environment
config's `init_state` and `spawn` parameters — no need to regenerate USD files
for adjustments:

```python
# Table: rotate 90° about Z to align mesh orientation
table = RigidObjectCfg(
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.0, 0.0, 0.0],
        rot=[0.707, 0.0, 0.0, 0.707],  # 90° Z-axis rotation
    ),
    spawn=UsdFileCfg(usd_path="...", scale=(0.001, 0.001, 0.001)),
)

# Racket: apply mm-to-m scale directly in spawn
racket_A = RigidObjectCfg(
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.0, -1.22, 0.60],
        rot=[1.0, 0, 0, 0],
    ),
    spawn=UsdFileCfg(usd_path="...", scale=(0.001, 0.001, 0.001)),
)
```

### Actions

```
Action space: Box (12,) per env
  arm_A (0:6):   [dx, dy, dz, droll, dpitch, dyaw]
  arm_B (6:12):  [dx, dy, dz, droll, dpitch, dyaw]
```

DiffIK with `use_relative_mode=True`, scale=0.15.

### Observations

```
Observation space: Box (45,) per env
  joint_pos_A (0:12):    left+right arm joints
  ee_pose_A   (12:18):   right_wrist_3_link pos(3)+Euler(3)
  joint_pos_B (18:30):   left+right arm joints
  ee_pose_B   (30:36):   right_wrist_3_link pos(3)+Euler(3)
  ball_state  (36:45):   ball pos(3)+lin_vel(3)+ang_vel(3)
```

Positions in environment-local coordinates. Rotations as ZYX Euler angles.

### Events

| Event | Mode | Function |
|-------|------|----------|
| `reset_all` | reset | `mdp.reset_scene_to_default` |

No automatic ball serve or joint resets — the swing script handles those.

### Terminations

Empty — no automatic episode termination. Ball respawn handled manually.

### Physics

```
sim.dt         = 0.02 s
decimation     = 1 (50 Hz)
episode_length = 20 s
use_fabric     = True (default)

GPU buffers:
  gpu_found_lost_pairs_capacity = 1M
  gpu_max_rigid_contact_count   = 1M
  gpu_max_rigid_patch_count     = 327K
```

## PingPongEnv Class (`tasks/pingpong_env.py`)

Overrides `step()` to track rackets to EE after physics update:

```python
class PingPongEnv(ManagerBasedRLEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # AFTER physics: robot arms have moved → snap rackets to wrists
        for robot_name, racket_name in [("robot_A", "racket_A"),
                                         ("robot_B", "racket_B")]:
            body_ids, _ = self.scene[robot_name].find_bodies(["right_wrist_3_link"])
            racket_pose = torch.cat([
                robot.data.body_pos_w[:, body_ids[0]],
                robot.data.body_quat_w[:, body_ids[0]],
            ], dim=-1)
            self.scene[racket_name].write_root_pose_to_sim(racket_pose)
        return obs, reward, terminated, truncated, info
```

Kinematic rackets are repositioned each frame with zero latency relative to
the arm, providing proper physics interaction with the ball.

## Swing Demo (`scripts/pingpong_swing.py`)

Each robot's right arm follows a sinusoidal X-axis trajectory. Robot A and B
are 180° out of phase (alternating swings).

### Swing Macro Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Amplitude | ±0.15 m | X axis only |
| Period | 60 steps | 1.2 s |
| Phase A | 0 rad | |
| Phase B | π rad | Alternating |
| Nominal Y | ±0.35 m | Over table |
| Nominal Z | 0.35 m | Above table |
| DiffIK gain | 0.3 | |

### Algorithm

```
for each step t:
    phase_A = 2π·t / PERIOD
    phase_B = phase_A + π

    target_A_x = AMP · sin(phase_A)
    target_B_x = AMP · sin(phase_B)

    read current EE positions
    delta = gain · (target - current)

    action = [dx_A, dy_A, dz_A, 0,0,0,  dx_B, dy_B, dz_B, 0,0,0]
    env.step(action)

    if ball.z < 0.05:
        respawn ball at center with random ±4 m/s + 4 m/s upward
```

### Ball

| Parameter | Value |
|-----------|-------|
| Spawn height | 0.60 m |
| Speed | 4 m/s horizontal, 4 m/s upward arc |
| Direction | ±Y (50/50) |
| Respawn | When z < 0.05 m |

## Mesh Assets & Conversion

### Source

Meshes under `meshes/` are from the original project. Ping pong assets:
- `pingpong/ping_pong_/ping_pong_table/table.stl` — table mesh
- `pingpong/racket.stl` — racket mesh
- `pingpong/semi_sphere.stl` — ball mesh (not used; procedural sphere instead)

### Conversion

`scripts/convert_meshes.py` converts STL → USD via `isaaclab.sim.converters.MeshConverter`:

```
MeshConverterCfg(
    asset_path="table.stl",
    usd_dir="assets/pingpong",
    make_instanceable=False,     # embed mesh directly (no broken refs)
    scale=(0.001, 0.001, 0.001), # mm → m
    rigid_props=(kinematic=True, disable_gravity=True),
    collision_props=CollisionPropertiesCfg(),
)
```

Output structure (verified):
```
/table [Xform]  RigidBody ✓  KINEMATIC
└─ /geometry/mesh [Mesh]  Collision ✓  PhysxCollision ✓
```

The `make_instanceable=False` flag embeds the mesh directly in the output USD,
avoiding broken cross-file references that occurred with instanced meshes.

When the mesh orientation or scale needs adjustment, modify the
`RigidObjectCfg.init_state.rot` and `UsdFileCfg.scale` in the env config
rather than regenerating the USD file:

```python
# Rotate the table 90° about Z at spawn time
init_state=RigidObjectCfg.InitialStateCfg(
    rot=[0.707, 0.0, 0.0, 0.707],  # quaternion 90° Z
)
```

## Observations Module

| Function | Dims | Description |
|----------|------|-------------|
| `robot_joint_positions` | (N, 12) | Left + right arm joint positions |
| `ee_poses` | (N, 6) | End-effector pos(3) + Euler ZYX(3) |
| `ball_state` | (N, 9) | Ball pos(3) + lin_vel(3) + ang_vel(3) |
| `ball_to_robot_relative` | (N, 3) | Ball pos relative to robot base |
| `ball_projected_state` | (N, 3) | Ball (x,y) + z_velocity |

Device handling: `env_origins` moved to `data.device` before subtraction.

## Events Module

| Function | Purpose |
|----------|---------|
| `serve_ball_random` | Place ball + set velocity |
| `reset_robot_joints` | Reset both robots to defaults |
| `reset_ball_to_serve` | Place ball at server's end |

## Termination Module

| Function | Check |
|----------|-------|
| `ball_out_of_bounds` | Ball outside x/y/z bounds |
| `robot_out_of_bounds` | Wrist outside safe workspace |
| `point_scored` | Ball crosses opponent's side |

## Plugging in RL

```python
from tasks.pingpong_env_cfg import PingPongDualArmEnvCfg
from tasks.pingpong_env import PingPongEnv

cfg = PingPongDualArmEnvCfg()
cfg.scene.num_envs = 64
cfg.ik_solver = "diffik"
env = PingPongEnv(cfg=cfg)
obs = env.reset()  # {"policy": (64, 45)}
action = policy(obs["policy"])  # (64, 12)
obs, reward, terminated, truncated, info = env.step(action)
```

## Known Issues

1. **GPU pipeline**: Isaac Lab 0.54.3 PhysX tensor API uses CUDA indices but
   Isaac Sim 5.1.0-rc.19 PhysX expects CPU indices. Works with `--device cpu`
   or inside the Apptainer container with matched versions.

2. **Rackets are kinematic**: Ball collision works but without compliant contact.

3. **Left arms uncontrolled**: Only the right arm receives IK commands.

## Stack

| Component | Version | Location |
|-----------|---------|----------|
| Isaac Sim | 5.1.0-rc.19 | `/home/vladi/isaac-sim-5.1.0/` |
| Isaac Lab | 0.54.3 (2.3.2) | `/home/vladi/IsaacLab/source/isaaclab/` |
| Python | 3.11 | `.master_venv` |
| PyTorch | 2.7.0+cu128 | GPU | " description="Implementation documentation" filePath="/home/vladi/IsaacLab/master_isaac/pingpong_dual_arm/implementation.md">/home/vladi/IsaacLab/master_isaac/pingpong_dual_arm/implementation.md</parameter>
