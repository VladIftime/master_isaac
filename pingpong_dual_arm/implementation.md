# Ping Pong Dual-Arm Environment — Implementation

Competitive ping pong simulation with two identical dual-arm UR5e robot systems in
Isaac Lab / Isaac Sim. Each robot has a rendered body, head, two arms, and a racket
attached to each arm's end-effector via fixed joints.

Full table-tennis game logic ported from
[Isaaclab-TableTennisRobot](https://github.com/org/repo): virtual paddle contact
detection, table zone scoring, randomized ball serves, and reward shaping.

## Project Structure

```
pingpong_dual_arm/
├── source/PingPong/                       # Gym environment registration package
│   ├── setup.py
│   ├── config/extension.toml
│   └── PingPong/
│       ├── __init__.py
│       └── tasks/
│           ├── __init__.py                # import_packages() for task discovery
│           └── pingpong_dual_arm/
│               ├── __init__.py            # gym.register("PingPong-DualArm-Direct-v0")
│               └── agents/
│                   └── skrl_ppo_cfg.yaml  # PPO agent configuration
├── ik_solvers/                            # Swappable IK solver builders
│   └── __init__.py
├── tasks/                                 # Environment implementation
│   ├── __init__.py
│   ├── pingpong_env_cfg.py                # Scene + MDP config (full game logic)
│   ├── pingpong_env.py                    # Environment class (game state, racket tracking)
│   ├── observations.py                    # Observation functions (joints, EE, ball)
│   ├── rewards.py                         # Reward functions (ported from TableTennisRobot)
│   ├── events.py                          # Reset events (serve, joint reset)
│   └── terminations.py                    # Boundary + round-end checks
├── scripts/
│   ├── skrl/
│   │   ├── train.py                       # Training launcher (skrl PPO)
│   │   └── play.py                        # Inference / playback
│   ├── test_env.py                        # Launch & step environment
│   ├── pingpong_swing.py                  # Swing demo with macro primitives
│   ├── test_ik.py                         # Compare IK solvers on reach tasks
│   ├── random_policy.py                   # Random action rollout
│   └── convert_meshes.py                  # STL → USD with MeshConverter
├── meshes/                                # All 3D assets
│   ├── custom_usd_pingpong/               # Custom USDs ported from TableTennisRobot
│   │   ├── Table_tennis.usd               # Table model
│   │   ├── Ping_pong_ball.usd             # Ball model (restitution 0.8)
│   │   └── UR10_instanceable_pong.usd     # UR10 reference (not used directly)
│   ├── dual_arm_head/                     # Robot head (OBJ)
│   ├── intuition_body/                    # Robot body (OBJ/DAE/STL)
│   ├── pingpong/                          # Table STL, racket STL, ball
│   ├── robotiq_2f_85_gripper_visualization/
│   ├── robotiq_2f_gripper_description/
│   └── ur5e/                              # UR5e arm collision/visual
├── assets/
│   ├── urdf/ur_robotics/ur5e/             # Single-arm URDF files
│   └── pingpong/                          # Generated USD files
├── urdf/
│   ├── dual_arm_robot.urdf                # Full dual-arm + body + head + Robotiq grippers
│   ├── dual_arm_robot_no_gripper_col.urdf  # Dual-arm + body + head + Robotiq grippers (col)
│   ├── dual_arm_robot_rackets.urdf          # Dual-arm + body + head + racket end-effectors
│   ├── ur10_racket.urdf                     # Single UR10 arm + racket (reference)
│   ├── UR10_instanceable_pong.usd           # UR10 USD source (legacy TableTennisRobot)
│   ├── config.yaml                        # URDF→USD converter config
│   ├── configuration/                     # Generated USD configs
│   └── cuMotion/                          # RMPflow + Lula IK configs
├── cfg/
│   ├── task/pingpong_dual_arm.yaml        # Task hyperparameters
│   └── agents/                            # (legacy agent configs — see source/)
└── implementation.md
```

## Scene Layout

```
                    Robot B (Y=+2.7, on stand)
                +---[BODY]---+                          z
                | left  right |                          ↑
                | arm   arm   |                          |
[BODY at z=0.6] |             |                          |
    [Stand z=0-0.6]           |
                |  [RacketB]  |
                |             |
          =======|=====|==========  Table (1.525 × 2.74 m, KINEMATIC)
                |             |
                |  [RacketA]  |
                |             |
[BODY at z=0.6] |             |
    [Stand z=0-0.6]           |
                | arm   arm   |
                | left  right |
                +---[BODY]---+
                    Robot A (Y=-2.7, on stand)
                        +Y →
```

- **Robot positions**: global constants `STAND_A_POS = (0, -2.7, 0.6)`, `STAND_B_POS = (0, 2.7, 0.6)` — matching the UR10 base position in TableTennisRobot
- **Stands**: Box cuboids (0.5×0.5×0.6 m) beneath each robot, kinematic, centered at half height (z=0.3), matching UR10 base height
- **Table**: `meshes/custom_usd_pingpong/Table_tennis.usd` — **kinematic** (fixed in place, prevents movement when arms collide)
- **Ball**: `meshes/custom_usd_pingpong/Ping_pong_ball.usd` — dynamic with gyroscopic forces
- **Robot A**: at Y=-2.7, body_base_link z=0.6 (on stand), facing +Y (toward table)
- **Robot B**: at Y=+2.7, body_base_link z=0.6 (on stand), rotated 180° (facing -Y)
- **Rackets**: Attached as fixed child links (`left_racket_link`, `right_racket_link`) to each arm's `tool0` end-effector frame via `dual_arm_robot_rackets.urdf`. Visual mesh `racket_bot_scale.stl` (in meters, origin at handle base, rotated 90° X).
- **Arm initial pose**: both arms on each robot start with the UR10 ready stance joint angles (see below)

## Running

```bash
source ~/env_isaaclab/bin/activate
cd /home/vlad/IsaacLab/vlad/master_isaac/pingpong_dual_arm

# Test environment (verify scene loads)
../../../isaaclab.sh -p scripts/test_env.py --ik diffik --num_envs 4 --headless

# Headless training
../../../isaaclab.sh -p scripts/skrl/train.py --task=PingPong-DualArm-Direct-v0 --headless --num_envs=2048

# With specific IK solver
../../../isaaclab.sh -p scripts/skrl/train.py --task=PingPong-DualArm-Direct-v0 --headless --ik_solver=osc --num_envs=64

# Playback
../../../isaaclab.sh -p scripts/skrl/play.py --task=PingPong-DualArm-Direct-v0 --num_envs=1

# Resume from checkpoint
../../../isaaclab.sh -p scripts/skrl/train.py --task=PingPong-DualArm-Direct-v0 --headless --checkpoint=logs/skrl/pingpong_dual_arm/.../checkpoints/agent_10000.pt
```

## Game Logic (ported from TableTennisRobot)

### Table Zones

Table contact zones use environment-local coordinates (origin at table center):

```
  Negative Y zone [-1.35, -0.1]: Robot A's OWN zone, Robot B's OPPONENT zone
  Positive Y zone  [0, 1.36]:    Robot B's OWN zone, Robot A's OPPONENT zone
  X zone: [-0.74, 0.74],  Z zone: [0.68, 0.735]
```

Z-zone match only triggers when the ball is at table surface height (0.68–0.735 m).

### Reward Structure

| Term | Weight | Description |
|------|--------|-------------|
| `paddle_contact_A/B` | 1.0 | Continuous 0-1 proximity bonus (distance threshold 6cm) |
| `velocity_A/B` | 0.5 | One-time bonus for fast return speed after paddle contact |
| `table_success_A/B` | 5.0 | Ball reaches opponent's table half after paddle contact |
| `table_fail_A/B` | -2.0 | Ball hits own table half (augmented by ball Y position) |
| `ball_floor` | -3.5 | Ball drops below z=0.65 |
| `ball_pos_A/B` | 2.0 | Position shaping proportional to forward progress on success |

All reward weights are centralized in `RewardsCfg`. Precomputed tensors in
`_compute_intermediate_values()` are raw (unscaled) — the config weight
multiplies them to produce the final per-step reward.

### Contact Detection

Virtual distance-based detection (no contact sensors):
- Paddle surface point computed at `wrist_3_link` + offset `(0, 0.265, 0)`
  rotated by link quaternion
- Contact score: `clamp((threshold - distance) / threshold, 0, 1)`
- Threshold: 6 cm
- Latched `has_touch_paddle` flag (once true, stays true for the episode)

### Termination

Episode ends (5s max) on:
- **Success**: Ball in opponent's table zone after paddle contact
- **Fail**: Ball in own table zone after bouncing from opponent's side
- **Floor**: Ball below z=0.65
- **Out of bounds**: Ball outside wide safety bounds (x=±2.0, y=±3.5, z<−1.0)
- **Time limit**: 5 seconds (~600 steps at 120 Hz)

### Ball Serve (Reset)

Alternating server each episode, ball spawns **outside** table zones:
- Server A (near -Y): ball at y=-1.5, velocity toward +Y (toward B)
- Server B (near +Y): ball at y=+1.5, velocity toward -Y (toward A)
- Velocity randomization (matching TableTennisRobot):
  - X speed: [-1, +1] m/s
  - Y speed: [3.5, 5] m/s (toward opponent)
  - Z speed: [2.0, 2.2] m/s (upward arc)
  - X position noise: [-0.2, +0.2] m

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
cfg.playing_arm_side = "left"
```

## Environment Configuration (`tasks/pingpong_env_cfg.py`)

### Robot Articulation

Each robot spawns from `dual_arm_robot_rackets.urdf` via `UrdfFileCfg`
with runtime URDF→USD conversion. The URDF includes:
- **Body**: `IRL_lab_robot_body.obj` + `robot_body_back.obj`
- **Head**: `blue_head_with_headphone.obj`
- **Two UR5e arms**: 6-DOF each (shoulder_pan/lift, elbow, wrist_1/2/3)
- **Two racket end-effectors**: `racket_bot_scale.stl` (left=Red, right=Blue), fixed joints from each arm's `tool0` frame

Global position constants:
```python
STAND_Z = 0.6          # robot base height matches UR10 base
STAND_A_POS = (0, -2.7, STAND_Z)  # Robot A (near -Y side)
STAND_B_POS = (0,  2.7, STAND_Z)  # Robot B (near +Y side)
```

Robot body base positioned at z=0.6 on top of a 0.5×0.5×0.6 m kinematic stand
cuboid. Stands are centered at half height (z=0.3), extending from z=0 to z=0.6.

Actuators: position-controlled implicit (stiffness=5000, damping=200).
Two groups: `arm_left` and `arm_right`. Rackets are fixed (no actuator).

### Arm Initial Joint Positions (UR10 ready stance)

Both arms on each robot start with the same joint pose as the UR10 reference
from TableTennisRobot:

```
shoulder_pan:  -0.29    (slight lateral rotation)
shoulder_lift: -1.212   (arm forward/down)
elbow:          1.712   (elbow bent)
wrist_1:        0.0
wrist_2:       -0.33
wrist_3:        1.39    (wrist rotated)
```

### Table (Kinematic)

The table is loaded from `meshes/custom_usd_pingpong/Table_tennis.usd` and set
to **kinematic** (`kinematic_enabled=True`, `disable_gravity=True`) to prevent
any movement when robot arms collide with it. The table stays fixed at its USD
default position (origin).

### Physics

```
sim.dt         = 1/120 s (~0.00833 s)
decimation     = 1 (120 Hz control)
episode_length = 5 s (600 steps)
physics_material = friction 1.0, restitution 0.8, combine_mode "min"

GPU buffers:
  gpu_found_lost_pairs_capacity = 1M
  gpu_max_rigid_contact_count   = 1M
  gpu_max_rigid_patch_count     = 327K
```

### Actions

```
Action space: Box (12,) per env
  arm_A (0:6):   [dx, dy, dz, droll, dpitch, dyaw]  — robot A playing arm
  arm_B (6:12):  [dx, dy, dz, droll, dpitch, dyaw]  — robot B playing arm
```

DiffIK with `use_relative_mode=True`, scale=0.15. Configurable via `ik_solver`.

### Observations

```
Observation space: Box (69,) per env
  joint_pos_A (0:12):     left+right arm joint positions
  joint_vel_A (12:24):    left+right arm joint velocities
  ee_pose_A   (24:30):    right_wrist_3_link pos(3) + Euler ZYX(3)
  joint_pos_B (30:42):    left+right arm joint positions
  joint_vel_B (42:54):    left+right arm joint velocities
  ee_pose_B   (54:60):    right_wrist_3_link pos(3) + Euler ZYX(3)
  ball_state  (60:69):    ball pos(3) + lin_vel(3) + ang_vel(3)
```

Positions in environment-local coordinates. Rotations as ZYX Euler angles.

### Events

| Event | Mode | Function |
|-------|------|----------|
| `reset_all` | reset | `mdp.reset_scene_to_default` |
| `randomize_robots` | reset | `reset_robot_joints` (both robots to defaults) |
| `serve_ball` | reset | `serve_ball_alternating` (randomized velocity, alternating side) |

### Terminations

| Term | Type | Function |
|------|------|----------|
| `time_limit` | truncated | `mdp.time_out` (5 s) |
| `table_success_A/B` | terminated | Ball in opponent's table zone after paddle contact |
| `table_fail_A/B` | terminated | Ball in own table zone |
| `ball_floor` | terminated | Ball below z=0.65 |
| `ball_out_of_bounds` | terminated | Wide safety bounds: x=±2.0, y=±3.5, z<−1.0 |

## PingPongEnv Class (`tasks/pingpong_env.py`)

Extends `ManagerBasedRLEnv`. Key additions:

### `step(action)` override
```python
def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    # Compute all game-state tensors from current physics
    self._compute_intermediate_values()
    # Snap kinematic rackets to wrist links
    self._track_rackets()
    return obs, reward, terminated, truncated, info
```

### Game State Tensors (per robot)

| Tensor | Type | Description |
|--------|------|-------------|
| `has_touch_paddle_A/B` | Bool (N,) | Latched: paddle made contact with ball |
| `has_first_bounce_A/B` | Bool (N,) | Latched: ball first bounced on own table half |
| `has_touch_own_table_prev_A/B` | Bool (N,) | Latched: ball ever touched own table half |
| `reward_vel_prev_A/B` | Float (N,) | Anti-double-count: velocity reward given once |
| `_contact_A/B` | Float (N,) | Precomputed contact reward (unscaled) |
| `_table_success_A/B` | Float (N,) | Precomputed table success reward (unscaled) |
| `_table_fail_A/B` | Float (N,) | Precomputed table fail penalty (unscaled) |
| `_ball_floor` | Bool (N,) | Ball below z=0.65 (env-local) |
| `_velocity_A/B` | Float (N,) | Precomputed velocity reward (unscaled) |
| `_ball_pos_rw_A/B` | Float (N,) | Precomputed position shaping reward (unscaled) |

### Racket Tracking

Kinematic rackets are repositioned each frame to the wrist_3_link pose:
```python
def _track_rackets(self):
    wrist = f"{self.cfg.playing_arm_side}_wrist_3_link"
    for robot_name, racket_name in [("robot_A", "racket_A"), ("robot_B", "racket_B")]:
        body_ids, _ = self.scene[robot_name].find_bodies([wrist])
        racket_pose = torch.cat([
            robot.data.body_pos_w[:, body_ids[0]],
            robot.data.body_quat_w[:, body_ids[0]],
        ], dim=-1)
        self.scene[racket_name].write_root_pose_to_sim(racket_pose)
```

### Ball Out-of-Bounds

Wide safety bounds accommodate the full ball trajectory between robots at
y=±2.7: x=±2.0, y=±3.5, z<−1.0. The ball starts outside the table zones
(y=±1.5) and arcs across. Zone detection (z=0.68–0.735) only triggers when
the ball is at table surface height.

## Observations Module

| Function | Dims | Description |
|----------|------|-------------|
| `robot_joint_positions` | (N, 12) | Left + right arm joint positions |
| `robot_joint_velocities` | (N, 12) | Left + right arm joint velocities |
| `ee_poses` | (N, 6) | End-effector pos(3) + Euler ZYX(3) |
| `ball_state` | (N, 9) | Ball pos(3) + lin_vel(3) + ang_vel(3) |
| `ball_to_robot_relative` | (N, 3) | Ball pos relative to robot base |
| `ball_projected_state` | (N, 3) | Ball (x,y) + z_velocity |

## RL Training Pipeline

### Registration

Environment registered in `source/PingPong/PingPong/tasks/pingpong_dual_arm/__init__.py`:
```python
gym.register(
    id="PingPong-DualArm-Direct-v0",
    entry_point="tasks.pingpong_env:PingPongEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.pingpong_env_cfg:PingPongDualArmEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

### Agent Config (`skrl_ppo_cfg.yaml`)

| Parameter | Value |
|-----------|-------|
| Network | MLP [1024, 512, 256, 128], ELU activations |
| rollouts | 512 |
| learning_epochs | 8 |
| mini_batches | 8 |
| discount_factor | 0.999 |
| λ (GAE) | 0.95 |
| learning_rate | 5e-4, KLAdaptiveLR (kl_thresh=0.008) |
| state/value preprocessor | RunningStandardScaler |
| Total timesteps | 300,000 |

### Training CLI

```bash
source ~/env_isaaclab/bin/activate
cd /home/vlad/IsaacLab/vlad/master_isaac/pingpong_dual_arm

../../../isaaclab.sh -p scripts/skrl/train.py \
    --task=PingPong-DualArm-Direct-v0 \
    --headless \
    --num_envs=2048 \
    --ik_solver=diffik \
    --playing_arm_side=right \
    --max_iterations=500
```

## Plugging in RL

```python
from tasks.pingpong_env_cfg import PingPongDualArmEnvCfg
from tasks.pingpong_env import PingPongEnv

cfg = PingPongDualArmEnvCfg()
cfg.scene.num_envs = 64
cfg.ik_solver = "diffik"
cfg.playing_arm_side = "right"
env = PingPongEnv(cfg=cfg)
obs = env.reset()  # {"policy": (64, 69)}
action = policy(obs["policy"])  # (64, 12)
obs, reward, terminated, truncated, info = env.step(action)
```

## Known Issues

1. **GPU pipeline**: Isaac Lab 0.54.3 PhysX tensor API uses CUDA indices but
   Isaac Sim 5.1.0-rc.19 PhysX expects CPU indices. Works with `--device cpu`
   or inside the Apptainer container with matched versions.

2. **Rackets are fixed attachments**: Rackets are fixed child links of each arm's `tool0` frame in the URDF articulation. No compliant contact — virtual contact detection is used for game logic instead.

3. **Left arms uncontrolled**: Only the configured playing arm receives IK commands.
   Both arms' joints are observed.

4. **Self-play reward**: Both robots share a combined reward signal. May need
   separate reward optimization for competitive play.

## Mesh Assets & Conversion

Custom USD files (`meshes/custom_usd_pingpong/`) are pre-built binary USD files
ported from Isaaclab-TableTennisRobot:
- `Table_tennis.usd` — table model (set to kinematic in config)
- `Ping_pong_ball.usd` — ball with restitution 0.8, gyroscopic forces, 8 solver iterations

These replace the previous procedural sphere and mesh-converted table.

## Stack

| Component | Version | Location |
|-----------|---------|----------|
| Isaac Sim | 5.1.0-rc.19 | `/home/vladi/isaac-sim-5.1.0/` |
| Isaac Lab | 0.54.3 (2.3.2) | `/home/vladi/IsaacLab/source/isaaclab/` |
| Python | 3.11 | `~/env_isaaclab` |
| PyTorch | 2.7.0+cu128 | GPU |
| skrl | >= 1.4.2 | `~/env_isaaclab` |
