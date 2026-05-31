# Ping Pong Dual-Arm Environment

Competitive ping pong with two dual-arm robot systems in Isaac Lab / Isaac Sim.
Each robot has a rendered body, head, and two UR5e arms with Robotiq grippers.

## Scene

```
                    Robot B
                +---[BODY]---+
                | left  right |
                | arm   arm   |
                |             |
                |  [RacketB]  |
         -Y  ===|=====|===========|=====  (net)
                |  [RacketA]  |
                |             |
                | arm   arm   |
                | left  right |
                +---[BODY]---+
                    Robot A
                        +Y →
```

- **2 dual-arm robot systems** (each: body + head + 2 UR5e arms + grippers)
- **Ping pong table** with net and ball
- **Kinematic rackets** attached to each robot's playing arm
- **Standard Gymnasium `reset()/step()` interface** — plug in any RL library

## Running

```bash
cd /home/vladi/IsaacLab/master_isaac/pingpong_dual_arm

# Test the environment (CPU — default, works out of the box)
../../isaaclab.sh -p scripts/test_env.py --ik diffik --num_envs 4 --headless

# Test a different IK solver
../../isaaclab.sh -p scripts/test_env.py --ik osc --headless

# IK solver comparison (single robot, reach task)
../../isaaclab.sh -p scripts/test_ik.py --solver rmpflow <<EOF
```

## Development

Use the venv for IDE support (autocomplete, type checking):

```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
```

Note: Runtime execution must use `isaaclab.sh` — it boots Isaac Sim's environment.
Do NOT activate the venv when running `isaaclab.sh` (it overrides Isaac Sim's Python).

## GPU Pipeline

GPU mode (`--device cuda:0`) requires version-matching between Isaac Lab and Isaac Sim.
The current Isaac Lab source (commit `d94504bcf`, 2026-04-29) has GPU tensor API changes
incompatible with Isaac Sim 5.1.0-rc.19. To enable GPU:

1. Checkout Isaac Lab at a known-compatible commit, or
2. Update Isaac Sim to match, or
3. Run inside the Apptainer container (`isaac-lab.sif`) which ships matched versions

## IK Solvers

| Solver | Description | Command |
|--------|-------------|---------|
| `diffik` | Differential IK (DLS) — default, no config files needed | `--ik diffik` |
| `osc` | Operational Space Control — impedance-based torque control | `--ik osc` |
| `rmpflow` | Riemannian Motion Policies — collision-aware GPU planning | `--ik rmpflow` |
| `curobo` | GPU-accelerated nonlinear IK (requires cuRobo) | `--ik curobo` |

Configure the IK solver in code:
```python
cfg = PingPongDualArmEnvCfg()
cfg.ik_solver = "osc"          # swap IK solver
cfg.playing_arm_side = "left"  # use left arm instead of right
```

Or programmatically:
```python
from ik_solvers import build_ik_action
action_cfg = build_ik_action("osc", asset_name="robot_A", side="right")
```

## Plugging in RL

The environment is a `ManagerBasedRLEnv` subclass exposing standard Gymnasium API:

```python
from tasks.pingpong_env import PingPongDualArmEnv
from tasks.pingpong_env_cfg import PingPongDualArmEnvCfg

cfg = PingPongDualArmEnvCfg()
cfg.scene.num_envs = 64
env = PingPongDualArmEnv(cfg)

obs = env.reset()
action = model(obs)
obs, reward, terminated, truncated, info = env.step(action)
```

Compatible with `stable-baselines3`, `rsl-rl`, `skrl`, or any PPO library.

## File Structure

```
pingpong_dual_arm/
├── ik_solvers/
│   └── __init__.py              # Swappable IK solver builders
├── tasks/
│   ├── pingpong_env_cfg.py      # Scene config + MDP (2 robots, table, ball, rackets)
│   ├── pingpong_env.py          # Environment class (ManagerBasedRLEnv)
│   ├── observations.py          # EE poses, ball state, joint positions
│   ├── rewards.py               # Rally time, ball contact, point scoring
│   ├── events.py                # Serve, robot reset
│   └── terminations.py          # Ball out-of-bounds, point detection
├── scripts/
│   ├── test_env.py              # Launch & step through environment
│   ├── test_ik.py               # Compare IK solvers
│   ├── random_policy.py         # Random action rollout
│   └── convert_pingpong_assets.py  # STL → USD mesh converter
├── meshes/                      # All mesh assets
│   ├── dual_arm_head/           # Robot head (blue_head)
│   ├── intuition_body/          # Robot body (multiple colors)
│   ├── pingpong/                # Table, racket, ball meshes
│   ├── robotiq_*_gripper*/      # Gripper meshes
│   └── ur5e/                    # UR5e arm meshes
├── urdf/
│   ├── dual_arm_robot.urdf      # Full dual-arm robot (body + head + arms + grippers)
│   ├── dual_arm_robot_no_gripper_col.usd  # Generated USD for simulation
│   └── cuMotion/                # RMPflow and Lula IK configs
└── cfg/task/
    └── pingpong_dual_arm.yaml   # Task parameters
```

## Stack

| Component | Version |
|-----------|---------|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.3 |
| PyTorch | 2.7.0 |
| Python | 3.11 |
