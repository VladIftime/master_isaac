# Throwing Environment

GPU-accelerated object throwing for a dual-arm robot in Isaac Lab / Isaac Sim.
The robot picks up a drink from a table and throws it into a basket at a randomised position.
A two-phase strategy (kinematic attachment then dynamic friction) protects the learning signal during training.

The 4-parameter macro-action (two shoulder angles, release timing, trajectory duration) mirrors the original Gazebo formulation from [Kasaei & Kasaei, ICRA 2023].

## File Structure

```
throwing_enviroment/
├── tasks/                              Core environment code
│   ├── throwing_env_cfg.py             ManagerBasedRLEnv config (27D obs, 6D act)
│   ├── throwing_env.py                 ThrowingEnv (ManagerBasedRLEnv)
│   ├── throwing_direct_env_cfg.py      DirectRLEnv config (10D obs, 4D act, fast)
│   ├── throwing_direct_env.py          ThrowingDirectEnv (DirectRLEnv, recommended)
│   ├── throwing_direct_camera_env_cfg.py  DirectRLEnv + TiledCamera (benchmarks)
│   ├── throwing_direct_camera_env.py   ThrowingDirectCameraEnv
│   ├── throwing_primitive_env_cfg.py    Legacy gymnasium wrapper config
│   ├── throwing_primitive_env.py        ThrowingPrimitiveEnv (gym.Env)
│   ├── sb3_vec_env.py                  SB3 VecEnv wrapper for DirectRLEnv
│   ├── throw_primitive.py              Throw execution (10-phase state machine)
│   ├── throw_validation_configs.py     10 predefined test targets
│   ├── observations.py                 Observation functions
│   ├── rewards.py                      Reward functions
│   ├── events.py                       Reset events, gripper control
│   └── terminations.py                 Termination conditions
├── scripts/
│   ├── train_sac.py                    SB3 SAC training (recommended)
│   ├── test_train_quick.py             Quick NaN sanity check
│   ├── test_env.py                     Step through env with zero actions
│   ├── test_throw.py                   Test throw primitive manually
│   ├── test_ik_throwing.py             Multi-phase IK pick-and-throw benchmark
│   ├── test_joint_throwing.py          Gazebo-style joint-space throw test
│   ├── validate_throw.py              Deterministic 10-target validation
│   ├── validate_heatmap.py            Dense grid evaluation with heatmap
│   ├── compare_validations.py         Side-by-side CSV comparison plots
│   ├── plot_tb_logs.py                TensorBoard log diagnostic plots
│   ├── benchmark_scalability.py       SPS at different env counts
│   ├── benchmark_zero_copy.py         GPU-resident vs CPU obs transfer
│   ├── benchmark_direct_vs_manager.py DirectRLEnv vs ManagerBased SPS
│   ├── benchmark_tiled_camera.py      Rendering overhead measurement
│   ├── benchmark_utils.py             Shared benchmark utilities
│   ├── prebake_robot_usd.py           One-time URDF to USD conversion
│   ├── prebake_drink.py               Apply physics to drink USD
│   ├── prebake_basket.py              Strip basket to single rigid body
│   ├── prebake_physics.py             Apply CollisionAPI to legacy USDs
│   ├── convert_meshes.py              OBJ to USD mesh conversion
│   ├── export_full_scene.py           Export FK-baked scene USD
│   ├── inspect_usd.py                Print USD prim hierarchy
│   ├── inspect_usd_summary.py        Summarise physics schemas per asset
│   └── skrl/
│       ├── train.py                   skrl PPO/SAC training (legacy)
│       └── play.py                    skrl agent playback
├── ik_solvers/
│   ├── __init__.py                    IK dispatcher (diffik/osc/rmpflow/curobo)
│   ├── curobo_ik.py                   cuRobo IK action term
│   └── ur5e_arm.yml                   cuRobo kinematics config
├── source/Throwing/                   Gymnasium env registration package
│   ├── setup.py
│   └── Throwing/tasks/throwing/
│       ├── __init__.py                Registers 3 gym envs
│       └── agents/
│           ├── skrl_ppo_cfg.yaml
│           └── skrl_sac_cfg.yaml
├── hpc/
│   ├── train_sac_direct.slurm         DirectRLEnv SAC (recommended)
│   ├── train_sac.slurm                ManagerBased SAC (legacy)
│   └── run_benchmarks.slurm           4-benchmark suite
├── gazebo_impl/                       Reference Gazebo/ROS2 code
│   ├── RL_tossing_object_with_obstacle_avoidance_v3.py
│   ├── new_impl.cpp
│   ├── primitive_design.cpp
│   └── behaviour_change.cpp
├── assets/                            USD/mesh assets (drink, basket, robot)
├── meshes/                            Robot mesh files
├── urdf/                              Robot URDF
├── cfg/task/throwing.yaml             Task hyperparameters
├── logs/                              Training logs, checkpoints, validation
├── papers/                            Reference paper
└── implementation.md                  Full project documentation
```

## Environment Variants

Three gymnasium environments are registered:

| Gym ID | Class | Action | Obs | Use case |
|--------|-------|--------|-----|----------|
| `Throwing-Direct-v0` | `ThrowingEnv` | 6D EE deltas | 27D | Per-step IK control (PPO) |
| `Throwing-Primitive-v0` | `ThrowingPrimitiveEnv` | 4D macro | 8D | 1-step throw episodes (legacy SAC) |
| `Throwing-Primitive-Direct-v0` | `ThrowingDirectEnv` | 4D macro | 10D | Fast GPU-batched throw (recommended SAC) |

The recommended variant is `ThrowingDirectEnv`. It runs the full 10-phase throw sequence (stabilise, approach, grasp, lift, wind-up, throw, flight) inside a single outer step using `decimation=320`, with no manager overhead.

The 4D macro-action encodes:
- `action[0]`: initial shoulder angle (wind-up)
- `action[1]`: final shoulder angle (release)
- `action[2]`: release timing as fraction of trajectory [0.05, 1.0]
- `action[3]`: trajectory duration in seconds [0.1, 1.0]

## Setup

Activate the project venv for running scripts:

```bash
source /home/vladi/IsaacLab/master_isaac/.master_venv/bin/activate
cd /home/vladi/IsaacLab/master_isaac/throwing_enviroment
```

All scripts use Isaac Lab's `AppLauncher` and can run locally or inside the Apptainer container on the HPC cluster.

## Training

### SB3 SAC (recommended)

```bash
python scripts/train_sac.py --headless --num_envs 4096 --max_iterations 100000
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_envs` | 4096 | Number of parallel environments |
| `--max_iterations` | 100000 | Total training iterations |
| `--seed` | 42 | Random seed |
| `--checkpoint` | None | Resume from SB3 `.zip` checkpoint |
| `--playing_arm_side` | right | Which arm throws |
| `--headless` | False | Disable rendering |

Output goes to `logs/sac/throwing_primitive/<timestamp>_sac_sb3/`:
- `latest_checkpoint.zip`: SB3 model (auto-saved periodically)
- `checkpoints/`: numbered checkpoints
- `SAC_0/events.out.tfevents.*`: TensorBoard logs

To resume training from a checkpoint:

```bash
python scripts/train_sac.py --headless --num_envs 4096 \
    --checkpoint logs/sac/throwing_primitive/2026-06-19_16-32-07_sac_sb3/latest_checkpoint.zip
```

### skrl SAC/PPO (legacy)

```bash
python scripts/skrl/train.py --task Throwing-Primitive-Direct-v0 --headless --num_envs 1024
```

### HPC (SLURM)

```bash
sbatch hpc/train_sac_direct.slurm
```

The SLURM script handles:
- Apptainer container execution
- Auto-resume on job preemption (SIGUSR1 trap, max 50 chains)
- Local scratch for fast I/O with rsync to NFS on exit
- Automatic checkpoint discovery for resume

## Loading a Trained Model

### SB3

```python
from stable_baselines3 import SAC

model = SAC.load("logs/sac/.../latest_checkpoint.zip")
action, _ = model.predict(obs, deterministic=True)
```

### skrl

```bash
python scripts/skrl/play.py \
    --task Throwing-Primitive-Direct-v0 \
    --checkpoint logs/skrl/throwing_primitive/.../checkpoints/agent_6000.pt \
    --num_envs 1
```

## Validation

### 10-target validation

Runs the trained policy deterministically against 10 predefined target positions with multiple attempts per target. Supports both SB3 `.zip` and skrl `.pt` checkpoints.

```bash
python scripts/validate_throw.py \
    --checkpoint logs/sac/.../latest_checkpoint.zip \
    --fast --headless \
    --num_tests 10 --attempts 20 --seed 42
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to `.zip` (SB3) or `.pt` (skrl) |
| `--fast` | False | Use DirectRLEnv (no IK overhead) |
| `--num_tests` | 10 | Number of target positions (max 10) |
| `--attempts` | 10 | Throws per target |
| `--success_threshold` | 0.15 | AABB distance threshold in metres |
| `--model_type` | auto | Force `sb3` or `skrl` |
| `--seed` | 42 | Seed for reproducibility |

Output:
- `logs/validation_results_fast_<rate>pct_<timestamp>.csv`
- `logs/validation_results_fast_<rate>pct_<timestamp>.png`

### Comparing checkpoints

Compare two or more validation CSV files side by side:

```bash
python scripts/compare_validations.py \
    logs/validation_results_fast_60pct_2026-06-19_23-10-56.csv \
    logs/validation_results_fast_80pct_2026-06-19_23-16-58.csv \
    --labels "20M Steps" "147M Steps" \
    --output_dir logs/comparisons/
```

Produces:
- `comparison_birdseye.png`: top-down scatter of mean landings per target
- `comparison_distances.png`: grouped bar chart of best and mean distance per target

### Heatmap validation

Dense grid evaluation across the table surface:

```bash
python scripts/validate_heatmap.py \
    --checkpoint logs/sac/.../latest_checkpoint.zip \
    --headless --grid_size 10
```

## Testing

### Step through environment

```bash
python scripts/test_env.py --ik diffik --num_envs 4
```

### Test throw primitive with manual parameters

```bash
python scripts/test_throw.py --direct --loop \
    --initial_jv 1.6 --final_jv -0.5 --releasing_time 0.3 --duration 0.5
```

### Quick NaN check (300-500 steps)

```bash
python scripts/test_train_quick.py --headless --steps 300
```

## IK Solvers

Four solvers available, all using 6D relative delta actions:

| Solver | Description | Flag |
|--------|-------------|------|
| `diffik` | Differential IK (DLS), no config files needed | `--ik diffik` |
| `osc` | Operational Space Control, impedance-based | `--ik osc` |
| `rmpflow` | Riemannian Motion Policies, collision-aware | `--ik rmpflow` |
| `curobo` | GPU-accelerated nonlinear IK (requires cuRobo) | `--ik curobo` |

Select in code:

```python
from ik_solvers import build_ik_action
action_cfg = build_ik_action("diffik", asset_name="robot", side="right")
```

The DirectRLEnv variant bypasses IK entirely and uses joint-space control for the throw.

## Benchmarks

Run all four benchmarks:

```bash
sbatch hpc/run_benchmarks.slurm
```

Or individually:

```bash
python scripts/benchmark_scalability.py --headless
python scripts/benchmark_zero_copy.py --headless
python scripts/benchmark_direct_vs_manager.py --headless
python scripts/benchmark_tiled_camera.py --headless
```

Output goes to `logs/benchmarks/<timestamp>/` with CSV and PNG per benchmark.

## Asset Prebaking

One-time steps to prepare USD assets for simulation. Run these before first training if the pre-baked files are missing:

```bash
python scripts/prebake_robot_usd.py      # URDF to USD
python scripts/prebake_drink.py           # MassAPI + CollisionAPI + friction
python scripts/prebake_basket.py          # Single rigid body + CollisionAPI
```

## Reward Function

The throw reward matches the original Gazebo formulation:

```
reward = 0.9 * exp(-d^2 / 0.01) + 0.1 * exp(-d^2 / 0.05)
```

Where `d` is the Euclidean distance from the object's landing position to the basket centre. A success bonus of `+1.0` is added if `d < 0.15 m`. Drops (object falls off table before throw) receive a penalty of `+0.0`.

## Stack

| Component | Version |
|-----------|---------|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.3 |
| PyTorch | 2.7.0 |
| Python | 3.11 |
| SB3 | 2.6.0 |
