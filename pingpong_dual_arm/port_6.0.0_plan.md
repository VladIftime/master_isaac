# Porting Plan: Isaac Sim 5.1.0 → 6.0.0

## Current State

```
Isaac Sim   5.1.0-rc.19   ~/isaac-sim
IsaacLab   2.2.1          ~/IsaacLab  (v3.0.0-beta available)
Python     3.11           ~/env_isaaclab
cuRobo     old version    pip-installed, custom ActionTerm wrapper
Lula RMP    isaacsim.robot_motion.motion_generation (deprecated in 6.0)
Env type   ManagerBasedRLEnv with config-driven pipeline
```

### What the Project Uses from Isaac Lab

| Module | Usage |
|--------|-------|
| `isaaclab.managers.action_manager` | `ActionTerm`, `ActionTermCfg` — base for IK action terms |
| `isaaclab.envs.mdp.actions.actions_cfg` | `DifferentialInverseKinematicsActionCfg`, `OperationalSpaceControllerActionCfg` |
| `isaaclab.envs.mdp.actions.rmpflow_actions_cfg` | `RMPFlowActionCfg` (Lula-based, **deprecated in 6.0**) |
| `isaaclab.controllers` | `DifferentialIKControllerCfg`, OSC controller, RmpFlowController |
| `isaaclab.envs` | `ManagerBasedRLEnv`, `ManagerBasedRLEnvCfg` |
| `isaaclab.scene` | `InteractiveSceneCfg` |
| `isaaclab.assets` | `ArticulationCfg`, `RigidObjectCfg`, `AssetBaseCfg` |
| `isaaclab.sim` | `SimulationCfg`, `UrdfFileCfg`, `UsdFileCfg`, `GroundPlaneCfg` |
| `isaaclab.actuators` | `ImplicitActuatorCfg` |
| `isaaclab.managers` | `ObservationTermCfg`, `RewardTermCfg`, `TerminationTermCfg`, `EventTermCfg`, `SceneEntityCfg` |
| `isaaclab.utils` | `configclass` |
| `isaaclab.utils.math` | `quat_apply`, `quat_mul` |
| `isaaclab.app` | `AppLauncher` |
| `isaaclab_rl.skrl` | `SkrlVecEnvWrapper` |
| `isaaclab_tasks.utils` | `import_packages`, `hydra_task_config` |

### Current IK Solver Factory

```python
IK_BUILDERS = {
    "diffik":   build_diffik_action,    # Torch DLS IK, relative delta actions
    "osc":      build_osc_action,       # Impedance controller, relative delta
    "rmpflow":  build_rmpflow_action,   # Lula RMPflow (WILL BREAK in 6.0)
    "curobo":   build_curobo_action,    # Custom ActionTerm wrapping cuRobo solve_batch()
}
```

---

## What Changes in Isaac Sim 6.0

### New Experimental Robot Motion API

```
Robot Motion (Experimental)
├── Motion Generation API          ← unified BaseController + RobotState framework
│   ├── BaseController             ← reset() + forward() interface
│   ├── RobotState                 ← joint/link/site/root space
│   ├── ControllerContainer        ← runtime controller switching
│   ├── ParallelController         ← combine multiple controllers
│   ├── SequentialController       ← chain controllers
│   ├── Trajectory + Path          ← time-indexable motion
│   ├── SceneQuery                 ← obstacle discovery
│   ├── WorldInterface             ← obstacle sync adapter
│   └── WorldBinding               ← auto-sync USD → planning world
│
├── cuMotion Integration           ← replaces Lula
│   ├── CumotionRobot              ← robot config + kinematics
│   ├── CumotionWorldInterface     ← obstacle management
│   ├── RmpFlowController          ← reactive collision-aware (BaseController)
│   ├── GraphBasedPlanner          ← RRT-based global planning
│   ├── TrajectoryGenerator        ← time-optimal waypoint → trajectory
│   └── TrajectoryOptimizer        ← collision-free optimization
│
└── PINK Integration               ← NEW
    ├── PinkRobot                  ← Pinocchio model + data
    ├── PinkIKController           ← QP differential IK (BaseController)
    ├── FrameTask                  ← EE pose tracking
    ├── PostureTask                ← joint regularization
    ├── SelfCollisionBarrier       ← minimum-distance CBFs
    └── PositionBarrier            ← workspace bounds
```

### Deprecated in 6.0

- Lula-based `Motion Generation` → now marked "Deprecated" in docs
- Old `RmpFlow`/`RmpFlowSmoothed` classes (`isaacsim.robot_motion.motion_generation`)
- Old `ArticulationMotionPolicy` wrapper

### cuRobo Status in 6.0

**cuRobo is NOT deprecated.** It is a standalone GPU-accelerated IK library.
cuMotion uses cuRobo as its backend. The Isaac Sim 6.0 docs still reference
cuRobo directly, though the old tutorial page is considered "no longer maintained"
in favor of the new cuMotion tutorials.

Latest cuRobo: open-source (Apache 2.0), installable via pip/source.
Requirements: GPU driver ≥ 580.65.06, CUDA ≥ 12, Python 3.10–3.12.

---

## Target Architecture

```
Isaac Sim 6.0.0    ~/isaac-sim-6.0
IsaacLab 3.x       ~/IsaacLab (latest tag tracking 6.0)
Python 3.11        ~/env_isaaclab_6  (NEW venv)
cuRobo latest      pip/uv install from github.com/NVlabs/curobo

IK_BUILDERS = {
    "diffik":   build_diffik_action,      # KEEP — verify import paths
    "osc":      build_osc_action,         # KEEP — verify import paths
    "curobo":   build_curobo_action,      # UPDATE — latest cuRobo, verify CUDA compat
    "pink":     build_pink_action,        # NEW — QP-based differential IK
    "cumotion": build_cumotion_action,    # NEW — replaces "rmpflow"
}
```

---

## Phase 1: Install Isaac Sim 6.0 + Update IsaacLab

**Effort: ~1 day | Risk: Medium**

### 1.1 Install Isaac Sim 6.0

```bash
# Download from https://developer.nvidia.com/isaac/sim
# Install to ~/isaac-sim-6.0 (keep 5.1.0 at ~/isaac-sim as fallback)

# Verify
cat ~/isaac-sim-6.0/VERSION
~/isaac-sim-6.0/isaac-sim.sh --version
```

### 1.2 Update IsaacLab to 3.x

```bash
cd ~/IsaacLab
git fetch origin
git checkout v3.0.0-beta    # or first stable 6.0-compatible tag

# Check the new VERSION
cat VERSION
```

### 1.3 Create New Python Environment

```bash
# Create new venv for 6.0 (keep ~/env_isaaclab for 5.1.0)
python3.11 -m venv ~/env_isaaclab_6
source ~/env_isaaclab_6/bin/activate

# Install IsaacLab from source
cd ~/IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_rl
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_assets

# Install skrl
pip install skrl>=1.4.2
```

### 1.4 Install Latest cuRobo

```bash
source ~/env_isaaclab_6/bin/activate

# Check driver CUDA version
nvidia-smi | grep CUDA

# If CUDA 12.x:
git clone https://github.com/NVlabs/curobo /tmp/curobo
cd /tmp/curobo
pip install .[cu12]       # or .[cu12-torch] for fresh PyTorch

# If CUDA 13.x:
pip install .[cu13]       # or .[cu13-torch]

# Verify
python -c "import curobo; print(curobo.__version__)"
```

### 1.5 Update isaaclab.sh Script

The `isaaclab.sh` launcher in the IsaacLab root needs to point to the new Isaac Sim 6.0 installation. Check the script for `ISAAC_SIM_PATH` or equivalent variable.

---

## Phase 2: Audit IsaacLab API Changes (2.2.1 → 3.x)

**Effort: ~0.5 day | Risk: Medium-High**

### 2.1 Check Core Imports

Verify these still work in IsaacLab 3.x:

```python
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.scene.interactive_scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg, UsdFileCfg, GroundPlaneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab.app import AppLauncher
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import import_packages, hydra_task_config
```

### 2.2 Check Config Schema Changes

- `ManagerBasedRLEnvCfg`: check field names (scene, observations, actions, events, rewards, terminations)
- `ActionTermCfg`: check if `asset_name` field still exists
- `ObservationTermCfg` / `RewardTermCfg` / `TerminationTermCfg` / `EventTermCfg`: check `func` + `params` convention
- `ImplicitActuatorCfg`: check `joint_names_expr` (regex patterns), `stiffness`, `damping`
- `ArticulationCfg`: check `spawn` → `UrdfFileCfg`, `init_state`, `actuators`

### 2.3 Check Pink/CuMotion Availability in IsaacLab 3.x

```bash
# Check what 3.0 added:
ls ~/IsaacLab/source/isaaclab/isaaclab/controllers/pink_ik/
ls ~/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/pink_*
# Check if cuMotion wrapper was added:
ls ~/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/cumotion_*  # may not exist yet
```

---

## Phase 3: IK Solver Migration

**Effort: ~3-5 days | Risk: High**

### 3A: diffik — Verify Only

**Risk: Low**

`DifferentialIKController` is pure PyTorch, no Isaac Sim coupling.
Verify the import path and config field names are unchanged in IsaacLab 3.0.

```python
# Expected to work unchanged:
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
```

### 3B: osc — Verify Only

**Risk: Low**

`OperationalSpaceController` is pure PyTorch, same as diffik. Verify imports.

### 3C: rmpflow (Lula) → cuMotion RMPflow (REPLACE)

**Risk: Medium-High**

The Lula-based `RMPFlowActionCfg` will not work in 6.0. Must replace with cuMotion.

#### What needs to happen:

1. **New dependency**: Isaac Sim 6.0 cuMotion extension (`isaacsim.robot_motion.cumotion` or equivalent)
2. **New config format**: cuMotion uses YAML robot configs — different from Lula's RMP config YAML + URDF + collision YAML triplet
3. **New integration pattern**: cuMotion's `RmpFlowController` implements the new `BaseController` interface (`reset()` + `forward()`) rather than the old `ArticulationMotionPolicy` wrapper

#### Implementation approach:

Create a new `build_cumotion_action()` function in `ik_solvers/__init__.py`:

```python
def build_cumotion_action(
    asset_name: str,
    joint_names: list[str],
    body_name: str,
    side: str,
    **kwargs,
) -> ActionTermCfg:
    """
    Build a cuMotion RMPflow action config for Isaac Sim 6.0.

    Requires the cuMotion extension to be enabled in Isaac Sim.
    Uses CumotionRobot for robot description and RmpFlowController
    implementing the BaseController interface.
    """
    # TODO: Implement based on cuMotion tutorial API:
    # https://docs.isaacsim.omniverse.nvidia.com/6.0.0/cumotion/tutorial_rmpflow.html
    ...
```

> **Note**: IsaacLab 3.0 may ship a `CumotionActionCfg` or similar wrapper.
> Check `isaaclab/envs/mdp/actions/` in the 3.0 release before implementing from scratch.
> If not, this will require a custom `ActionTerm` subclass similar to the existing
> `CuroboInverseKinematicsAction` pattern.

### 3D: cuRobo — Update to Latest

**Risk: Low-Medium**

cuRobo is standalone and independent of Isaac Sim version. The custom wrapper
(`curobo_ik.py`) reads IsaacLab `Articulation.data` tensors and writes via
`set_joint_position_target()` — these are IsaacLab APIs, not Isaac Sim APIs.

#### Migration checklist:

1. **Install latest cuRobo** (see Phase 1.4)

2. **Verify imports still work** — core classes unchanged:
   ```python
   from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # same
   from curobo.types.math import Pose                                   # same
   from curobo.types.robot import RobotConfig                           # same
   from curobo.types.base import TensorDeviceType                       # same
   from curobo.util_file import get_robot_configs_path, join_path, load_yaml  # same
   ```

3. **Verify UR5e robot config** exists in latest cuRobo:
   ```python
   ur_yaml = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
   ```
   If the config path or format changed, adapt accordingly.

4. **CUDA graph warm-up** (lines 108-124 in `curobo_ik.py`):
   - Current code works around CUDA 11.x limitation by using `num_seeds=1` at warm-up
   - Isaac Sim 6.0 likely bundles CUDA ≥ 12.0 where graph reset is supported
   - The warm-up pattern should still work; consider updating to use actual runtime parameters if CUDA ≥ 12

5. **Frame conversion** (lines 153-156):
   - `self._asset.data.body_pos_w` and `self._env.scene.env_origins` are IsaacLab APIs — should be unchanged
   - `quat_apply` and `quat_mul` from `isaaclab.utils.math` — should be unchanged

6. **Joint name patterns** (lines 70-71, 82-89):
   - `self._asset.find_joints(...)` — IsaacLab API, unlikely to change
   - `self._asset.find_bodies(...)` — same

7. **Testing cuRobo in 6.0**:
   ```bash
   source ~/env_isaaclab_6/bin/activate
   cd ~/IsaacLab
   ./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_ik.py \
       --ik_solver curobo --num_envs 4
   ```

### 3E: PINK — Add as New Option

**Risk: Low**

PINK is already integrated in IsaacLab 2.2.1 (controllers, action config, action term).
IsaacLab 3.0 should have more polished version.

**Add `build_pink_action()` to `ik_solvers/__init__.py`:**

```python
def build_pink_action(
    asset_name: str,
    joint_names: list[str],
    body_name: str,
    side: str,
    **kwargs,
) -> ActionTermCfg:
    """Build a PINK IK action config using QP-based differential IK."""
    from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg

    return PinkInverseKinematicsActionCfg(
        asset_name=asset_name,
        joint_names=joint_names,
        body_name=body_name,
        use_relative_mode=False,  # absolute target like cuRobo
        position_command_scale=0.1,
        rotation_command_scale=0.1,
    )
```

**Key differences from diffik/osc:**
- Action format: absolute target pose (same as cuRobo), NOT relative deltas
- Uses Pinocchio for kinematics (not PyTorch Jacobian)
- OSQP solver backend
- Supports self-collision barriers (via optional config)

**Training scripts that branch on `ik_solver` type** must handle PINK's absolute
target format correctly (same branch as cuRobo).

---

## Phase 4: Scene Configuration + Physics

**Effort: ~0.5 day | Risk: Low**

### 4.1 URDF Import

`UrdfFileCfg` with runtime URDF→USD conversion should work.
Your `dual_arm_robot_rackets.urdf` has full collision meshes — verify they import correctly.

### 4.2 Physics Materials

Check if 6.0 changed the physics material API:
```python
# Current config:
physics_material=sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.8,
    compliant_contact_mode="min",
)
```
Verify `RigidBodyMaterialCfg` fields are unchanged and `compliant_contact_mode` is still valid.

### 4.3 Kinematic Rigid Bodies

Table and stands use `kinematic_enabled=True`. Verify this is still supported in PhysX 5.x.
Should work — this is standard USD/PhysX.

### 4.4 Ball Physics

```python
# Gyroscopic forces:
RigidObjectCfg(
    ...
    enable_gyroscopic_forces=True,  # check availability in 6.0
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
    sleep_threshold=0.005,
    stabilization_threshold=0.001,
)
```
Verify `enable_gyroscopic_forces` exists in Isaac Sim 6.0.

### 4.5 Fabric Simulation

Your config uses `fabric=True`. In 6.0, Fabric may be the default physics backend.
Check if `enable_fabric` field name changed.

### 4.6 GPU Buffer Sizes

```python
physx = sim_utils.PhysxCfg(
    gpu_found_lost_pairs_capacity=2**20,
    gpu_max_rigid_contact_count=2**20,
    gpu_max_rigid_patch_count=2**15,
    ...
)
```
These should still be valid PhysX 5.x parameters.

---

## Phase 5: Environment Registration + Training Script

**Effort: ~1 day | Risk: Low-Medium**

### 5.1 Environment Registration

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
Gymnasium registration API is standard — unlikely to change.

### 5.2 Update IK_SOLVER_TYPE

In `tasks/pingpong_env_cfg.py`:
```python
# Update the Literal type:
IK_SOLVER_TYPE = Literal["diffik", "osc", "curobo", "pink", "cumotion"]
```

### 5.3 Training Script Changes

In `scripts/skrl/train.py`:
- Update `--ik_solver` choices: remove `rmpflow`, add `cumotion`, `pink`
- Verify `AppLauncher` API unchanged in IsaacLab 3.0
- Verify `SkrlVecEnvWrapper` compatible with new IsaacLab version
- Action format branching: PINK uses absolute targets (same branch as cuRobo)

### 5.4 Play Script

Same changes as training script for action format branching.

---

## Phase 6: Game Logic

**Effort: ~0.5 day | Risk: Low**

All game logic in `pingpong_env.py`, `observations.py`, `rewards.py`,
`terminations.py`, `events.py` uses IsaacLab abstractions that are
Isaac-Sim-version independent:

- `Articulation.data.body_pos_w`, `body_quat_w`, `joint_pos`, `joint_vel`
- `RigidObject.data.root_pos_w`, `root_lin_vel_w`, `root_ang_vel_w`
- `robot.find_bodies(...)`, `robot.find_joints(...)`
- `ball.write_root_pose_to_sim(...)`, `robot.write_joint_state_to_sim(...)`

No expected changes needed here.

---

## Phase 7: Testing + Validation

**Effort: ~2 days | Risk: Medium**

### 7.1 Smoke Test

```bash
source ~/env_isaaclab_6/bin/activate
cd ~/IsaacLab

# Test basic scene loading with each solver:
./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_env.py \
    --ik diffik --num_envs 4 --headless

./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_env.py \
    --ik curobo --num_envs 4 --headless

./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_env.py \
    --ik pink --num_envs 4 --headless

./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_env.py \
    --ik cumotion --num_envs 4 --headless
```

### 7.2 IK Solver Benchmark

```bash
# Compare IK throughput across solvers:
./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_ik_swing.py \
    --ik diffik --num_envs 64

./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_ik_swing.py \
    --ik curobo --num_envs 64

./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/test_ik_swing.py \
    --ik pink --num_envs 64
```

### 7.3 Random Policy Rollout

```bash
# Verify no CUDA errors, rewards compute correctly:
./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/random_policy.py \
    --ik diffik --num_envs 64 --episodes 10
```

### 7.4 Short Training

```bash
# 1000 steps to verify data pipeline end-to-end:
./isaaclab.sh -p vlad/master_isaac/pingpong_dual_arm/scripts/skrl/train.py \
    --task=PingPong-DualArm-Direct-v0 --headless --num_envs=64 \
    --ik_solver=curobo --max_iterations=100
```

### 7.5 Compare with 5.1.0 Baseline

Run the same short training in the old environment for reward curve comparison.

---

## Solver Decision Matrix

| Solver | Action Format | GPU? | Collision? | Best For |
|--------|--------------|------|------------|----------|
| **diffik** | Rel delta (6D) | No (torch) | No | Debugging, baseline |
| **osc** | Rel delta (6D) | No (torch) | Impedance | Compliant control |
| **curobo** | Abs target (6D) | **Yes** | Optional | **Max throughput IK** (CUDA graph) |
| **pink** | Abs target (6D) | No (qp) | Self-collision | Multi-task, QP-based IK |
| **cumotion** | Abs target (6D) | Yes | Full env | Reactive + safe (RMPflow) |

### Recommendation

1. **Keep `curobo` as primary** — best GPU throughput, proven in your setup, minimal code changes
2. **Add `pink` as lightweight fallback** — fewer dependencies, self-collision support, already in IsaacLab
3. **Keep `diffik`/`osc`** — useful for debugging and as baseline
4. **Add `cumotion` only if needed** — higher integration effort, collision-aware reactive control may fight RL learning

---

## File Changes Summary

| File | Change | Effort |
|------|--------|--------|
| `ik_solvers/__init__.py` | Remove Lula rmpflow builder; add `build_pink_action()`, `build_cumotion_action()`; keep curobo/diffik/osc | ♢♢♢ |
| `ik_solvers/curobo_ik.py` | Verify imports; update CUDA graph warm-up if needed; verify UR5e config path | ♢ |
| `tasks/pingpong_env_cfg.py` | Update `IK_SOLVER_TYPE` union; verify config schemas | ♢ |
| `scripts/skrl/train.py` | Update solver choices, action format branching for pink/cumotion | ♢♢ |
| `scripts/skrl/play.py` | Same as train.py | ♢ |
| `scripts/test_env.py` | Update solver choices | ♢ |
| `scripts/test_ik_swing.py` | Add pink/cumotion action branches | ♢♢ |
| `scripts/random_policy.py` | Update solver choices | ♢ |
| `scripts/test_ik.py` | Update solver choices | ♢ |
| `tasks/*.py` | Game logic — no changes expected | — |
| `source/PingPong/.../__init__.py` | No changes expected | — |

---

## Risk Summary

| Phase | Risk | Key Unknown |
|-------|------|-------------|
| 1. Install | Medium | CUDA version match between Sim 6.0 and cuRobo |
| 2. API Audit | Med-High | IsaacLab 3.0 may restructure modules |
| 3C. cuMotion | High | New API, no pre-existing IsaacLab wrapper |
| 3D. cuRobo | Low-Med | Import paths, CUDA graph behavior |
| 3E. PINK | Low | Already integrated in IsaacLab |
| 4. Scene | Low | Mostly USD/PhysX standard |
| 5. Training | Low-Med | AppLauncher / SkrlVecEnvWrapper API |
| 6. Logic | Low | All IsaacLab abstractions |
| 7. Testing | Medium | Edge cases in new PhysX version |

**Total estimated effort: 9–11 days**

## Appendix: Quick Reference — Key URLs

- Isaac Sim 6.0 Migration Guide: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/migration_guides/isaac_sim_6_0/index.html
- Motion Generation (Experimental): https://docs.isaacsim.omniverse.nvidia.com/6.0.0/motion_generation/index.html
- cuMotion Integration: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/cumotion/index.html
- PINK Integration: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/pink/index.html
- cuRobo + cuMotion (legacy page): https://docs.isaacsim.omniverse.nvidia.com/6.0.0/manipulators/manipulators_curobo.html
- Latest cuRobo docs: https://nvlabs.github.io/curobo/latest/
- Latest cuRobo IK tutorial: https://nvlabs.github.io/curobo/latest/getting-started/inverse_kinematics.html
- Latest cuRobo installation: https://nvlabs.github.io/curobo/latest/getting-started/installation.html
- cuRobo GitHub: https://github.com/NVlabs/curobo
