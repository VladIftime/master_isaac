# Porting Plan: Isaac Sim 5.1.0 → 6.0.0

## Current State

```
Isaac Sim   5.1.0-rc.19   ~/isaac-sim-5.1.0
IsaacLab   2.3.2          ~/IsaacLab  (main branch; v3.0.0-beta tag exists)
Python     3.11           ~/master_isaac/.master_venv (venv from 5.1.0 kit python)
cuRobo     0.7.8          editable install from curobo_source/
Lula RMP    isaacsim.robot_motion.motion_generation (deprecated in 6.0)
Env type   ManagerBasedRLEnv with config-driven pipeline
```

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

## Migration Attempt Findings (June 2026)

A full migration to Isaac Sim 6.0.0-rc.59 was attempted. Below are the results.

### What Worked

| Step | Status | Notes |
|------|--------|-------|
| Isaac Sim 6.0.0 installed at `~/isaac-sim-6.0.0` | OK | VERSION: `6.0.0-rc.59+release.41464.5f2772bc.gl` |
| IsaacLab v3.0.0-beta checkout | OK | Tag exists at commit `a4a7602f2` |
| Venv creation with 6.0 kit Python | OK | Python 3.12.13 |
| `isaacsim` import | OK | Requires `LD_PRELOAD=$ISAAC_SIM_PATH/kit/libcarb.so` |
| `isaaclab` imports (v3.0.0-beta) | OK | Required installing `isaaclab_contrib`, `isaaclab_physx`, `isaaclab_newton`, `isaaclab_visualizers`, `isaaclab_ov` |
| DiffIK and OSC action configs | OK | Import paths unchanged |
| cuRobo v2 install (from `main` branch) | OK | `pip install /tmp/curobo[cu12]` works |
| PINK action config builder | OK | Created `build_pink_action()` — needs test-script adaptation for 7D action format |

### Environment Differences (5.1 → 6.0)

| Item | 5.1.0 | 6.0.0-rc.59 |
|------|-------|-------------|
| Kit Python | 3.11 | 3.12 |
| Venv base python | `kit/python/bin/python3` (3.11) | `kit/python/bin/python3` (3.12) |
| Carb init | Implicit via kit | **Requires** `LD_PRELOAD=$ISAAC_SIM_PATH/kit/libcarb.so` |
| Required env vars | `CARB_APP_PATH`, `EXP_PATH`, `ISAAC_PATH` | Same + `LD_PRELOAD` |
| URDF importer | Works from kit | Broken cached extension must be purged (`~/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.4.31+109.0.1.lx64.r.cp312`) |

### Breaking API Changes in IsaacLab v3.0.0-beta

| API | v2.3.2 (main) | v3.0.0-beta |
|-----|---------------|-------------|
| PhysX config | `self.sim.physx.gpu_found_lost_pairs_capacity = ...` | `self.sim.physics = PhysxCfg(...)` |
| PhysxCfg location | `isaaclab.sim` (built-in) | `isaaclab_physx.physics` (separate package) |
| PhysX packages | All in `isaaclab` | Split into `isaaclab`, `isaaclab_physx`, `isaaclab_newton` |
| `isaaclab_contrib` | v0.0.2 optional | v0.3.0 **required** (imported by `interactive_scene.py`) |
| Articulation roots | Auto-pick first one | Strict check: **exactly 1** required or `articulation_root_prim_path` must be set |
| `ArticulationRootAPI` | Tolerant of multiple | **Error**: "Found multiple articulation roots" |

---

## ⛔ Current Blocker: Physics Initialization Crash

**Status: UNRESOLVED**

After fixing all import, config, and articulation-root issues, the simulation crashes during `scene.update(dt)` on the first environment creation step.

### Error Chain

```
omni.physics.tensors: "Simulation view object is invalidated"
Exception: Failed to get DOF velocities from backend
```

Call stack:
```
manager_based_env.py:197   _init_sim() → self.scene.update(dt)
interactive_scene.py:468   scene.update(dt) → articulation.update(dt)
articulation.py:281         self.data.update(dt)
articulation_data.py:796    self._root_view.get_dof_velocities()
omni.physics.tensors api.py:1693   raise Exception(...)
```

### What Was Tested (All Failed)

| Attempt | Result |
|---------|--------|
| `articulation_root_prim_path="/world"` + `fix_root_link=True` | DOF velocity crash |
| `articulation_root_prim_path="/Geometry/world"` + `fix_root_link=True` | DOF velocity crash |
| No `articulation_root_prim_path` + no `fix_root_link` | DOF velocity crash |
| `self.sim.physics = PhysxCfg(...)` with GPU buffers | DOF velocity crash |
| `self.sim.physics = None` (default) | DOF velocity crash |
| `use_fabric = False` | DOF velocity crash |
| Throwing env (different URDF, no rackets) | Same crash |

### Root Cause Hypothesis

The `omni.physics.tensors` API in Isaac Sim 6.0.0-rc.59 appears incompatible with the articulation view creation in IsaacLab v3.0.0-beta when using dual-arm URDF robots. The view is created successfully (passes the `root_view._backend is None` check in `_initialize_impl`), but becomes invalidated before the first data read.

Possible causes:
1. **Version mismatch**: `6.0.0-rc.59` is a release candidate; the `omni.physics.tensors` v110.1.11 API may differ from what IsaacLab v3.0.0-beta expects
2. **Dual-arm URDF**: Multiple `ArticulationRootAPI` prims may trigger a PhysX internal path that wasn't tested upstream
3. **Missing `v3.0.0` final**: Only `v3.0.0-beta` tag exists — a final release may contain fixes

### Next Steps to Unblock

1. **Update Isaac Sim** past `rc.59` — a release build may have matching physics tensors
2. **Check for IsaacLab `v3.0.0` final** — the beta may have known issues
3. **Test with a single-arm URDF** — isolate whether it's dual-arm-specific
4. **File an issue** with IsaacLab/Isaac Sim teams with the stack trace

---

## cuRobo Migration Issues

### v0.7.8 → v2 API Break

cuRobo v2 (`main` branch, post v0.7.8) completely rewrites the public API:

| v0.7.8 API (current code uses) | v2 API |
|------|-------|
| `curobo.wrap.reacher.ik_solver.IKSolver` | Removed |
| `curobo.types.math.Pose` | Different |
| `curobo.types.robot.RobotConfig` | Different |
| `curobo.util_file.get_robot_configs_path` | Removed |
| `curobo.util_file.join_path` | Removed |
| `curobo.util_file.load_yaml` | Removed |

The custom `CuroboInverseKinematicsAction` in `ik_solvers/curobo_ik.py` needs a full rewrite for the v2 API.

### Build Issues

| Issue | Detail |
|-------|--------|
| System CUDA | `nvcc 12.0` (Ubuntu apt) |
| Isaac Sim 6.0 torch | `2.11.0+cu128` (CUDA 12.8) |
| Driver | `580.159.03` (supports CUDA 13.0) |
| cuRobo v0.7.8 rebuild | **Fails**: `torch.utils.cpp_extension` detects CUDA 12.0 but PyTorch expects 12.8 |
| cuRobo v2 pip install | Works via `pip install /tmp/curobo[cu12]` (uses pip CUDA toolkit) |
| cuRobo v0.7.8 .so files | Built for Python 3.11 — cannot load in Python 3.12 |

**Bottom line**: cuRobo needs a full rewrite of `curobo_ik.py` for the v2 API, OR system CUDA must be upgraded to 12.8+ to rebuild v0.7.8 for Python 3.12.

---

## Required File Changes (Updated)

| File | Change | Tested? |
|------|--------|---------|
| `ik_solvers/__init__.py` | Replace `rmpflow` with `pink` in `IK_SOLVER_TYPE` + `IK_BUILDERS`; add `build_pink_action()` | Config compiles, not runtime-tested |
| `ik_solvers/curobo_ik.py` | Full rewrite for cuRobo v2 API (or rebuild v0.7.8 for Python 3.12) | **Not done** |
| `tasks/pingpong_env_cfg.py` | `self.sim.physx.*` → `self.sim.physics = PhysxCfg(...)`; add `isaaclab_physx` import; add `articulation_root_prim_path="/Geometry/world"` | Compiles, crashes at runtime |
| `tasks/throwing_env_cfg.py` | Same as pingpong | Compiles, crashes at runtime |
| `scripts/test_ik_swing.py` | Remove `rmpflow` from `--ik` choices, add `pink`; handle 7D pink action format | Not runtime-tested |
| `scripts/test_ik_throwing.py` | Same as swing | Not runtime-tested |
| `scripts/skrl/train.py`, `play.py` | Same solver choice updates | Not done |

---

## Updated Venv Setup (6.0)

For reference, the venv activation script needs:

```bash
export CARB_APP_PATH="$ISAAC_SIM_PATH/kit"
export ISAAC_PATH="$ISAAC_SIM_PATH"
export EXP_PATH="$ISAAC_SIM_PATH/apps"
export LD_PRELOAD="$ISAAC_SIM_PATH/kit/libcarb.so"  # CRITICAL for 6.0
export LD_LIBRARY_PATH="$ISAAC_SIM_PATH:$ISAAC_SIM_PATH/kit:..."  # Must exclude 5.1.0 paths
```

And a `.pth` file with paths to `python_packages/isaacsim`, `exts/isaacsim.simulation_app`, `kit/kernel/py`, and `cumotion`/`pink` pip prebundles.

---

## Risk Summary (Updated)

| Phase | Risk | Actual Finding |
|-------|------|----------------|
| 1. Install | Medium | Python 3.12 works; `libcarb.so` LD_PRELOAD required |
| 2. API Audit | **High** | `self.sim.physx` → `self.sim.physics`; `PhysxCfg` moved to `isaaclab_physx`; `isaaclab_contrib` now required |
| 3C. cuMotion | High | Not attempted |
| 3D. cuRobo | **High** | v0.7.8 can't rebuild (CUDA mismatch); v2 requires full API rewrite |
| 3E. PINK | Low | Built config, but 7D action format needs test-script changes |
| 4. Scene/PhysX | **Critical** | **Crashes on init** — dual-arm URDF DOF velocity read fails |
| 5. Training | Medium | Not yet tested |
| 6. Logic | Low | IsaacLab abstractions unchanged |
| 7. Testing | **Blocked** | Cannot test until physics crash is resolved |

**Revised estimate: BLOCKED** — cannot proceed until the Isaac Sim 6.0 physics tensor incompatibility is resolved.
