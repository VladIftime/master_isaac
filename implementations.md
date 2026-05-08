# Implementation Plan — Push-PPO Baseline

**Date**: 2026-05-08

---

## Overview

Create a **baseline PPO approach** for tabletop pushing that can be compared against the
existing DRL/ASP framework (`train_curobo.py`). The baseline uses a single PPO agent
with a **push primitive** maco-action (no Alice/Bob, no ABC, no goal encoder needed for
the simplest variant). The agent predicts push parameters; the environment executes a
multi-step push trajectory using cuRobo IK.

### Comparison targets:
1. **ASP + Goal Encoder** (Plappert + Sukhbaatar) — already implemented in `train_curobo.py`
2. **ASP + Goal Encoder (Charlie)** — same as above
3. **Push-PPO Baseline** (this implementation) — single-agent PPO with push primitive

---

## Design Decisions (from user feedback)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Action space | Relative offsets from object | Generalizes across object positions |
| Push trajectory | Vertical retract | Physically realistic tabletop pushing |
| Objects | Single object only | Simplest possible baseline |
| Action frequency | Macro-action | Agent predicts push params once; env executes multi-step trajectory |

---

## Architecture

### Push Primitive Execution

The push function receives parameters from the PPO agent and executes a multi-phase
trajectory using cuRobo IK:

```
Phase 1: Approach (10 steps)
  EE moves to approach_pos = obj_pos + (offset_x, offset_y, approach_height_z)
  Orientation: tool-down, gripper open

Phase 2: Orient (5 steps)
  EE rotates to target_yaw (around Z axis)
  Gripper stays open

Phase 3: Descend (5 steps)
  EE moves down to surface level at (obj_x+offset_x, obj_y+offset_y, table_z)
  Gripper stays open

Phase 4: Engage (3 steps)
  Close gripper (grasp/pre-push contact)

Phase 5: Push (20 steps)
  EE moves in straight line: (approach_xy) → (approach_xy + push_dx, push_dy)
  Gripper closed, orientation maintained
  Linear interpolation with cuRobo IK per substep

Phase 6: Release (3 steps)
  Open gripper

Phase 7: Retract (5 steps)
  EE moves up to safe height
  Gripper open

Total: ~51 steps per push macro-action
```

### Action Space (PPO Agent Output)

MultiCategorical: **6D × 11 bins** (matching existing setup for consistency)

| Dim | Parameter | Range | Bin → Value |
|-----|-----------|-------|-------------|
| 0 | `approach_offset_x` | [-0.15, 0.15] m | `(bin-5)/5 * 0.15` |
| 1 | `approach_offset_y` | [-0.15, 0.15] m | `(bin-5)/5 * 0.15` |
| 2 | `push_dx` | [-0.30, 0.30] m | `(bin-5)/5 * 0.30` |
| 3 | `push_dy` | [-0.30, 0.30] m | `(bin-5)/5 * 0.30` |
| 4 | `yaw` | [-π, π] rad | `(bin-5)/5 * π` |
| 5 | `push_dz` | [-0.03, 0.03] m | `(bin-5)/5 * 0.03` |

### Observation Space

Single object, flat observation (no goal encoder, no permutation-invariant encoder for the baseline):

```
[ee_pos(3) | ee_euler(3) | gripper(1) | obj_pos(3) | obj_euler(3) | obj_linvel(3) |
 obj_angvel(3) | ee_obj_dist(1) | obj_contact(1) | goal_pos(3) | goal_euler(3) |
 pos_dist(1) | rot_dist(1)]
= 3+3+1+3+3+3+3+1+1+3+3+1+1 = 29D
```

### Network Architecture (Push-PPO Agent)

```
obs (29D)
  │
  ├─ Linear(29 → 512) → ReLU
  ├─ Linear(512 → 256) → ReLU
  ├─ LSTM(256 → 256)       ← keeps temporal reasoning
  │
  ├─ Actor head: Linear(256 → 66) → (6, 11) → MultiCategorical
  └─ Critic head: Linear(256 → 128) → ReLU → Linear(128 → 1)
```

Simpler than the ASP model: no PI encoder, no goal encoder injection, but keeps the LSTM for learning sequential push strategies.

### Reward Structure

Sparse reward mimicking Bob's reward from the ASP paper:
- **+1** per object within success threshold (pos < 0.05m, rot < 2°)
- **-1** if object at goal moves away
- **+5** completion bonus when all objects at goal
- **0** otherwise

### Environment

- Single UR5e robot with Robotiq gripper
- Single object on table (from pool: concave, cube, cylinder, rect, triangle)
- Object randomly placed at episode start
- Goal randomly placed in workspace
- Episode ends when: object at goal (success), max pushes reached (timeout), or object falls off table

---

## Files to Create

| File | Purpose |
|------|---------|
| `asyncDualPlayPPO/tasks/push_task_curobo.py` | Environment config for push task (single agent, single object) |
| `asyncDualPlayPPO/tasks/utils/wrapper_push.py` | Push env wrapper: macro-action execution, reward, reset |
| `asyncDualPlayPPO/tasks/utils/action_push.py` | Push primitive: trajectory generation + cuRobo IK |
| `asyncDualPlayPPO/algorithms/rl/ppo/module_push.py` | Simplified ActorCritic for push-PPO (flat MLP + LSTM, no PI/goal encoder) |
| `asyncDualPlayPPO/train_push.py` | Training script for push-PPO baseline |
| `asyncDualPlayPPO/tests/validate_push.py` | Validation script using existing test configs |

## Files to Modify

| File | Change |
|------|--------|
| `net.md` | Add push-PPO architecture section |
| `README.md` | Add push-PPO baseline to comparisons, file structure, and running instructions |
| `implementations.md` | This plan |

---

## Implementation Steps

1. **Create `action_push.py`** — push primitive function using cuRobo IK
   - `PushAction` dataclass: stores push parameters
   - `execute_push()` function: takes push params + current robot state, returns trajectory of joint positions
   - Uses `ik_solver.solve_batch()` for each waypoint
   - Handles IK failures gracefully (fall back to last valid pose)

2. **Create `push_task_curobo.py`** — environment config
   - Copy from `async_dual_play_curobo.py`, simplify to single agent
   - Single-object observation layout
   - `JointPositionActionCfg` for arm control (cuRobo computes joint positions externally)
   - Same scene setup (table, UR5e, random object pool)

3. **Create `wrapper_push.py`** — push environment wrapper
   - Manages push macro-action execution
   - `step(action)` → executes full push trajectory via cuRobo IK, returns cumulative reward
   - Computes sparse {+1/-1/+5} rewards similar to Bob
   - Handles reset, goal sampling, object placement
   - Tracks push count, object position, goal state

4. **Create `module_push.py`** — simplified ActorCritic
   - Takes flat 29D observation
   - MLP encoder: Linear(29→512→256) with ReLU
   - LSTM(256→256)
   - Actor: Linear(256→66) → MultiCategorical(6,11)
   - Critic: Linear(256→128→1)
   - `bins_to_push_action()` decoder: bin indices → push parameters

5. **Create `train_push.py`** — training script
   - Import cuRobo before AppLauncher (same pattern as train_curobo.py)
   - Single PPO agent loop (no Alice/Bob, no ABC, no historical pool)
   - Per-step: agent predicts push params → execute push trajectory → get cumulative reward
   - PPO storage records one transition per push macro-action
   - Standard PPO update (3 epochs, 4 minibatches, clip=0.2, ent_coef=0.05)
   - Logging: push success rate, mean reward, IK fail rate
   - Checkpoint save/resume

6. **Create `validate_push.py`** — validation script
   - Uses existing `validation_configs.py` test configs
   - Loads trained checkpoint, runs push agent for each test
   - Records success rate, completion time, push count
   - Outputs results summary for comparison with ASP approach

7. **Update `net.md`** — add push-PPO architecture diagram
8. **Update `README.md`** — add push-PPO baseline section

---

## Dependencies on Existing Code

| Dependency | Usage |
|------------|-------|
| `train_curobo.py` | Pattern for cuRobo import ordering, IK solver setup, action decoding |
| `module.py` / `ActorCritic` | Reference for MultiCategorical, LSTM, encoding patterns |
| `ppo.py` / `PPO` | Base PPO class (reused as-is) |
| `storage.py` / `RolloutStorage` | PPO storage (reused as-is) |
| `observations.py` | Observation functions (ee_poses, object_states, goal_distance) |
| `rewards.py` | Reward constants |
| `validation_configs.py` | Test scene configurations for validation |
| `test_curobo_follow_target.py` | cuRobo IK usage pattern, workspace clamping, TCP offset |
| `wrapper.py` | Reference for sparse reward computation, env wrapper pattern |

---

## Reward Function

**Dense shaping** — continuous reward computed **after each push macro-action**:

```
reward = α · (d_prev − d_now) − β · d_now + completion_bonus
```
where:
- `d_prev` = L2 position error before the push (meters)
- `d_now` = L2 position error after the push (meters)
- `α = 10.0` (improvement gain — rewards getting closer to goal)
- `β = 0.5`  (small penalty proportional to remaining distance)
- `completion_bonus = +5.0` when object enters goal zone (pos < 0.05m, rot < 2°)

This provides a smooth gradient toward the goal at every push, enabling fast
convergence. The improvement term `(d_prev − d_now)` rewards pushes that move
the object closer, while the penalty term `−β · d_now` prevents the agent from
settling far from the target.
