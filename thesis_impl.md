# Thesis Implementation Plan: PBRS Reward Redesign for Push-Primitive RL

**Branch**: `asp_goal_encoder`
**Date**: 2026-06-11
**Status**: Implementation plan — not yet started

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Root-Cause Analysis](#2-root-cause-analysis)
3. [Theoretical Foundations](#3-theoretical-foundations)
4. [Reward Architecture Design](#4-reward-architecture-design)
5. [Three Experimental Models](#5-three-experimental-models)
6. [Implementation Plan](#6-implementation-plan)
7. [Evaluation Protocol](#7-evaluation-protocol)

---

## 1. Motivation & Problem Statement

Two training scripts solve the same planar-pushing task (UR5e + T-block on a
table, cuRobo IK, 4D MultiCategorical push primitives):

| Script | Approach | Best SR | PosErr | RotErr |
|--------|----------|---------|--------|--------|
| `train_push.py` | Single-agent PPO (rel-obs + rel-act) | **18.4%** | 0.15 m | 0.78 rad |
| `train_push_asp.py` | ASP Alice/Bob self-play | **0.07%** | 0.55 m | — |

Both use the same 4D action space (`r, phi, length, theta`), the same cuRobo
IK pipeline, and the same physics scene.  The Push-PPO baseline learns to
solve position but plateaus at ~18% SR because it never masters rotation.
The Push-ASP variant fails to learn at all.

### Why Push-PPO plateaus at 18%

The current reward in `wrapper_push.py` uses a fractional improvement formula:

```
R = alpha * (d_prev - d_now) / d_prev          # position improvement
  + alpha * (y_prev - y_now) / y_prev          # rotation improvement
  - beta  * d_now                               # distance penalty
  - beta_rot * y_now                            # yaw penalty
  + 5.0   (one-time, position-only gate)        # completion bonus
  + 2.0   (one-time, pos AND rot gate)          # rotation sub-bonus
  - 5.0   (if tipped)                           # tip penalty
```

Problems with this formulation:

1. **Division-by-zero instability**: `d_prev` is clamped at 0.01 but near the
   goal, ratios explode.  A 1 mm improvement from 1.2 cm away produces
   `3.0 * 0.001 / 0.012 = 0.25`, but a 1 mm improvement from 1.01 cm away
   produces `3.0 * 0.001 / 0.0101 = 0.30` — the reward accelerates
   unpredictably as the agent approaches the goal.

2. **Cyclical exploitation**: The fractional formula rewards oscillation.
   Push the object 5 cm closer (reward +1.5), then push it 5 cm further
   (reward -1.5 but the distance penalty `-beta * d_now` is smaller because
   `d_now` was smaller at the closer position).  The net reward per cycle
   is slightly positive because the penalty term differs between the two
   positions.  The agent can harvest small positive returns by oscillating
   without ever solving the task.

3. **Not policy-invariant**: The raw penalty terms (`-beta * d_now`,
   `-beta_rot * y_now`) are state-dependent rewards that change the optimal
   policy relative to the underlying sparse task.  The agent is incentivised
   to minimise distance at all costs, even if a temporarily suboptimal
   trajectory (e.g., pushing the object around an obstacle) would lead to
   faster completion.

4. **Multi-objective gradient interference**: Position and rotation rewards
   operate on different scales.  A 5 cm position improvement at 25 cm
   distance earns `3.0 * 0.05 / 0.25 = 0.60`.  A 0.5 rad rotation
   improvement at 1.5 rad distance earns `3.0 * 0.5 / 1.5 = 1.00`.
   The critic network receives a scalarised sum and cannot separate which
   action component drove which reward component.

5. **Position-only termination blocks rotation learning**: `check_done`
   terminates the episode when `pos_err < 0.05 m`.  The agent collects +5
   and resets before it ever gets a chance to fix rotation.  Episodes are
   too short to learn rotation refinement.

6. **Euler-angle discontinuity**: `_yaw_distance_rad` uses modular
   arithmetic with conditional branches.  While correct, it produces
   gradient discontinuities at the pi/-pi boundary that the neural
   network must learn to handle.

### Why Push-ASP Bob fails entirely

Detailed investigation revealed that the Push-ASP Bob's reward structure
differs from Push-PPO in critical ways, despite using the same dense formula:

1. **Completion bonus requires BOTH pos AND rot**: The sparse `+5`
   in `wrapper_push_asp.py` gates on `(pos_err < 0.05) & (rot_err < 0.2)`.
   Push-PPO gates on position only.  This makes the +5 ~100x harder to
   trigger for ASP Bob.

2. **No `check_done` for Bob**: Bob runs all 10 pushes regardless of
   tipping, launching, or success.  After reaching the goal on push 3,
   pushes 4-10 push the object away.  This random-walk effect explains the
   0.55-0.90 m PosError in the logs.

3. **Missing reward components**: Bob's dense reward has no completion
   bonus, no rotation sub-bonus, and no tip penalty.  These are present
   in Push-PPO's unified `compute_push_reward()`.

4. **Latent bugs**: `bob_done_now` is not set for early completions, and
   Bob's LSTM hidden state is not zeroed when early completion fires.
   These affect ~0.07% of transitions (negligible now but would matter if
   SR improved).

---

## 2. Root-Cause Analysis

### 2.1 The oscillation problem (Push-PPO)

The fractional improvement formula `alpha * (d_prev - d_now) / d_prev`
is not potential-based.  For a push sequence A -> B -> A:

```
Step 1 (A -> B): reward = alpha * (d_A - d_B) / d_A
Step 2 (B -> A): reward = alpha * (d_B - d_A) / d_B
```

Sum = `alpha * (d_A - d_B) * (1/d_A - 1/d_B)`.  If `d_B < d_A` (the agent
moved closer then back), both factors have the same sign, so the sum is
**positive**.  The agent profits from oscillation.  With proper PBRS using
`gamma * Phi(s') - Phi(s)`, the sum is:

```
Step 1: gamma * Phi(B) - Phi(A)
Step 2: gamma * Phi(A) - Phi(B)
Sum = (gamma - 1) * (Phi(B) - Phi(A))
```

With `gamma < 1`, the sum is negative if `Phi(B) > Phi(A)` (state B is
closer to goal), so oscillation is penalised.  With `gamma = 1`, the sum
is exactly zero — no profit, no loss.  Oscillation is strictly eliminated.

### 2.2 The sparse-signal starvation problem (Push-ASP)

At early training, Alice produces goals with ~0.04 m displacement.  Most
are invalid.  The few valid goals are "shallow" (0.05-0.10 m).  Bob's
sparse reward requires BOTH position AND rotation thresholds, which
essentially never fires.  The dense reward provides gradient but the
distance penalty produces consistently negative mean returns (-0.80/push),
and the critic learns to predict negative values for all states.

### 2.3 The Euler-angle boundary fear

The `_yaw_distance_rad` function:

```python
diff = (yaw_a - yaw_b) % (2 * pi)
diff = where(diff > pi, 2*pi - diff, diff)
```

This is mathematically correct but creates a gradient discontinuity at the
`pi` boundary.  When `diff` crosses `pi`, the derivative flips sign
instantaneously.  The neural network observes this as a cliff in the loss
landscape and learns to avoid rotations near the boundary.

The cosine distance `(1 - cos(theta_target - theta_current)) / 2` is
everywhere smooth, bounded in [0, 1], and monotonically related to the
true angular distance.

---

## 3. Theoretical Foundations

### 3.1 Potential-Based Reward Shaping (PBRS)

Ng et al. (1999) proved that adding a shaping reward:

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

to a base reward `R(s, a, s')` preserves the optimal policy of the
underlying MDP for any potential function `Phi: S -> R`.

Key properties:
- **Policy invariance**: The optimal policy under `R + F` is the same as
  under `R` alone.
- **Zero-sum over cycles**: Any cyclic trajectory sums to
  `(gamma^T - 1) * Phi(s_0)`, which is non-positive for `gamma < 1`.
- **No reward hacking**: The agent cannot profit from oscillation.

### 3.2 Bounded exponential potentials

A linear potential `Phi(s) = -d(s)` is unbounded, making the critic's
approximation task harder.  An exponential potential:

```
Phi_pos(s) = exp(-k_p * d(s)^2)
Phi_rot(s) = exp(-k_r * c(s))
```

where `c(s) = (1 - cos(theta_target - theta_current)) / 2` is strictly
bounded in [0, 1].

The exponential form provides:
- Gentle gradient when far from goal (broad exploration)
- Steep gradient near the goal (precise refinement)
- No singularities or infinities
- Bounded critic targets

### 3.3 Temperature parameter selection

The temperature `k` controls the width of the reward gradient.  It must
be tuned to the workspace scale.

For this task the workspace is ~1 m across and typical goal distances
range from 0.05 m to 0.50 m.  We need meaningful signal across this
entire range.

**Position** (`Phi_pos = exp(-k_p * d^2)`):

| k_p | d=0.30 m | d=0.20 m | d=0.10 m | d=0.05 m |
|-----|----------|----------|----------|----------|
| 100 | 0.0001   | 0.018    | 0.368    | 0.779    |
| 50  | 0.011    | 0.135    | 0.607    | 0.882    |
| 20  | 0.165    | 0.449    | 0.819    | 0.951    |
| 15  | 0.259    | 0.549    | 0.861    | 0.963    |
| 10  | 0.405    | 0.670    | 0.905    | 0.975    |

`k_p = 100` (cited in Fetch literature) gives near-zero potential at
0.30 m — no learning signal for typical starting distances.
`k_p = 15` provides gradient across the full workspace.

**Rotation** (`Phi_rot = exp(-k_r * c)`, where `c = (1-cos)/2 in [0,1]`):

| k_r | c=0.5 (90 deg) | c=0.1 (36 deg) | c=0.02 (11 deg) | c=0.0 (aligned) |
|-----|----------------|----------------|-----------------|-----------------|
| 10  | 0.007          | 0.368          | 0.819           | 1.000           |
| 5   | 0.082          | 0.607          | 0.905           | 1.000           |
| 3   | 0.223          | 0.741          | 0.942           | 1.000           |

`k_r = 5` gives good gradient across the typical rotation range.

### 3.4 Cosine angular distance

For planar pushing (SE(2)), only yaw matters.  The cosine distance:

```
c = (1 - cos(theta_target - theta_current)) / 2
```

Properties:
- Range: [0, 1] (0 = aligned, 1 = 180 deg opposite)
- Continuous and smooth everywhere (no wrap-around discontinuity)
- Monotonic with true angular distance
- Differentiable at all points (unlike modular arithmetic with conditionals)

This replaces `_yaw_distance_rad` in both wrappers.

---

## 4. Reward Architecture Design

### 4.1 Dense reward (PBRS)

```
Phi_pos(s) = exp(-k_p * ||p_current - p_target||^2)         k_p = 15.0
Phi_rot(s) = exp(-k_r * (1 - cos(yaw_current - yaw_target)) / 2)   k_r = 5.0

r_dense_pos = gamma * Phi_pos(s') - Phi_pos(s)
r_dense_rot = gamma * Phi_rot(s') - Phi_rot(s)

r_dense = w_pos * r_dense_pos + w_rot * r_dense_rot
```

where `gamma = 0.95` (PPO discount factor).

**No distance penalty**.  The `-beta * d_now` and `-beta_rot * y_now` terms
are removed.  They are not part of the PBRS framework and change the
optimal policy.

**No division by `d_prev`**.  The fractional formula is entirely replaced.

### 4.2 Sparse reward (task objective)

```
r_sparse_pos = +5.0  if pos_err < 0.05 m  (first time per episode)
r_sparse_rot = +2.0  if pos_err < 0.05 m AND cos_rot_err < delta_rot  (first time per episode)
```

The rotation bonus is **gated behind position**: the agent cannot earn
rotation points while the object is far from the target.  This enforces
a natural curriculum (position first, rotation second).

### 4.3 Penalties and termination

```
r_tip = -5.0    if |roll| > 0.3 rad OR |pitch| > 0.3 rad    -> terminate
r_launch = -5.0 if obj_z > 0.10 m                            -> terminate
r_oob = -5.0    if ||obj_pos - goal_pos|| > 0.50 m           -> terminate
r_table = -5.0  if tcp_z < -0.01 m (arm through table)       -> terminate
```

All penalties trigger immediate episode termination.  This prevents further
data collection in unrecoverable states and gives the agent a clean
negative signal.

### 4.4 Episode termination conditions

```
done = terminated                              # physics termination
     | (push_count >= max_pushes)              # budget exhausted
     | tipped                                  # object unrecoverable
     | launched                                # object airborne
     | out_of_bounds                           # object too far from goal
     | arm_through_table                       # robot collision
     | (pos_success AND rot_success)           # BOTH thresholds met
```

The episode ends when BOTH position and rotation are correct.  Position-only
success gives +5 but does NOT terminate — the agent continues pushing to
fix rotation.  Only achieving both triggers the +2 bonus and episode end.

### 4.5 Total reward per push

```
R(t) = w_pos * r_dense_pos(t) + w_rot * r_dense_rot(t)
     + r_sparse_pos + r_sparse_rot
     + r_penalties
```

### 4.6 Scaling coefficients

The PBRS dense terms produce values in approximately [-0.3, +0.3] per push.
The sparse bonuses are +5 and +2.  The penalties are -5.  To ensure the
dense signal is meaningful relative to sparse:

```
w_pos = 10.0    # scales PBRS pos to approximately [-3, +3] range
w_rot = 10.0    # scales PBRS rot to approximately [-3, +3] range
```

These can be tuned.  The key property of PBRS is that any scalar multiple
of the potential difference is also a valid shaping reward (it changes
learning speed but not the optimal policy).

---

## 5. Three Experimental Models

### Model A: PBRS rewards only (no forced curriculum)

**Hypothesis**: The PBRS dense reward combined with tiered sparse bonuses
provides sufficient gradient for a single agent to learn both position
and rotation simultaneously, without any curriculum scheduling.

**Script**: `train_push.py` (Push-PPO baseline)
**Wrapper**: `wrapper_push.py` (modified)

**Changes from current baseline**:
- Replace fractional improvement with PBRS dense (Section 4.1)
- Replace `_yaw_distance_rad` with cosine distance
- Remove `-beta * d_now` and `-beta_rot * y_now` penalties
- Keep +5 position-only bonus (NO termination on position-only)
- Add +2 both-threshold bonus (WITH termination)
- Add arm-through-table termination
- `check_done` terminates on: both-success, tip, launch, OOB, table, max_pushes
- `w_pos = 10.0, w_rot = 10.0` (equal weighting from start)

**Expected behavior**:
- Early training: agent learns to push toward goal (position gradient)
- Position and rotation improve simultaneously (no forced staging)
- +5 fires when position is correct, episode continues for rotation
- +2 fires when both are correct, episode terminates cleanly
- No oscillation (PBRS guarantee)

**Risk**: Without curriculum gating, the rotation gradient may interfere
with position learning at the start.  The agent tries to optimise both
simultaneously from push 1 and may converge slower than a staged approach.

### Model B: PBRS rewards + forced curriculum

**Hypothesis**: Sequentially gating the rotation reward behind a position
competency threshold accelerates convergence by allowing the agent to
master position first, then add rotation as a refinement objective.

**Script**: `train_push.py` (Push-PPO baseline)
**Wrapper**: `wrapper_push.py` (modified)

**Changes from Model A**:
- Add curriculum controller in the training loop
- Track EMA of position error across iterations
- `w_rot` starts at 0.0 and linearly ramps to 10.0 when
  `ema_pos_err < 0.08 m` (sustained across 50 iterations)
- Ramp duration: 200 iterations (smooth transition)
- The sparse +2 rotation bonus is also gated behind the same curriculum
  flag (not awarded until the rotation reward is active)
- Log `w_rot` and `curriculum_phase` to TensorBoard

**Curriculum phases**:

```
Phase 1 (position only):
  w_rot = 0.0
  Sparse: +5 for pos < 0.05 m (no termination)
  Termination: tip, launch, OOB, table, max_pushes, pos-only (at-goal) termination ACTIVE
  Agent learns pure translation.

Phase 2 (transition):
  EMA pos_err < 0.08 m triggers ramp
  w_rot linearly increases from 0.0 to 10.0 over 200 iterations
  Sparse +2 for both thresholds enabled (with termination)
  Position-only termination REMOVED (episode continues for rotation)
  Agent begins learning rotation while maintaining position skill.

Phase 3 (full):
  w_rot = 10.0 (stable)
  Full reward: dense pos + dense rot + sparse pos + sparse rot
  Termination: both-success, tip, launch, OOB, table, max_pushes
```

**Expected behavior**:
- Phase 1: rapid position learning (same as current Push-PPO but with PBRS)
- Phase 2: rotation signal smoothly introduced, agent refines orientation
- Phase 3: full multi-objective optimisation

**Risk**: The phase transition may cause a temporary performance dip as the
critic re-learns value estimates with the new rotation component.  The
ramp duration (200 iterations) should smooth this.  Also, the switch from
position-only termination (Phase 1) to both-required termination (Phase 2)
changes the episode length distribution, which affects GAE.

### Model C: PBRS rewards + ASP curriculum (Alice produces the curriculum)

**Hypothesis**: Alice's adversarial goal generation provides a natural,
emergent curriculum that replaces explicit position/rotation staging.
As Alice learns to create harder goals, Bob is continuously challenged
at the frontier of his capability.

**Script**: `train_push_asp.py` (Push-ASP)
**Wrapper**: `wrapper_push_asp.py` (modified)

**Changes from current Push-ASP**:
- Replace Bob's dense reward with PBRS (Section 4.1)
- Replace `_yaw_distance_rad` with cosine distance
- Remove distance/rotation penalties
- Bob's sparse: +5 position-only (no termination) + +2 both (terminate Bob's phase)
- Add `check_done` for Bob: tip, launch, OOB, arm-through-table (terminate Bob's phase early)
- Fix `bob_done_now` for early completions (set True when +2 fires)
- Fix LSTM hidden state zeroing on early completion
- Match approach radius: `max_r = 0.08` (was 0.15)
- `w_pos = 10.0, w_rot = 10.0` (no curriculum gating — Alice IS the curriculum)
- Alice's reward stays outcome-based (unchanged)

**Bob's full reward per push**:
```
r_dense = 10.0 * (gamma * Phi_pos(s') - Phi_pos(s))
        + 10.0 * (gamma * Phi_rot(s') - Phi_rot(s))
r_sparse = +5.0  (first time pos < 0.05 m, no phase termination)
         + +2.0  (first time pos < 0.05 AND rot < threshold, TERMINATE Bob's phase)
r_penalties = -5.0 for tip/launch/OOB/table (TERMINATE Bob's phase)

R(t) = r_dense + r_sparse + r_penalties
```

**Bob's phase termination**:
```
bob_phase_done = phase_step >= bob_pushes          # budget
               | tipped                             # unrecoverable
               | launched                           # airborne
               | out_of_bounds                      # too far
               | arm_through_table                  # collision
               | (pos_success AND rot_success)      # solved
```

When Bob's phase ends early (completion or catastrophe):
1. Set `bob_done_now[env_ids] = True`
2. Zero Bob's LSTM hidden state for those envs
3. Transition to Alice or reset episode (existing logic)

**Alice's curriculum role**:
- Alice starts with random pushes producing shallow goals (~0.04 m)
- As Alice improves, goals become more challenging
- Bob's PBRS reward provides gradient at ANY goal distance (no sparse starvation)
- The exponential potential gives Bob meaningful signal even for
  Alice's initial shallow goals (0.05-0.10 m displacement)
- As Bob improves, Alice must create harder goals to earn her outcome reward

**Expected behavior**:
- Early training: Alice produces easy goals, Bob learns basic pushing via PBRS
- Mid training: Alice creates progressively harder goals (larger displacement, rotation)
- Late training: Alice and Bob co-evolve, Bob achieves position + rotation

**Risk**: Alice's learning speed gates Bob's exposure.  If Alice learns too
slowly, Bob is starved of challenging goals.  If Alice learns too fast, Bob
faces goals beyond his current capability.  The PBRS reward mitigates the
latter — Bob gets gradient signal at any distance, not just within threshold.
The former is inherent to ASP and controlled by Alice's entropy coefficient
and learning rate.

---

## 6. Implementation Plan

### 6.1 Shared changes (all three models)

#### 6.1.1 New utility: `reward_pbrs.py`

Create `asyncDualPlayPPO/tasks/utils/reward_pbrs.py`:

```python
"""
Potential-Based Reward Shaping (PBRS) for push-primitive tasks.

Implements bounded exponential potentials with cosine angular distance.
Policy-invariant shaping guarantee (Ng et al. 1999).
"""

def cosine_rot_error(yaw_current, yaw_target):
    """
    Cosine angular distance, bounded in [0, 1].
    0 = aligned, 1 = 180 deg opposite.
    Smooth everywhere, no wrap-around discontinuity.
    """
    return (1.0 - torch.cos(yaw_target - yaw_current)) / 2.0

def potential_pos(obj_pos, goal_pos, k_p=15.0):
    """
    Bounded exponential position potential.
    Returns values in [0, 1]. 1 = at goal, 0 = far away.
    """
    d_sq = ((obj_pos[..., :2] - goal_pos[..., :2]) ** 2).sum(dim=-1)
    return torch.exp(-k_p * d_sq)

def potential_rot(yaw_current, yaw_target, k_r=5.0):
    """
    Bounded exponential rotation potential.
    Returns values in [0, 1]. 1 = aligned, 0 = opposite.
    """
    c = cosine_rot_error(yaw_current, yaw_target)
    return torch.exp(-k_r * c)

def pbrs_reward(phi_prev, phi_now, gamma=0.95):
    """
    PBRS shaping: gamma * Phi(s') - Phi(s).
    """
    return gamma * phi_now - phi_prev
```

#### 6.1.2 Cosine rotation metric

Replace all uses of `_yaw_distance_rad` in reward computation with
`cosine_rot_error`.  Keep `_yaw_distance_rad` for logging/metrics only
(it still reports a meaningful radian-scale error for human inspection).

The cosine-based threshold for rotation success replaces `rot_err < 0.2 rad`:

```
cos_threshold = (1 - cos(0.2)) / 2 = 0.00997
```

So `rot_success = cosine_rot_error(yaw_cur, yaw_goal) < 0.01`.

#### 6.1.3 Arm-through-table detection

In both training loops, after each waypoint substep, check:

```python
tcp_z = _tcp_pos_local()[:, 2]
arm_through_table = tcp_z < -0.01  # 1 cm below table surface
```

This triggers the same handling as `terminated` (hold joints, apply penalty).

### 6.2 Model A specific changes

#### Files modified:
- `asyncDualPlayPPO/tasks/utils/wrapper_push.py`
- `asyncDualPlayPPO/train_push.py`

#### `wrapper_push.py` changes:

1. **Import `reward_pbrs`** at the top.

2. **Add potential state tracking**:
   ```python
   self.prev_phi_pos = None  # set in capture_pre_push
   self.prev_phi_rot = None
   ```

3. **Modify `capture_pre_push`** to compute and store potentials:
   ```python
   def capture_pre_push(self, obs):
       self.prev_obj_pos = self._get_obj_pos(obs).clone()
       self.prev_obj_euler = self._get_obj_euler(obs).clone()
       goal_pos = self._get_goal_pos(obs)
       goal_euler = self._get_goal_euler(obs)
       self.prev_phi_pos = potential_pos(self.prev_obj_pos, goal_pos, k_p=self.k_p)
       self.prev_phi_rot = potential_rot(
           self.prev_obj_euler[..., 2], goal_euler[..., 2], k_r=self.k_r
       )
   ```

4. **Replace `compute_push_reward`** with new PBRS version:
   ```python
   def compute_push_reward(self, obs, w_pos=10.0, w_rot=10.0):
       cur_obj_pos = self._get_obj_pos(obs)
       cur_obj_euler = self._get_obj_euler(obs)
       goal_pos = self._get_goal_pos(obs)
       goal_euler = self._get_goal_euler(obs)

       # PBRS dense
       phi_pos_now = potential_pos(cur_obj_pos, goal_pos, k_p=self.k_p)
       phi_rot_now = potential_rot(
           cur_obj_euler[..., 2], goal_euler[..., 2], k_r=self.k_r
       )
       dense_pos = pbrs_reward(self.prev_phi_pos, phi_pos_now, gamma=self.gamma)
       dense_rot = pbrs_reward(self.prev_phi_rot, phi_rot_now, gamma=self.gamma)
       reward = w_pos * dense_pos + w_rot * dense_rot

       # Sparse
       pos_err = (cur_obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1)
       rot_err = cosine_rot_error(cur_obj_euler[:, 2], goal_euler[:, 2])
       pos_success = pos_err < 0.05
       rot_success = rot_err < 0.01

       new_pos_completion = pos_success & ~self._gave_completion
       reward += new_pos_completion.float() * 5.0
       self._gave_completion |= new_pos_completion

       new_both = pos_success & rot_success & ~self._gave_rot_bonus
       reward += new_both.float() * 2.0
       self._gave_rot_bonus |= new_both

       # Tip penalty
       tipped = (cur_obj_euler[:, 0].abs() > 0.3) | (cur_obj_euler[:, 1].abs() > 0.3)
       reward += tipped.float() * (-5.0)

       # Store for metrics
       self.at_goal = pos_success & rot_success
       self._last_pos_err = pos_err
       self._last_rot_err = rot_err  # keep radian metric for logging
       self.push_count += 1
       return reward
   ```

5. **Modify `check_done`**:
   ```python
   def check_done(self, obs, terminated):
       max_pushes = self.push_count >= self.max_pushes_per_episode
       # Object catastrophes
       obj_z = obs[:, _OBS_ROBOT_DIM + 2]
       launched = obj_z > 0.10
       tipped = (obs[:, _OBS_ROBOT_DIM + 3].abs() > 0.3) | \
                (obs[:, _OBS_ROBOT_DIM + 4].abs() > 0.3)
       obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
       goal_pos = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM:
                       _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
       out_of_bounds = (obj_pos - goal_pos).norm(dim=-1) > 0.5
       # Success: BOTH position AND rotation
       both_success = self.at_goal
       return terminated | max_pushes | both_success | launched | tipped | out_of_bounds
   ```

   Note: position-only success (`at_goal_pos`) is removed from termination.

#### `train_push.py` changes:

1. Add arm-through-table check inside the waypoint loop:
   ```python
   tcp_z = _tcp_pos_local()[:, 2]
   arm_bad = tcp_z < -0.01
   terminated |= arm_bad
   ```

2. Pass `w_pos` and `w_rot` to `compute_push_reward`:
   ```python
   reward = env.compute_push_reward(obs, w_pos=10.0, w_rot=10.0)
   ```

3. Add new TensorBoard metrics:
   - `Metrics/PhiPos` (mean potential position)
   - `Metrics/PhiRot` (mean potential rotation)
   - `Metrics/DensePos` (mean PBRS position component)
   - `Metrics/DenseRot` (mean PBRS rotation component)

### 6.3 Model B specific changes

All Model A changes plus:

#### `train_push.py` additional changes:

1. **Curriculum controller**:
   ```python
   curriculum_active = False
   curriculum_ramp_start = None
   CURRICULUM_RAMP_ITERS = 200
   CURRICULUM_POS_THRESHOLD = 0.08  # EMA pos_err must be below this
   CURRICULUM_LOOKBACK = 50         # sustained for this many iterations
   ema_pos_err_hist = deque(maxlen=CURRICULUM_LOOKBACK)
   ```

2. **Per-iteration curriculum check**:
   ```python
   ema_pos_err_hist.append(mean_pos_err)
   if not curriculum_active and len(ema_pos_err_hist) == CURRICULUM_LOOKBACK:
       if all(e < CURRICULUM_POS_THRESHOLD for e in ema_pos_err_hist):
           curriculum_active = True
           curriculum_ramp_start = iteration
           print(f"[Curriculum] Phase 2: rotation reward ramp started at iter {iteration}")

   if curriculum_active:
       ramp_progress = min(1.0, (iteration - curriculum_ramp_start) / CURRICULUM_RAMP_ITERS)
       w_rot = 10.0 * ramp_progress
   else:
       w_rot = 0.0
   w_pos = 10.0
   ```

3. **Phase 1 termination override**:
   ```python
   if not curriculum_active:
       # Phase 1: terminate on position-only success (faster episodes)
       pos_only_success = env._last_pos_err < 0.05
       done = done | pos_only_success
   ```

4. **Phase 2 rotation bonus gating**:
   ```python
   reward = env.compute_push_reward(obs, w_pos=w_pos, w_rot=w_rot,
                                     enable_rot_sparse=curriculum_active)
   ```

   The wrapper's `compute_push_reward` gains an `enable_rot_sparse` parameter
   that gates the +2 both-threshold bonus.

5. **Log curriculum state**:
   ```python
   writer.add_scalar("Curriculum/w_rot", w_rot, iteration)
   writer.add_scalar("Curriculum/phase", 2 if curriculum_active else 1, iteration)
   ```

### 6.4 Model C specific changes

#### Files modified:
- `asyncDualPlayPPO/tasks/utils/wrapper_push_asp.py`
- `asyncDualPlayPPO/train_push_asp.py`

#### `wrapper_push_asp.py` changes:

1. **Import `reward_pbrs`**.

2. **Add potential tracking** (same as Model A).

3. **Replace `compute_bob_dense_push_reward`** with PBRS version:
   ```python
   def compute_bob_dense_push_reward(self, push_obs, w_pos=10.0, w_rot=10.0):
       cur_obj_pos = self._get_obj_pos(push_obs)
       cur_obj_euler = self._get_obj_euler(push_obs)
       goal_pos = self._get_goal_pos(push_obs)
       goal_euler = self._get_goal_euler(push_obs)

       phi_pos_now = potential_pos(cur_obj_pos, goal_pos, k_p=self.k_p)
       phi_rot_now = potential_rot(
           cur_obj_euler[..., 2], goal_euler[..., 2], k_r=self.k_r
       )
       dense_pos = pbrs_reward(self.prev_phi_pos, phi_pos_now, gamma=0.95)
       dense_rot = pbrs_reward(self.prev_phi_rot, phi_rot_now, gamma=0.95)
       return w_pos * dense_pos + w_rot * dense_rot
   ```

4. **Replace `compute_bob_push_reward`** (sparse):
   ```python
   def compute_bob_push_reward(self, push_obs):
       rewards = torch.zeros(self.num_envs, device=self.device)
       cur_obj_pos = self._get_obj_pos(push_obs)
       cur_obj_euler = self._get_obj_euler(push_obs)
       goal_pos = self._get_goal_pos(push_obs)
       goal_euler = self._get_goal_euler(push_obs)

       pos_err = (cur_obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1)
       rot_err = cosine_rot_error(cur_obj_euler[:, 2], goal_euler[:, 2])
       pos_success = pos_err < 0.05
       rot_success = rot_err < 0.01

       new_pos = pos_success & ~self._bob_gave_completion
       rewards += new_pos.float() * 5.0
       self._bob_gave_completion |= new_pos

       new_both = pos_success & rot_success & ~self._bob_gave_rot_bonus
       rewards += new_both.float() * 2.0
       self._bob_gave_rot_bonus |= new_both

       self._bob_at_goal = pos_success & rot_success
       return rewards
   ```

5. **Add `check_bob_done` method**:
   ```python
   def check_bob_done(self, push_obs, terminated):
       obj_euler = self._get_obj_euler(push_obs)
       tipped = (obj_euler[:, 0].abs() > 0.3) | (obj_euler[:, 1].abs() > 0.3)
       obj_z = push_obs[:, _OBS_ROBOT_DIM + 2]
       launched = obj_z > 0.10
       obj_pos = self._get_obj_pos(push_obs)
       goal_pos = self._get_goal_pos(push_obs)
       oob = (obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1) > 0.5
       both_success = self._bob_at_goal

       penalties = torch.zeros(self.num_envs, device=self.device)
       should_end = tipped | launched | oob | terminated
       penalties[should_end] = -5.0

       return should_end | both_success, penalties
   ```

6. **Add `_bob_gave_rot_bonus` buffer** initialised alongside
   `_bob_gave_completion` in `__init__` and cleared on phase reset.

#### `train_push_asp.py` changes:

1. **Add arm-through-table check** in waypoint loop (same as Model A).

2. **Add Bob phase early termination** after reward computation:
   ```python
   if len(bob_indices) > 0:
       bob_should_end, bob_end_penalties = env.check_bob_done(full_push_obs, terminated)
       bob_rewards += bob_end_penalties
       # Early termination: transition Bob out
       early_end_ids = torch.where(bob_should_end & is_bob & ~bob_done_mask)[0]
       if len(early_end_ids) > 0:
           bob_done_now[early_end_ids] = True
           # ... handle phase transition (same as handle_bob_phase_end)
   ```

3. **Fix `bob_done_now` for early completions** (existing bug):
   ```python
   if bob_achieved_completion.any():
       completion_ids = torch.where(bob_achieved_completion)[0]
       bob_progress_rew += env.handle_bob_early_success(completion_ids, full_push_obs)
       bob_done_now[completion_ids] = True  # FIX: was missing
   ```

4. **Fix LSTM zeroing for early completions**:
   ```python
   # After the existing bob_done_mask LSTM zero block:
   if bob_hidden is not None:
       early_done = bob_done_now & ~bob_done_mask  # envs that ended early
       if early_done.any():
           early_ids = torch.where(early_done)[0]
           bob_hidden[0][early_ids] = 0.0
           bob_hidden[1][early_ids] = 0.0
   ```

5. **Match approach parameters**:
   ```python
   min_r = 0.02   # was 0.03
   max_r = 0.08   # was 0.15
   max_l = 0.20   # was 0.25
   ```

---

## 7. Evaluation Protocol

### 7.1 Training configuration

All three models use the same environment and hyperparameters:

| Parameter | Value |
|-----------|-------|
| num_envs | 2048 |
| max_iterations | 3000 |
| push_nsteps (PPO rollout) | 15 |
| max_pushes_per_episode | 5 (Models A/B), 10 Bob pushes (Model C) |
| gamma | 0.95 |
| ent_coef | 0.002 |
| learning_rate | 3e-4 |
| k_p (position temperature) | 15.0 |
| k_r (rotation temperature) | 5.0 |
| w_pos | 10.0 |
| w_rot | 10.0 (Model A/C), 0.0 -> 10.0 curriculum (Model B) |

### 7.2 Metrics tracked

For all models, log to TensorBoard:

| Metric | Description |
|--------|-------------|
| Metrics/SuccessRate | Both pos AND rot within threshold |
| Metrics/PositionSR | Position-only success rate |
| Metrics/RotationSR | Rotation-only success rate (gated behind position) |
| Metrics/PosError | Mean position error (m) |
| Metrics/RotError | Mean cosine rotation error |
| Metrics/RotErrorRad | Mean yaw error (radians, for human readability) |
| Reward/Dense/Pos | Mean PBRS position component |
| Reward/Dense/Rot | Mean PBRS rotation component |
| Reward/Sparse/PosBonus | Mean +5 bonus |
| Reward/Sparse/RotBonus | Mean +2 bonus |
| Reward/Penalties | Mean penalty rewards |
| Metrics/EpisodeLength | Mean pushes per episode |
| Metrics/TermReason/* | Count per termination reason (success/tip/launch/OOB/table/budget) |

Model B additionally:
| Curriculum/w_rot | Current rotation weight |
| Curriculum/Phase | 1 or 2 |

Model C additionally:
| Metrics/Alice/* | Alice's existing metrics (ValidGoals, MeanDisp3D, etc.) |
| Metrics/Bob/PhaseLength | Mean Bob pushes before phase end |

### 7.3 Comparison analysis

After training, run `analyze_tensorboard.py` comparing all three:

```bash
python asyncDualPlayPPO/extras/analyze_tensorboard.py \
    --summary-dirs runs/pbrs_model_a/summary \
                   runs/pbrs_model_b/summary \
                   runs/pbrs_model_c/summary \
    --labels "A: PBRS only" "B: PBRS+Curriculum" "C: PBRS+ASP" \
    --csv
```

### 7.4 Success criteria

| Metric | Model A target | Model B target | Model C target |
|--------|----------------|----------------|----------------|
| SuccessRate (both) | > 0.30 | > 0.40 | > 0.15 |
| PositionSR | > 0.50 | > 0.60 | > 0.30 |
| RotationSR (gated) | > 0.40 | > 0.50 | > 0.20 |
| PosError | < 0.10 m | < 0.08 m | < 0.15 m |

Model C targets are lower because ASP inherently has fewer Bob training
episodes (Alice takes half the rollout) and goal quality depends on
Alice's learning progress.

### 7.5 Ablation studies (if time permits)

1. **k_p sensitivity**: Run Model A with k_p in {5, 10, 15, 20, 30}
2. **w_rot sensitivity**: Run Model A with w_rot in {5, 10, 20}
3. **Curriculum threshold**: Run Model B with threshold in {0.05, 0.08, 0.12}
4. **Pure sparse + HER**: If an off-policy implementation (SAC) is available,
   test pure sparse reward with HER relabeling for comparison

---

## Appendix A: Mathematical Reference

### A.1 PBRS policy invariance proof sketch

For MDP M with reward R, define shaped MDP M' with reward R' = R + F where
F(s, s') = gamma * Phi(s') - Phi(s).  For any policy pi:

```
V'_pi(s) = V_pi(s) + Phi(s)
Q'_pi(s, a) = Q_pi(s, a) + Phi(s)
```

Since Phi(s) is added uniformly to all actions in state s, the
arg max is preserved: `pi* under R' = pi* under R`.

### A.2 Why the fractional formula is not PBRS

The fractional improvement `alpha * (d_prev - d_now) / d_prev` can be
rewritten as:

```
alpha * (1 - d_now / d_prev)
```

This depends on BOTH `d_prev` and `d_now`, not on a potential evaluated
at individual states.  No function `Phi(s)` exists such that:

```
gamma * Phi(s') - Phi(s) = alpha * (1 - d(s') / d(s))
```

for all state pairs, because the right-hand side is not separable into
functions of `s'` and `s` individually (due to the ratio `d(s')/d(s)`).

### A.3 Cosine distance vs modular arithmetic

For yaw angles a, b:

```
Modular:  diff = |((a - b + pi) % 2pi) - pi|     # range [0, pi]
Cosine:   diff = (1 - cos(a - b)) / 2              # range [0, 1]
```

Relationship: `cosine_diff = (1 - cos(modular_diff)) / 2`

The cosine form is preferred because:
- No modular arithmetic (% operator has discontinuous gradient)
- No conditional (where/if for the pi wraparound)
- Bounded in [0, 1] (vs [0, pi] for modular)
- Smooth second derivative everywhere
