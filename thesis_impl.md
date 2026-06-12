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

1. **Non-separable state-pair dependence**: The reward `alpha * (d_prev - d_now) / d_prev`
   depends on both the previous and current state through the `1/d_prev` ratio.
   Two identical pushes producing identical outcomes yield different rewards
   depending on what `d_prev` was.  This violates the Markov property in the
   reward signal — the critic must implicitly model trajectory history, not
   just states, making value function approximation harder.

2. **Asymmetric gradient amplification near the goal**: A 1 cm improvement
   from 5 cm earns `3.0 * 0.01 / 0.05 = 0.60`, but the same 1 cm from
   30 cm earns `3.0 * 0.01 / 0.30 = 0.10` — a 6x difference for identical
   physical actions.  Near the goal, reward noise from physics (PhysX object
   jitter, IK tolerance, contact dynamics) is amplified by the same 6x
   factor, making the critic's approximation task harder and GAE advantage
   estimates noisier.

3. **Penalty terms bias the optimal policy**: The `-beta * d_now` and
   `-beta_rot * y_now` terms are state-dependent rewards not part of the
   task objective.  They incentivise "be close right now" rather than "reach
   the goal eventually."  A temporarily suboptimal trajectory (e.g.,
   approaching from a better angle for rotation) is penalised even if it
   leads to faster completion.  No potential function exists that reproduces
   this penalty — it fundamentally changes the optimal policy relative to
   the sparse-reward task.

4. **No formal optimality guarantee**: Without PBRS structure, there is no
   theorem ensuring the shaped optimal policy matches the sparse-reward
   optimal policy.  The agent can converge to a reward-hacking local optimum
   that maximises the dense signal without solving the sparse task.

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

### 2.1 The fractional formula is not potential-based

The fractional improvement formula `alpha * (d_prev - d_now) / d_prev`
is **not** a potential-based shaping reward, but — contrary to initial
intuition — it does **not** reward oscillation.  For a push sequence
A -> B -> A where `d_B < d_A`:

```
Step 1 (A -> B): reward = alpha * (d_A - d_B) / d_A      (positive)
Step 2 (B -> A): reward = alpha * (d_B - d_A) / d_B      (negative)
```

Since `d_B < d_A`, we have `1/d_B > 1/d_A`, so `|Step 2| > |Step 1|`.
The return trip always costs more than the forward trip earned.  The
formula is inherently anti-oscillation.

The actual problems are structural, not about exploitation:

1. **Non-separable state-pair dependence**: The reward for transitioning
   from state s to s' depends on `d(s)` in the denominator.  This means
   two identical actions producing identical outcomes from identical states
   will receive different rewards if the *prior* state differed (because
   `d_prev` was set at the prior push).  No potential function `Phi(s)`
   can reproduce `alpha * (1 - d(s')/d(s))` because the right-hand side
   is not separable into functions of `s` and `s'` individually (the
   `d(s')/d(s)` ratio couples them).

2. **Gradient amplification near the goal**: The `1/d_prev` factor
   amplifies both signal and noise near the goal by up to 6x relative
   to the far-field, making value function approximation harder.

3. **Penalty terms change the optimal policy**: The `-beta * d_now` and
   `-beta_rot * y_now` terms are not potential differences.  They add
   a per-step cost to being far from the goal, which can bias the policy
   away from temporarily-suboptimal-but-globally-better trajectories
   (e.g., approaching from a better angle).

### 2.2 The sparse-signal starvation problem (Push-ASP)

At early training, Alice produces goals with ~0.04 m displacement.  Most
are invalid.  The few valid goals are "shallow" (0.05-0.10 m).  Bob's
sparse reward requires BOTH position AND rotation thresholds, which
essentially never fires.  The dense reward provides gradient but the
distance penalty produces consistently negative mean returns (-0.80/push),
and the critic learns to predict negative values for all states.

### 2.3 The Euler-angle boundary

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

### 3.2 Episodic PBRS with gamma_shaping = 1.0

The original Ng et al. (1999) PBRS uses the MDP's discount factor gamma.
With `gamma = 0.95`, this creates a "discounting tax" near the goal:
when `Phi(s)` is already high (agent is close), the shaping reward
`0.95 * Phi(s') - Phi(s)` is negative even when making progress, because
the 5% discount exceeds the marginal potential improvement.

For **episodic** MDPs (which push-primitive tasks are — they terminate on
success, failure, or budget), Grzes & Kudenko (2009) proved that using
`gamma_shaping = 1.0` preserves policy invariance:

```
F(s, s') = Phi(s') - Phi(s)
```

Properties with gamma_shaping = 1.0:
- Cycles sum to exactly zero (A -> B -> A earns `Phi(B) - Phi(A) + Phi(A) - Phi(B) = 0`)
- No near-goal sign inversion (every improvement produces positive signal)
- Valid for episodic MDPs with terminal states (policy-invariant)
- Distinct from the PPO discount `gamma = 0.95` used in GAE

This is the recommended formulation for this implementation.

### 3.3 Bounded exponential potentials

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

### 3.4 Temperature parameter selection

The temperature `k` controls the width of the reward gradient.  It must
be tuned to the workspace scale.

For this task the workspace is ~1 m across and typical goal distances
range from 0.05 m to 0.50 m.  We need meaningful signal across this
entire range.

**Position** (`Phi_pos = exp(-k_p * d^2)`, using `F = Phi(s') - Phi(s)`):

The reward for a 5 cm improvement at various starting distances, scaled
by `w_pos = 10.0`:

| k_p | 0.30→0.25 m | 0.20→0.15 m | 0.10→0.05 m | 0.05→0.00 m |
|-----|-------------|-------------|-------------|-------------|
| 15  | 1.36 | 0.89 | 1.02 | 0.37 |
| 30  | 1.03 | 2.34 | 1.87 | 0.72 |
| 50  | 0.33 | 1.90 | 2.75 | 1.18 |

`k_p = 30` provides the best balance:
- Far-field (0.30 m): ~1.0 reward per 5 cm push (meaningful gradient)
- Mid-field (0.15 m): ~2.3 reward (strong signal)
- Near-goal (0.05 m): ~0.7 for the final 5 cm (+5 sparse bonus compensates)
- No sign inversion at any distance (gamma_shaping = 1.0)

`k_p = 15` gives only 0.37 for the critical final 5 cm.
`k_p = 50` gives only 0.33 for the 30→25 cm range (weak far-field signal).

**Recommended: k_p = 30.**

**Rotation** (`Phi_rot = exp(-k_r * c)`, where `c = (1-cos)/2 in [0,1]`):

| k_r | c=0.5 (90 deg) | c=0.1 (36 deg) | c=0.02 (11 deg) | c=0.0 (aligned) |
|-----|----------------|----------------|-----------------|-----------------|
| 10  | 0.007          | 0.368          | 0.819           | 1.000           |
| 5   | 0.082          | 0.607          | 0.905           | 1.000           |
| 3   | 0.223          | 0.741          | 0.942           | 1.000           |

`k_r = 5` gives good gradient across the typical rotation range.

### 3.5 Cosine angular distance

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

### 4.1 Dense reward (PBRS, gamma_shaping = 1.0)

```
Phi_pos(s) = exp(-k_p * ||p_current - p_target||_2D^2)         k_p = 30.0
Phi_rot(s) = exp(-k_r * (1 - cos(yaw_current - yaw_target)) / 2)   k_r = 5.0

r_dense_pos = Phi_pos(s') - Phi_pos(s)          # gamma_shaping = 1.0
r_dense_rot = Phi_rot(s') - Phi_rot(s)          # gamma_shaping = 1.0

r_dense = w_pos * r_dense_pos + w_rot * r_dense_rot
```

Properties:
- Strictly zero-sum over any cycle (no exploitation possible)
- Positive for any improvement at any distance
- No near-goal sign inversion
- Bounded: max single-push reward ~ w * 1.0 (when Phi goes from 0 to 1)
- Policy-invariant for episodic MDPs (Grzes & Kudenko 2009)

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

The PBRS dense terms with `k_p = 30, k_r = 5` and `gamma_shaping = 1.0`
produce values in approximately [-0.2, +0.3] per push for typical 3-5 cm
push distances.  Scaling by `w = 10.0` yields [-2.0, +3.0] per push.

```
w_pos = 10.0    # scales PBRS pos to approximately [-2, +3] range
w_rot = 10.0    # scales PBRS rot to approximately [-2, +3] range
```

The key property of PBRS is that any scalar multiple of the potential
difference is also a valid shaping reward (it changes learning speed
but not the optimal policy).

---

## 5. Three Experimental Models

### Model A: PBRS rewards only (no forced curriculum)

**Hypothesis**: The PBRS dense reward combined with tiered sparse bonuses
provides sufficient gradient for a single agent to learn both position
and rotation simultaneously, without any curriculum scheduling.

**Script**: `train_push.py --pbrs` (Push-PPO baseline)
**Wrapper**: `wrapper_push.py` (modified)

**Changes from current baseline**:
- Replace fractional improvement with PBRS dense (Section 4.1)
- Replace `_yaw_distance_rad` with cosine distance in reward computation
- Remove `-beta * d_now` and `-beta_rot * y_now` penalties
- Keep +5 position-only bonus (NO termination on position-only)
- Add +2 both-threshold bonus (WITH termination)
- Add arm-through-table termination
- `check_done` terminates on: both-success, tip, launch, OOB, table, max_pushes
- `w_pos = 10.0, w_rot = 10.0` (equal weighting from start)
- `k_p = 30.0, k_r = 5.0, gamma_shaping = 1.0`

**Expected behavior**:
- Early training: agent learns to push toward goal (position gradient)
- Position and rotation improve simultaneously (no forced staging)
- +5 fires when position is correct, episode continues for rotation
- +2 fires when both are correct, episode terminates cleanly
- No exploitation of the dense signal (PBRS guarantee)

**Risk**: Without curriculum gating, the rotation gradient may interfere
with position learning at the start.  The agent tries to optimise both
simultaneously from push 1 and may converge slower than a staged approach.

### Model B: PBRS rewards + forced curriculum

**Hypothesis**: Sequentially gating the rotation reward behind a position
competency threshold accelerates convergence by allowing the agent to
master position first, then add rotation as a refinement objective.

**Script**: `train_push.py --pbrs --curriculum` (Push-PPO baseline)
**Wrapper**: `wrapper_push.py` (modified)

**Changes from Model A**:
- Add curriculum controller in the training loop
- Track EMA of position error across iterations
- `w_rot` starts at 0.0 and linearly ramps to 10.0 when
  `ema_pos_err < 0.08 m` (sustained across 50 iterations)
- Ramp duration: 200 iterations (smooth transition)
- The sparse +2 rotation bonus is also gated behind the same curriculum
  flag (not awarded until the rotation reward is active)
- Position-only termination gradually fades out during the ramp
- Log `w_rot` and `curriculum_phase` to TensorBoard

**Curriculum phases**:

```
Phase 1 (position only):
  w_rot = 0.0
  Sparse: +5 for pos < 0.05 m (terminates episode — fast iteration)
  Both-threshold +2 also terminates if triggered (rare in Phase 1)
  Agent learns pure translation with rapid episode turnover.

Phase 2 (transition, 200 iterations):
  Triggered when ema_pos_err < 0.08 m for 50 consecutive iterations.
  w_rot linearly increases from 0.0 to 10.0 over 200 iterations.
  Position-only termination threshold smoothly DECREASES:
      pos_term_threshold = 0.05 * (1.0 - ramp_progress)
  As ramp_progress -> 1.0, pos_term_threshold -> 0.0 (effectively disabled).
  Both-threshold termination always active.
  Net effect: as w_rot increases, pos-only termination fades out,
  giving the agent progressively more time for rotation refinement.

Phase 3 (full):
  w_rot = 10.0 (stable)
  pos_term_threshold = 0.0 (disabled)
  Episode terminates on: both-success, tip, launch, OOB, table, max_pushes
  Full multi-objective optimisation.
```

**Expected behavior**:
- Phase 1: rapid position learning (same as current Push-PPO but with PBRS)
- Phase 2: rotation signal smoothly introduced while position-only
  termination smoothly disabled — episode length increases organically
- Phase 3: full multi-objective optimisation

**Risk**: The critic must adapt to the changing reward landscape during
Phase 2.  The 200-iteration ramp and smooth termination threshold change
should smooth this transition, but a temporary performance dip is possible.

### Model C: PBRS rewards + ASP curriculum (Alice produces the curriculum)

**Hypothesis**: Alice's adversarial goal generation provides a natural,
emergent curriculum that replaces explicit position/rotation staging.
As Alice learns to create harder goals, Bob is continuously challenged
at the frontier of his capability.

**Script**: `train_push_asp.py --pbrs` (Push-ASP)
**Wrapper**: `wrapper_push_asp.py` (modified)

**Changes from current Push-ASP**:
- Replace Bob's dense reward with PBRS (Section 4.1)
- Replace `_yaw_distance_rad` with cosine distance
- Remove distance/rotation penalties
- Bob's sparse: +5 position-only (no phase termination) + +2 both (terminate Bob's phase)
- Add `check_bob_done` method: tip, launch, OOB, arm-through-table (terminate Bob's phase early)
- Fix `bob_done_now` for early completions (set True when +2 fires)
- Fix LSTM hidden state zeroing on early completion
- Remove `compute_bob_progress_reward` (redundant — PBRS provides per-push gradient)
- Match approach radius: `min_r = 0.02, max_r = 0.08, max_l = 0.20`
- `w_pos = 10.0, w_rot = 10.0` (no curriculum gating — Alice IS the curriculum)
- `k_p = 30.0, k_r = 5.0, gamma_shaping = 1.0`
- Alice's reward stays outcome-based (unchanged)

**Bob's full reward per push**:
```
r_dense = 10.0 * (Phi_pos(s') - Phi_pos(s))
        + 10.0 * (Phi_rot(s') - Phi_rot(s))
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

**Why `compute_bob_progress_reward` is removed**: The phase-end progress
reward (Fix P28) was added because sparse-only `{+1/-1/+5}` provided zero
per-push gradient.  With PBRS dense active on every push, Bob receives
meaningful directional signal at any distance.  The progress reward is
redundant and adds a non-PBRS component that breaks policy invariance.

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

Create `asyncDualPlayPPO/tasks/utils/reward_pbrs.py` with bounded exponential
potentials and cosine angular distance.  Uses `gamma_shaping = 1.0` (valid
for episodic MDPs, Grzes & Kudenko 2009).

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
arm_through_table = tcp_z < -0.01
```

This triggers the same handling as `terminated` (hold joints, apply penalty).

### 6.2 Model A & B: `--pbrs` and `--curriculum` flags in `train_push.py`

Both models use the same `wrapper_push.py` with PBRS methods (gated behind
`--pbrs` flag).  Model B additionally uses `--curriculum` to enable the
staged w_rot ramp and position-only termination in Phase 1.

### 6.3 Model C: `--pbrs` flag in `train_push_asp.py`

Enables PBRS dense reward for Bob, adds `check_bob_done`, fixes early
completion bugs, removes progress reward.

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
| gamma (PPO discount) | 0.95 |
| gamma_shaping | 1.0 (episodic PBRS) |
| ent_coef | 0.002 |
| learning_rate | 3e-4 |
| k_p (position temperature) | 30.0 |
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
| Metrics/CosRotError | Mean cosine rotation error |
| Metrics/RotErrorRad | Mean yaw error (radians, for human readability) |
| Reward/Dense/Pos | Mean PBRS position component |
| Reward/Dense/Rot | Mean PBRS rotation component |
| Reward/Sparse/PosBonus | Mean +5 bonus |
| Reward/Sparse/RotBonus | Mean +2 bonus |
| Reward/Penalties | Mean penalty rewards |
| Metrics/EpisodeLength | Mean pushes per episode |
| Metrics/TermReason/* | Count per termination reason |

Model B additionally:

| Metric | Description |
|--------|-------------|
| Curriculum/w_rot | Current rotation weight |
| Curriculum/Phase | 1 or 2 |
| Curriculum/pos_term_threshold | Position-only termination threshold |

Model C additionally:

| Metric | Description |
|--------|-------------|
| Metrics/Alice/* | Alice's existing metrics (ValidGoals, MeanDisp3D, etc.) |
| Metrics/Bob/PhaseLength | Mean Bob pushes before phase end |

### 7.3 Success criteria

| Metric | Model A target | Model B target | Model C target |
|--------|----------------|----------------|----------------|
| SuccessRate (both) | > 0.30 | > 0.40 | > 0.15 |
| PositionSR | > 0.50 | > 0.60 | > 0.30 |
| RotationSR (gated) | > 0.40 | > 0.50 | > 0.20 |
| PosError | < 0.10 m | < 0.08 m | < 0.15 m |

Model C targets are lower because ASP inherently has fewer Bob training
episodes (Alice takes half the rollout) and goal quality depends on
Alice's learning progress.

### 7.4 Ablation studies (if time permits)

1. **k_p sensitivity**: Run Model A with k_p in {10, 15, 30, 50, 100}
2. **w_rot sensitivity**: Run Model A with w_rot in {5, 10, 20}
3. **Curriculum threshold**: Run Model B with threshold in {0.05, 0.08, 0.12}
4. **gamma_shaping**: Run Model A with gamma_shaping in {0.95, 0.99, 1.0}

---

## Appendix A: Mathematical Reference

### A.1 PBRS policy invariance proof sketch

For MDP M with reward R, define shaped MDP M' with reward R' = R + F where
F(s, s') = Phi(s') - Phi(s) (gamma_shaping = 1.0, episodic MDP).

For any policy pi, the shaped return of a trajectory (s_0, s_1, ..., s_T) is:

```
G'_pi = sum_{t=0}^{T-1} gamma^t * [R(s_t, a_t, s_{t+1}) + Phi(s_{t+1}) - Phi(s_t)]
      = G_pi + sum_{t=0}^{T-1} gamma^t * [Phi(s_{t+1}) - Phi(s_t)]
```

The key insight is that in any state s, the shaping contribution
`Phi(s_{t+1}) - Phi(s_t)` depends only on the resulting state, not the
action taken.  Therefore:

```
Q'_pi(s, a) = Q_pi(s, a) + C(s)
```

where C(s) is action-independent.  Since `arg max_a Q'(s, a) = arg max_a Q(s, a)`,
the optimal policy is preserved.

For the episodic case with gamma_shaping = 1.0, Grzes & Kudenko (2009)
showed that policy invariance holds provided Phi(s_terminal) is constant.
Since our potentials satisfy Phi(at_goal) = 1.0 for both position and
rotation (constant for all terminal states), this condition is met.

### A.2 Why the fractional formula is not PBRS

The fractional improvement `alpha * (d_prev - d_now) / d_prev` can be
rewritten as:

```
alpha * (1 - d_now / d_prev)
```

This depends on BOTH `d_prev` and `d_now`, not on a potential evaluated
at individual states.  No function `Phi(s)` exists such that:

```
Phi(s') - Phi(s) = alpha * (1 - d(s') / d(s))
```

for all state pairs, because the right-hand side is not separable into
functions of `s'` and `s` individually (due to the ratio `d(s')/d(s)`).

Note: while the fractional formula naturally penalises oscillation (the
return trip costs more than the forward trip earns due to the smaller
denominator), it is still not policy-invariant.  The shaped optimal
policy may differ from the sparse-reward optimal policy.

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

### A.4 Equivalence of rotation thresholds

The current rotation success threshold is `rot_err < 0.2 rad` using
`_yaw_distance_rad`.  The equivalent cosine threshold:

```
cos_threshold = (1 - cos(0.2)) / 2 = (1 - 0.9801) / 2 = 0.00997 ~ 0.01
```

Note: the cosine mapping is nonlinear.  Errors near 0 are compressed
(0.1 rad -> cos = 0.0025) while errors near pi are expanded (3.0 rad ->
cos = 0.99).  For the success threshold (0.2 rad), the correspondence
is tight enough that the effective gate is unchanged.
