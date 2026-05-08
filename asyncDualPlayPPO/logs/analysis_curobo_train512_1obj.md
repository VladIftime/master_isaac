# cuRobo Training Analysis — 512 envs, 1 object

Two independent runs analysed. Both exhibit the same core pathology: Alice's curriculum
collapses because her reward signal is disconnected from Bob's outcome.

---

## 1. Run Summaries

| | Run A (28838810) | Run B (28838828) |
|---|---|---|
| **Resumed from** | iter 3230 | fresh |
| **Iterations** | 3 (cancelled) | 230 |
| **entropy_coef (resumed)** | 0.100 | — |
| **ABC buffer at start** | 500 full | 0 (building) |

---

## 2. Shared Symptoms

### 2.1 Bob Success Rate ≈ 0

| Run A iter | SR | Run B iter | SR |
|---|---|---|---|
| 3230 | 0.06% | 1 | 0.00% |
| 3231 | 0.06% | 17 | 0.00% |
| 3232 | 0.06% | 230 | 0.08% |

Bob's sparse {+1/−1/+5} reward structure provides no intermediate gradient for the
contact-manipulation task. The ABC buffer provides the only learning signal, but it
is **stale** (Run A: pre-filled from old training; Run B: poorly populated).

### 2.2 Alice's Mean Object Displacement is Declining or Static

| Run A | MeanDisp3D | Run B | MeanDisp3D |
|---|---|---|---|
| 3230 | 0.272 m | 1 | 0.046 m |
| 3231 | 0.250 m | 17 | 0.025 m |
| 3232 | 0.221 m | 230 | 0.030 m |

Alice is learning to move objects **less** over time, not more. She converges to
the minimum displacement that passes `validate_goal()`'s 5 cm threshold.

### 2.3 "not_moved" Ratio is Stuck > 85%

| Run A | not-moved / total | Run B | not-moved / total |
|---|---|---|---|
| 3230 | 121 / 629 = 19% | 1 | 1150 / 1330 = 86% |
| 3231 | 147 / 618 = 24% | 17 | 1308 / 1388 = 94% |
| 3232 | 180 / 605 = 30% | 230 | 1245 / 1358 = 92% |

The vast majority of Alice-phase completions produce **no measurable object
displacement at all**. Alice has converged to a policy that avoids the object.

### 2.4 High Termination Rate (robot_through_table + objects_off_table)

| Run A | robot_through_table | objects_off_table | Total |
|---|---|---|---|
| 3230 | 255 | 108 | 363/512 = **71%** |
| 3232 | 223 | 72 | 295/512 = **58%** |

| Run B | robot_through_table | objects_off_table |
|---|---|---|
| 1 | ~167 | ~25 |
| 50 | ~45 | ~50 |
| 230 | — | **63** (growing) |

In Run A, 71% of envs terminate mid-phase. In Run B, `objects_off_table` is
growing — Alice has learned to displace objects, but too aggressively, sending
them off the table.

---

## 3. Root Cause #1: Alice's Reward is Disconnected from Bob's Outcome

### What the code does (`goal_validator.py:122-138`)

```python
rewards = torch.where(
    off_table,      -3.0,
    torch.where(
        not_moved,   0.0,
        torch.where(
            unstable,     0.0,
            torch.where(
                out_of_zone,  -3.0,
                +1.0,                          # <— valid goal = +1 ALWAYS
            ),
        ),
    ),
)
```

Alice's reward is **purely geometric**: does the object stay on the table and
move more than 5 cm? This reward is identical whether Bob succeeds OR fails.

### How the reward is paid out (`wrapper.py:455,494`)

```python
# When Bob's phase ends (success or failure):
extras["alice_total_reward"][bob_done_ids] = self.delayed_alice_reward[bob_done_ids]
```

`delayed_alice_reward` was set during Alice's phase completion via
`validate_goal()`. Bob's outcome is **never consulted**.

### What the paper requires

In Plappert et al. 2021, the adversarial dynamic is:

```
Alice proposes a goal → Bob attempts it →
  Bob succeeds → Alice gets NEGATIVE reward (goal was too easy)
  Bob fails   → Alice gets POSITIVE reward (she found a blind spot)
```

Alice's reward must be **Bob's negative success signal** so that she is
incentivised to find goals at the edge of Bob's capability. Without this,
Alice has no reason to produce challenging goals.

### Consequence

Alice learns: *"Barely move the object ≥ 5 cm without knocking it off the table."*
This produces **trivially easy goals** that Bob should be able to solve — but
Bob can't even solve those because his own reward is sparse (see §5). The
curriculum collapses: Alice produces easy-but-Bob-can't-solve goals, Bob never
succeeds, no adversarial pressure exists, nothing improves.

---

## 4. Root Cause #2: ABC warm=NO Bug (Run B specific)

### The gating code (`ppo_abc.py:115-117`)

```python
effective_abc_coef = self.abc_coef
if alice_mean_rew < self.abc_warmup_threshold:   # 0.0
    effective_abc_coef = 0.0
```

`abc_warmup_threshold = 0.0` (Fix 9: "always active"). But when Alice receives
negative rewards (-3.0 for off-table goals), `last_alice_mean_rew` goes
**negative**:

```python
# train_curobo.py — perform_bob_update()
loss_val, loss_surr, loss_abc, _ = bob_ppo.update(
    alice_mean_rew=last_alice_mean_rew   # <— CAN BE NEGATIVE
)
```

Check: `alice_mean_rew < 0.0` → `-1.5 < 0.0` → **True** → `effective_abc_coef = 0`.

### Verification in logs

Run B logs `ABC warm: NO` on iterations 28–41 and beyond, even with `buf=500`.
The `_abc_warm` logging scalar (`train_curobo.py:1305`) uses the same condition:

```python
_abc_warm = 1.0 if ema_alice_rew >= bob_ppo.abc_warmup_threshold else 0.0
```

When `ema_alice_rew` drops below 0.0 (from accumulating -3.0 penalty rewards),
both the logging shows NO and the actual ABC gate disables imitation learning.

### Consequence

On iterations with ABC warm=NO:
- ABC loss = 0.0000 → Bob learns from PPO only (useless with SR=0)
- Bob's value function grows slowly (0.008 → 0.035) but no policy improvement
- Alice receives no benefit from demonstrations being fed to Bob
- The adversarial loop is **doubly** broken

**Fix**: Change the comparison from `alice_mean_rew < 0.0` to `alice_mean_rew <= -1e6`
or remove the gate entirely (since Fix 9 already intended ABC to run unconditionally):

```python
# ppo_abc.py:116 — REPLACE
if alice_mean_rew < self.abc_warmup_threshold:
# WITH
if False:  # ABC always active per Fix 9
```

---

## 5. Root Cause #3: Sparse Reward for Contact Manipulation

Alice receives **zero per-step reward** (`wrapper.py:966`):

```python
if is_alice.any():
    rewards[is_alice] = 0.0
```

She only gets the `validate_goal()` outcome at phase end (100 steps later). For
a multi-step contact-manipulation task (approach → touch → push), PPO has
almost no credit-assignment signal. The 86–94% "not-moved" ratio shows Alice has
converged to a policy that largely **avoids the object** — a classic sparse-reward
exploration failure.

Note: Fix 3 intentionally removed per-step shaping. The ASP paper uses
zero per-step reward, but the paper's task (block rearrangement with simpler
physics) had more natural exploration gradients than this cuRobo + Isaac Sim
contact task.

---

## 6. Root Cause #4: ABC Buffer Starvation (too_short trajectories)

### The gate (`train_curobo.py:1184`)

```python
just_failed_bob = bob_dones_now & (~bob_success) & goal_valid
```

### The length check (`train_curobo.py:1186,1202`)

```python
min_demo_steps = max(10, alice_timesteps // 2)  # = 50
...
if t_len < min_demo_steps:
    _abc_dbg_too_short += 1
    continue
```

### The data (Run A, iter 3231)

```
[A] bob_done=1 failed=1 goal_valid=1 gate=1 min_steps=50
    too_short=1 accepted=0 buf_size=500          ← all single-env completions too_short
[A] bob_done=90 failed=90 goal_valid=90 gate=90 min_steps=50
    too_short=0 accepted=90 buf_size=500          ← only batch completions get accepted
```

The high termination rate (71% per iter) means Alice phases are frequently
truncated by `robot_through_table` or `objects_off_table`. These short
trajectories don't meet the 50-step minimum, so they're rejected from the ABC
buffer. In Run A, the buffer is stale (500 entries from previous training). In
Run B, new trajectories trickle in slowly.

---

## 7. What Would Fail in the Diagnostic Suite

### Test 2 (`test_alice_sandbox.py`) — **FAILS**

| Check | Result |
|---|---|
| ValidGoals trend (late_avg > early_avg) | ❌ FAIL — Run B: late ≈ 120, early ≈ 189; ValidGoals declined |
| GoalValidityRate no NaN | ✔ PASS |
| EntropyCoef = 0.05 | ✔ PASS (0.05 fixed) |
| InvalidGoals > 0 | ✔ PASS |

### Test A (proposed, not implemented) — **FAILS**

Asserting `MeanDisp3D.mean() > 0.04 m` would fail. The mean is 0.026–0.035 m
across Run B.

---

## 8. Priority Fixes — Ordered by Impact

### P0: Wire Alice's reward to Bob's outcome

```python
# wrapper.py — _handle_alice_completion (line 577)
# CURRENT:
self.delayed_alice_reward[env_ids] = val_reward  # from validate_goal()

# FIX:
# Remove. Instead, in the Bob-completion handler (line 455, 494):
if bob_succeeded:
    self.delayed_alice_reward[env_ids] = -1.0   # Alice's goal was too easy
else:
    self.delayed_alice_reward[env_ids] = +1.0   # Alice found a blind spot
```

This restores the **adversarial dynamic** that drives the entire curriculum.
Without this, nothing else matters.

### P0: Fix ABC warm=NO — remove the negative-mean-reward shutdown

```python
# ppo_abc.py:116 — REPLACE both lines with:
effective_abc_coef = self.abc_coef  # always active, per Fix 9
```

This prevents the imitation learning signal from being silently disabled
whenever Alice accumulates -3.0 penalties.

### P1: Diagnostic — add small per-step shaping for Alice (temporary)

```python
# wrapper.py:966 — temporarily revert Fix 3 to confirm Alice can learn contact
if is_alice.any():
    # Diagnostic: small shaping for any object displacement
    curr_pos = obs_dict["target_object"][:, :3]  # or from object_state
    prev_pos = self._prev_alice_obj_pos
    delta_3d = torch.norm(curr_pos - prev_pos, dim=-1)
    rewards[is_alice] += 0.01 * (delta_3d > 0.01).float()  # tiny bonus for any movement
    self._prev_alice_obj_pos = curr_pos.clone()
```

Run with 50–200 iterations. If ValidGoals and MeanDisp3D rise sharply, the
problem is confirmed as sparse-reward exploration, not a learning capacity issue.
Remove after diagnosis.

### P1: Investigate objects_off_table spike

Options:
- Add a workspace-relative velocity penalty for the object
- Clamp EE target velocity when in contact zone
- Increase object mass / friction to make it harder to knock off
- Widen table bounds

### P2: Run `--alice_sandbox` to isolate Alice from Bob's failure

```bash
python train_curobo.py --num_envs 32 --max_iterations 200 \
  --exp_name "diag_alice_sandbox" --alice_sandbox
```

This confirms whether Alice's curriculum problem is kinematic (she can't learn
contact manipulation at all) or adversarial (Bob's failure drags her down).

### P3: Add Test A to diagnostic suite

```python
# diagnostics/test_alice_sandbox.py — check 6
disp3d = df[df["tag"] == "Metrics/Alice/MeanDisp3D"]["value"]
assert disp3d[-10:].mean() > 0.04, \
    f"MeanDisp3D floor violated: {disp3d[-10:].mean():.3f}m (need > 0.04m)"
```

---

## 9. Architecture Summary

```
                ┌──────────────────────────────┐
                │   Alice (PPO, 21D obs)        │
                │   100 steps, 0 per-step reward │  ← BROKEN: no intermediate gradient
                │                              │
                │   Outcome: validate_goal()    │  ← BROKEN: reward = f(geometry)
                │   +1 valid, -3 off-table,     │     not f(Bob outcome)
                │   0 no-movement               │
                └──────────┬───────────────────┘
                           │ goal state (pos+euler)
                           ▼
                ┌──────────────────────────────┐
                │   Bob (PPOABC, 29D obs)       │
                │   200 steps                    │
                │                              │
                │   Reward: sparse {+1/-1/+5}   │  ← PROBLEM: no shaping in
                │   ABC: β=0.5 imitation loss   │     contact task → SR ≈ 0
                │                              │
                │   GoalEncoder: 6D→8D latent   │
                └──────────┬───────────────────┘
                           │ bob_success / bob_fail
                           ▼
                ┌──────────────────────────────┐
                │   Alice's delayed reward       │  ← BROKEN: always +1 for valid
                │   paid at Bob-phase end        │     goal regardless of Bob outcome
                └──────────────────────────────┘

        Desired adversarial loop (paper):
        Alice reward = +1 if Bob fails, -1 if Bob succeeds
        → Alice seeks goals at Bob's capability boundary
        → Bob improves → Alice must find harder goals
        → Curriculum emerges automatically

        Current loop (broken):
        Alice reward = +1 if goal is geometrically valid
        → Alice learns minimal movement
        → Bob's sparse reward prevents any success
        → No feedback loop → dead curriculum
```

---

## 10. Key Files to Modify

| File | Lines | Change |
|---|---|---|
| `ppo_abc.py` | 115–117 | Remove `alice_mean_rew < 0.0` gate |
| `wrapper.py` | 455, 494, 577 | Route Bob outcome into Alice's delayed reward |
| `goal_validator.py` | 122–138 | Alice reward should come from Bob, not geometry |
| `wrapper.py` | 965–966 | (Temporary diagnostic) Small per-step shaping |
| `diagnostics/test_alice_sandbox.py` | after 113 | Add Test A: MeanDisp3D floor assertion |
