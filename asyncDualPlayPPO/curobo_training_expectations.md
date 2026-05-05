# cuRobo Training — Expected Behaviour and Failure Modes

> Written against `train_curobo.py` as implemented: 4D action space (XYZ + gripper),
> `max_delta = 0.025 m/step`, `_JC_ALPHA = 0.2` joint EMA, cuRobo IK seeded from
> `_prev_joint_cmd`, `JointPositionActionCfg(scale=1.0, use_default_offset=False)`.
> Hyperparameters from `cfg/ppo/ppo_continuous.yaml` and `cfg/task/AsyncDualPlay.yaml`.

---

## 1. Phase-by-Phase Learning Trajectory

### Phase 1 — Iterations 0–50: Noise Floor

**What to expect**

- Alice outputs near-uniform MultiCategorical (high entropy ≈ 1.0).  
  Every bin is equally likely → EE target does a random walk within the workspace clamp.
- Most Alice phases produce invalid goals (object not moved ≥ 0.05 m XY) because the
  random walk has low probability of consistently pushing the block in one direction.
- Bob sees no valid goal → no Bob episodes start → `SR = 0.00` throughout.
- Dense reward for Alice is small and noisy (potential shaping only rewards net
  displacement, not random oscillation).
- `IK_fail` rate may be non-zero early if the random EE target drifts outside cuRobo's
  reachable set despite the workspace clamp. Expect **< 5%** once the clamp is proven
  effective; higher means the clamp bounds need adjusting.
- ABC buffer stays cold (`ABC warm: NO`) because Alice mean reward < 1.0.

**Key metrics to watch**

| Metric | Expected value |
|---|---|
| `[AliceDisp] avg XY` | < 0.05 m (mostly not moving block) |
| `Goals valid / total` | 0 – 5 / 16 envs per iteration |
| `Bob SR` | 0.00 |
| `Metrics/IKFailRate` | < 0.05 |

---

### Phase 2 — Iterations 50–250: Alice Learns to Move Objects

**What to expect**

- Alice's policy gradient starts picking up signal from the dense potential reward
  (object displacement) + sparse +1 for valid goal.
- `[AliceDisp] avg XY` climbs from < 0.05 m toward 0.10–0.20 m.
- Valid goal rate rises to 30–60%.
- Bob starts receiving valid goals. Early Bob behaviour is random → SR stays low
  (0.05–0.15).
- Entropy coef still in exponential decay (1.0 → 0.10 by iter 250). Alice explores
  broadly; her goals are easy because she is still near-random.
- The action space reduction (4D vs 6D) means Alice cannot vary orientation.
  Object rotation goals will never appear — all valid goals are purely translational.
  This is intentional (Option B) but limits goal diversity.
- EMA joint smoothing (`_JC_ALPHA=0.2`) means the arm responds slowly. At 100 Alice
  steps × 0.025 m/step the maximum reachable displacement is 2.5 m accumulated delta
  — far more than the workspace allows — so the workspace clamp is the real limit.
  Effective coverage ≈ ±0.65 × 0.55 m table area.

**Key metrics to watch**

| Metric | Expected value |
|---|---|
| `[AliceDisp] avg XY` | 0.08–0.20 m |
| `Goals valid / total` | 5–10 / 16 |
| `Bob SR` | 0.05–0.15 |
| `Alice EntropyCoef` | 1.0 → 0.10 (decaying) |
| `ABC warm` | NO (Alice EMA reward still < 1.0) |

---

### Phase 3 — Iterations 250–1000: Adaptive Curriculum Engages

**What to expect**

- Alice entropy controller switches to Phase 2 (proportional controller targeting
  Bob SR = 0.50). Entropy rises if Bob SR > 0.5, falls if SR < 0.5.
- If Bob SR is tracking correctly, expect oscillation around SR = 0.50 with amplitude
  ± 0.10 over ~50 iterations.
- ABC buffer warms when Alice EMA reward ≥ 1.0. Once warm, Bob's loss gets an
  imitation term (`ABC: 0.5 → 0.0` over 3000 iters). This typically accelerates Bob's
  early learning by 2–4×.
- GoalEncoder embedding starts carrying meaningful signal (watch
  `GoalEncoder/embedding_std` — should increase from ~0.01 toward ~0.3 as the encoder
  learns to distinguish goals).
- Alice displacement plateaus: she cannot make goals harder than the workspace allows
  (max XY ≈ 0.60 m from start). Difficulty is gated by how far she can move the block
  in 100 steps × 0.025 m.

**Key metrics to watch**

| Metric | Expected value |
|---|---|
| `Bob SR` | 0.40–0.60 (tracked by controller) |
| `Alice EntropyCoef` | oscillates 0.10–0.40 |
| `ABC warm` | YES |
| `GoalEncoder/embedding_std` | > 0.10 |
| `IK_fail` | < 0.02 (policy learned to stay in workspace) |

---

### Phase 4 — Iterations 1000–10000: Fine-Tuning

**What to expect**

- Bob SR stabilises around 0.50 (curriculum target). Alice is setting goals at the edge
  of Bob's competence.
- ABC annealing completes at iter 3000 (`abc_coef → 0.0`). Bob switches to pure RL.
  There may be a short SR dip (5–10%) as the imitation scaffolding is removed.
- Max goal displacement approaches the practical ceiling:
  block can be pushed to ~0.35–0.55 m XY depending on table layout and robot reach.
- `not-moved` count in `[AliceDisp]` should approach zero (Alice reliably contacts
  and moves the block).

---

## 2. Expected Problems and Mitigations

### 2.1 IK Failure Cascade (Early Training)

**Symptom**: `IK_fail` > 0.20 in first 50 iterations; many `[ALICE END] ik_fails=N`
with N > 10; episode resets visible in the log.

**Root cause**: Random policy sends EE target near workspace boundary; cuRobo cannot
find a solution → episode reset → arm teleports → next episode starts with a
potentially unreachable ee_target (wrist_3 might be at an awkward position after reset).

**Mitigation already in place**: workspace clamp `_WS_X/Y/Z`.

**What to do if it persists**: Tighten the clamp by 0.05 m on each edge, or initialise
`ee_target_local` to a known-good central position (e.g., `[0.0, 0.45, 0.35]` in local
frame) on every reset instead of from current wrist position.

---

### 2.2 Alice Never Produces Valid Goals

**Symptom**: `Goals valid / total = 0` beyond iter 100; `[AliceDisp] avg XY` stuck < 0.05 m;
dense reward remains near zero.

**Root causes**:

1. **EE target not reaching the block**: the workspace clamp or max_delta is too
   conservative. With `max_delta = 0.025 m` and `_JC_ALPHA = 0.2`, the arm moves
   at most `0.025 × 0.2 = 0.005 m/step` toward a new target — effectively very slow.
   100 steps × 0.005 m = 0.5 m maximum actual arm travel if target is constant, but
   random targets make net displacement << 0.5 m.

2. **Block too far from EE home position**: if the robot's home pose parks the EE away
   from the block spawn zone, early random walks never contact the block.

3. **Dense reward not propagating**: check that `alice_rew_buf` is non-empty after
   iter 10. If `Reward/Alice` is exactly 0.00 every iteration, the backfill logic has
   a bug.

**Mitigation**: Run with `--num_envs 4 --debug_rewards` and watch `[ALICE END]`
lines. `dense_acc` should be non-zero for envs where the arm made contact. If it is
always 0.00, the potential-shaping reward is not being stored.

---

### 2.3 Bob SR Stuck at 0.00 Beyond Iter 200

**Symptom**: Valid goals appear in logs but Bob never solves any; `Bob SR = 0.00`
persists past iter 200.

**Root causes**:

1. **GoalEncoder not converging**: if `GoalEncoder/embedding_norm` is NaN or > 100,
   the encoder has exploded. Add gradient clipping to GoalEncoder if not already
   present.

2. **Goal too hard too early**: Alice is setting goals at maximum displacement
   (0.5 m+) before Bob has learned to move at all. Check `[AliceDisp] avg XY` — if
   it is > 0.30 m before iter 100, Alice is overperforming relative to Bob.

3. **Bob obs not including goal correctly**: verify `Bob Obs Dim = 29` is printed at
   startup. If it is lower, the goal encoding is missing from Bob's observation.

---

### 2.4 Joint EMA Prevents Alice from Moving Block

**Symptom**: Arm moves visibly slowly; `[AliceDisp] avg XY` never exceeds 0.03 m even
after 200 iterations; IK succeeds but arm barely reaches block.

**Root cause**: `_JC_ALPHA = 0.2` means 80% of the previous joint command persists
each step. Combined with `max_delta = 0.025 m`, the effective EE velocity is low. If
the block is 0.30 m from the arm's home EE position, it takes
`0.30 / (0.025 × mean_bin_magnitude)` steps to reach it — potentially 60–200+ steps
out of 100 allowed.

**Fix**: Raise `_JC_ALPHA` to 0.35 or raise `max_delta` back to 0.05 m. The correct
trade-off depends on whether joint oscillation returns. Monitor with `--debug_rewards`.

---

### 2.5 ABC Buffer Never Warms

**Symptom**: `ABC warm: NO` persists beyond iter 300; `Metrics/Alice/EMAReward` stays
below 1.0.

**Root cause**: Alice never earns a valid goal reward (+1.0 sparse) because of any of
the above. The EMA of 0.0 rewards never crosses the warmup threshold.

**Fix**: Temporarily lower `abc_warmup_threshold` to 0.3 in the YAML to force ABC
activation and check if Bob's learning improves with imitation. If it does, Alice's
reward signal is the bottleneck.

---

### 2.6 Entropy Controller Drives Alice Entropy to Maximum

**Symptom**: `Alice EntropyCoef` ratchets up to 1.0 and stays there; Bob SR drops below
0.20 and does not recover.

**Root cause**: Bob SR < 0.50 (controller target) → controller raises Alice entropy →
Alice explores more → goals become harder → Bob SR drops further. Positive feedback loop.

**This is the main training instability risk for cuRobo.** It does not occur in
RMPFlow training because RMPFlow's implicit trajectory smoothing keeps goal quality
higher, giving Bob an easier early curriculum.

**Fix**: Lower `alice_entropy_target_sr` to 0.30 in the YAML for the first cuRobo run.
This gives Bob more slack and prevents the runaway feedback loop.

---

### 2.7 cuRobo Config Mismatch (URDF vs YAML)

**Symptom**: `IK_fail` is 1.00 on every step from the start; arm never moves;
`[ALICE END] ik_fails = 100` for every env.

**Root cause**: The `ur5e.yml` used by cuRobo does not match the URDF loaded by Isaac
Lab (different joint limits, link names, or base offset).

**Diagnosis**: Run `test_curobo_follow_target.py` and check the 10-step error printout.
If `err > 0.05 m` consistently, the config is mismatched.

**Fix**: Compare joint limits in cuRobo's `ur5e.yml` vs the URDF in `urdf/`. If the
Isaac Lab URDF has been modified (e.g., joint limit changes for safety), update the
cuRobo YAML to match.

---

### 2.8 HPC Container Missing cuRobo

**Symptom**: Job fails immediately with `[ERROR] cuRobo not found`.

**Diagnosis**:
```bash
apptainer exec --nv isaac-lab.sif python -c "import curobo; print(curobo.__version__)"
```

**Fix**: Either rebuild the Apptainer image with cuRobo installed, or bind-mount the
local cuRobo wheel:
```bash
--bind /path/to/curobo_wheel:/curobo_wheel
```
and pip-install it inside the container entrypoint.

---

## 3. Comparison vs RMPFlow Baseline

| Metric at Iter 500 | RMPFlow (expected) | cuRobo (expected) |
|---|---|---|
| Bob SR | 0.45–0.55 | 0.30–0.50 |
| Alice valid goal rate | 50–70% | 40–65% |
| Alice avg XY disp | 0.15–0.30 m | 0.10–0.25 m |
| IK fail rate | N/A | < 0.03 |
| Arm motion quality | Smooth (RMPFlow reactive) | Smooth if `_JC_ALPHA` tuned correctly |
| Steps/sec (512 envs, A100) | baseline | −5% to −10% (IK overhead) |

cuRobo's main advantage over RMPFlow appears **after** iter 1000: singularity-free
joint trajectories mean Alice can explore configurations that RMPFlow would refuse
(near-singular elbow positions), giving Bob a richer goal distribution. This benefit
is only observable at high iteration counts (> 5000).

---

## 4. Diagnostic Checklist (First Run)

Run with `--num_envs 16 --max_iterations 50 --debug_rewards`:

- [ ] Startup prints `max_delta=2.5 cm` and `[cuRobo] CUDA graph warm-up done`
- [ ] `[ALICE END]` lines appear within first 5 iterations
- [ ] `dense_acc` is non-zero for at least some envs (reward signal flowing)
- [ ] `ik_fails` < 10 per 100-step phase (< 10% failure rate)
- [ ] `[BOB END]` lines appear within first 20 iterations (valid goals are forming)
- [ ] `Metrics/IKFailRate` < 0.05 by iter 10
- [ ] `GoalEncoder/embedding_std` > 0.01 by iter 30 (encoder not collapsed)
- [ ] `Alice EntropyCoef` decaying from 1.0 (entropy decay active)

If all boxes pass, the run is healthy. Submit the full production job with
`sbatch hpc/train_curobo.slurm`.
