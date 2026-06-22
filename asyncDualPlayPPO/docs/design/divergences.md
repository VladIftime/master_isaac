# Implementation vs Paper Divergences

Papers: `asymetric-self-play.pdf` (OpenAI 2021) + `asp-summ.md` (theoretical analysis)

---

## Correct — Matches Paper

- [x] MultiCategorical 6D × 11 bins (`use_multicategorical=True` in module.py)
- [x] `max_goals_per_episode=5`
- [x] `alice_timesteps=100`, no STOP action (Table 3)
- [x] HistoricalPolicyPool 20% past policy versions
- [x] PPO clip ε=0.2, ABC clip ε=0.2 (Table 2)
- [x] LSTM actor + PermInvEncoder (max-pool, 512-dim)
- [x] ABC demos gated to Bob-failure trajectories only
- [x] β=0.5 in config (before SR-coupling overrides it)

---

## Critical

### 1. SR-Coupled `abc_coef`
- **File:** `train_curobo.py:1237`
- **Paper:** β=0.5 constant (Table 2)
- **Implementation:** `_abc_coef_start * (1 − bob_sr)` with EMA 0.95 after anneal period
- **Impact:** asp-summ.md — "Highly destructive; injects parasitic second-order feedback loops, destroys trust region." Bob's high-variance gradient contaminates Alice's update via the coupling term.
- **Fix applied:** `train_curobo.py:1237-1253` — removed SR-coupling and anneal schedule; replaced with `bob_ppo.abc_coef = ppo_cfg["params"]["learn"].get("abc_coef", 0.5)` (fixed β per Table 2)

### 2. SR-Coupled Alice Entropy
- **File:** `train_curobo.py:1211`
- **Paper:** `entropy_coef=0.01` fixed (Table 2)
- **Implementation:** Phase 2 (iter≥250) uses PI controller coupled to `SR_B`
- **Impact:** asp-summ.md — "Vicious feedback loops, premature mode collapse." When Bob struggles, entropy is raised; this randomises Alice, producing easier goals, reducing Bob's challenge, collapsing the curriculum.
- **Fix applied:** `train_curobo.py:1211-1227` — removed PI controller and decay schedule; `entropy_coef` is now read-only from config (default 0.01). Logging kept.

### 3. Dense Potential Shaping for Alice
- **File:** `wrapper.py:1009`
- **Paper:** Alice gets **zero** per-step reward — only outcome {+5/+1/−3/0} at phase end
- **Implementation:** `Φ(s) = 3.0·(1−exp(−5·dist))` applied every Alice step, rewarding object displacement
- **Impact:** Alice optimises for moving objects far rather than finding Bob's blind spots. She accumulates shaping reward even on eventually-invalid goals, decoupling her incentive from the adversarial objective.
- **Fix applied:** `wrapper.py:992-1015` — removed entire shaping block and base_rewards passthrough; Alice step now sets `rewards[is_alice] = 0.0` (covers Fix 11 simultaneously)

### 4. Dense Potential Shaping for Bob
- **File:** `wrapper.py:1106`
- **Paper:** Sparse {+1/−1/+5} only
- **Implementation:** `F = γ·Φ(s') − Φ(s)`, Φ(s) = −Σ pos\_dist, every Bob step
- **Impact:** Bob receives a dense gradient toward the goal regardless of goal difficulty. Undermines the autocurriculum — Bob can learn without Alice finding his genuine blind spots.
- **Fix applied:** `wrapper.py:1092-1129,1196` — removed shaping block (phi_curr, phi_prev, shaping tensors, prev_pos_dists updates); changed final reward line to `rewards = step_rewards + completion_bonus`

### 5. ABC as Separate Backward Pass
- **File:** `ppo_abc.py:285`
- **Paper:** `L = L_PPO + β·L_ABC` as a single combined loss per mini-batch, every epoch
- **Implementation:** All `num_epochs × num_batches` PPO steps complete first, then one ABC backward pass
- **Impact:** (a) PPO and ABC gradients never co-mingle in the same optimizer step; (b) the policy shifts by `n_epochs×n_batches` steps before ABC corrects it; (c) ABC evaluates a different policy than collected the demos.
- **Fix applied:** `ppo_abc.py:126-293` — `bc_loss` now computed once per epoch before the mini-batch loop; added to the LAST mini-batch's PPO loss → single `optimizer.step()` with `L = L_PPO + β·L_ABC` per epoch. Separate ABC backward pass removed entirely.

---

## High

### 6. GoalEncoder Architecture — Not in Paper
- **Files:** `module.py:421`, `goal_encoder.py`
- **Paper (Figure 12):** Goal state processed by the **same** PI encoder as current object state, then summed: `PI_enc(obj_state) + PI_enc(goal_state) → ReLU → MLP → LSTM`
- **Implementation:** Separate GoalEncoder φ MLP maps `(goal_pose, current_pose) → K=8` latent; injected additively after actor layer 1: `h = act(LN(W₁·enc + W_g·g))` — "Charlie" paper architecture
- **Impact:** Different information pathway and different weights. Paper reuses the same encoder for goals and current state; implementation has a dedicated lightweight encoder. Goal representation is fundamentally different.
- **Fix applied:** NOT fixed — GoalEncoder architecture is a deliberate design choice ("Charlie" paper extension). Fixing would require adding a second PermInvEncoder for goal states and removing GoalEncoder entirely; deferred.

### 7. Sum-Pool Instead of Max-Pool for Goal Embedding
- **File:** `module.py:431`
- **Paper:** Max-pool over objects throughout
- **Implementation:** `g_pooled = g_per_obj.sum(dim=1)` — sum-pool after `encode_per_object()`
- **Impact:** Inconsistent with both the paper's PI encoder and the GoalEncoder's own internal max-pool (`goal_encoder.py:187`). Sum-pool doesn't saturate on outlier objects the way max-pool does.
- **Fix applied:** `module.py:431` — changed to `g_pooled = g_per_obj.max(dim=1)[0]`

---

## Medium

### 8. EMA Joint Smoothing
- **File:** `train_curobo.py:877`
- **Paper:** Direct TCP servoing, no filtering
- **Implementation:** `_JC_ALPHA=0.2` → `smoothed = 0.2·raw_IK + 0.8·prev_cmd`
- **Impact:** Arm position at step t is dominated by IK solutions from steps t−2 through t−5. Policy output and actual EE motion are decorrelated in time (~5 step lag).
- **Fix applied:** `train_curobo.py:707` — changed `_JC_ALPHA = 0.2` to `_JC_ALPHA = 1.0` (no smoothing; raw IK solution used directly)

### 9. `abc_warmup_threshold` Gate
- **File:** `ppo_abc.py:115`
- **Paper:** β=0.5 from iteration 1
- **Implementation:** ABC held at 0 until `alice_mean_rew ≥ abc_warmup_threshold`
- **Impact:** Delays ABC engagement. May help early stability but deviates from the paper's constant coupling.
- **Fix applied:** `ppo_abc.py:62` — set `self.abc_warmup_threshold = 0.0` unconditionally (always active); config key still read but overridden

### 10. Min XY Displacement Filter (7cm)
- **File:** `wrapper.py:581`
- **Paper:** Requires only 3D displacement > Bob's success threshold (~4cm); no XY-specific filter
- **Implementation:** Alice's goal rejected if no object moves >7cm in XY
- **Impact:** Filters out rotation-only goals and short-range displacement goals that the paper explicitly includes. Constrains Alice's curriculum space; removes valid goals the paper would accept.
- **Fix applied:** `wrapper.py:576-597` — removed entire XY displacement filter block; goal validity now determined solely by `validate_goal()` (position + rotation thresholds in 3D)

### 11. Alice Receives Physics Penalties from Base Rewards
- **File:** `wrapper.py:993`
- **Paper:** Alice gets zero per-step reward during her phase
- **Implementation:** `rewards[is_alice] = base_rewards[is_alice]` — collision/OOB penalties from RewardManager pass through
- **Impact:** Combined with dense shaping (#3), Alice's per-step signal is complex and not outcome-only as the paper specifies.
- **Fix applied:** Covered by Fix 3 — Alice block now sets `rewards[is_alice] = 0.0` unconditionally

### 12. `num_cat_dims` Default is 4, Not 6
- **File:** `module.py:186`
- **Paper:** 6D × 11 bins (XYZ + Rx/Ry + gripper) — Appendix B.2
- **Implementation:** `model_cfg.get("num_cat_dims", 4)` — default 4 dims; `bins_to_delta` assumes 6 (slices `:3`, `3:5`, `5:6`)
- **Impact:** If training config doesn't override to 6, `bins_to_delta` silently produces wrong-shaped outputs (rot indices misaligned, gripper index empty). Config footgun.
- **Fix applied:** `module.py:186` — changed default from 4 to 6: `model_cfg.get("num_cat_dims", 6)`

---

## Low

### 13. Aux Loss Head on GoalEncoder
- **File:** `goal_encoder.py:254`
- **Paper:** No auxiliary loss; object states fed directly
- **Implementation:** GoalEncoder predicts (pos\_dist, rot\_dist) as supervised auxiliary signal
- **Impact:** Provides extra supervision outside the PPO/ABC loop. Probably helpful in practice but not in the paper.
- **Fix applied:** NOT fixed — aux loss is a useful training signal; kept as-is

### 14. GoalEncoder Frozen During ABC
- **File:** `module.py:424`
- **Paper:** No separate goal encoder to freeze
- **Implementation:** `detach_goal_encoder=True` during ABC forward — GoalEncoder receives no gradient from ABC loss
- **Impact:** ABC cannot adjust how goals are represented. Minor by design choice.
- **Fix applied:** `ppo_abc.py:98` — changed `detach_goal_encoder=True` to `detach_goal_encoder=False`; GoalEncoder now receives ABC gradients

### 15. `ppo.py:log()` Crashes with MultiCategorical
- **File:** `ppo.py:226`
- **Paper:** N/A
- **Implementation:** `self.actor_critic.log_std.exp().mean()` — `log_std` not created when `use_multicategorical=True` → `AttributeError` at logging time
- **Impact:** Base class logging is broken for MC mode. ppo_abc.py doesn't call `log()`, so it's silent in production, but will crash if ppo.py is used directly.
- **Fix applied:** `ppo.py:226-233` — guarded with `if hasattr(self.actor_critic, "log_std"):` so MC mode skips the std logging line

### 16. KL Adaptive LR is Dead Code in MC Mode
- **File:** `ppo_abc.py:167`
- **Paper:** Fixed lr=3×10⁻⁴ (Table 2)
- **Implementation:** KL formula uses `sigma_batch`; in MC mode `sigma=zeros` always → KL=0 always → adaptive LR never fires. `desired_kl=None` by default anyway.
- **Impact:** Dead code path; no runtime effect, but misleading.
- **Fix applied:** `ppo_abc.py:173` — added `and not self.actor_critic.use_multicategorical` guard to the KL block condition
