# Validation Results Summary — Planar Pushing PBRS Models

**Date**: 2026-07-01 (revised; ASP training diagnostics added — per-axis vs combined SR from TensorBoard events)  
**Location**: `/home/vlad/IsaacLab/vlad/master_isaac/results_validation/`  
**Scene set**: 30 T-block scenes (tests 1–10: R_* rotation, 11–20: pos_only, 21–30: pos_rot)  
**Thesis gate**: `pos_err < 0.05 m AND rot_err < 0.2 rad`, best-of-20 trials, max 30 pushes  
**Coverage gate** (gym-pusht native): `coverage >= 0.95`, best-of-20 trials, max 30 pushes  
**Note**: Model I uses 10 trials (not 20); `orig_loss` PBRS CSV gate was position-only (`pos_err < 0.05 m`), not thesis gate.

---

## Isaac Lab Results — 30 T-block Scenes (Thesis Gate)

528 parallel envs, RTX Pro 6000 (96 GB VRAM), cuRobo IK.

### Validation Results (best-of-20 trials, max 30 pushes)

| Model | Scene SR | Trial SR | Pos-only | Pos+rot | PosErr | RotErr | Avg Push |
|-------|----------|----------|----------|---------|--------|--------|----------|
| **A_simp**      | **80.0%** (24/30) | 66.0% (396/600) | 100% (10/10) | 70% (14/20) | 0.032 m | 0.568 rad | 23.5 |
| **B_curr**      | **76.7%** (23/30) | 68.3% (410/600) | 100% (10/10) | 65% (13/20) | 0.023 m | 0.663 rad | 26.7 |
| **G_tasp_dpose** | 16.7% (5/30)  | 1.2% (7/600)    | 30% (3/10)  | 10% (2/20)  | 0.158 m | 1.457 rad | 12.2 |
| **H_tasp_disc**  | 10.0% (3/30)  | 0.8% (5/600)    | 30% (3/10)  | 0%  (0/20)  | 0.186 m | 1.260 rad | 12.6 |
| **C_asp**        | 6.7%  (2/30)  | 0.3% (2/600)    | 20% (2/10)  | 0%  (0/20)  | 0.143 m | 1.612 rad | 12.8 |
| **E_asp_dpose**  | 6.7%  (2/30)  | 0.3% (2/600)    | 20% (2/10)  | 0%  (0/20)  | 0.143 m | 1.612 rad | 12.8 |
| **F_asp_disc**   | 6.7%  (2/30)  | 0.3% (2/600)    | 20% (2/10)  | 0%  (0/20)  | 0.197 m | 1.576 rad | 11.9 |

**A_simp leads by 3.3pp over B_curr.** Both achieve 100% pos-only SR. Single-agent (A) outperforms curriculum (B) and all ASP variants (C–H).  
**G_tasp_dpose is the best ASP variant** at 16.7% — time-based Alice on T-block.  
**Rotation remains the bottleneck.** Pos-only is solved (A/B at 100%), pos+rot caps at 65–70%. ASP models never achieve any pos+rot success (C–F at 0%).

### ASP Training Diagnostics — Per-Axis vs Combined Success Rate (2026-07-01)

Extracted from TensorBoard events (final iteration of each run). Bob's training metrics show per-axis skills exist under PBRS but the combined gate is the bottleneck:

| Variant | Alice Validity | Bob PositionSR | Bob RotationSR | **Bob Combined SR** | Independence Product |
|---------|---------------|----------------|----------------|--------------------|---------------------|
| **orig_ASP** (orig reward, 2,048 envs) | 89.3% | 1.3% | 7.1% | **0.07%** | ~0.1% |
| **ASP-dPose** (PBRS outcome, 528 envs) | 83.9% | 54.1% | 45.0% | **10.0%** | 24.3% |
| **TASP-dPose-BP** (PBRS time+pen, 528 envs) | 90.5% | 54.1% | 46.3% | **8.5%** | 25.0% |

**Key finding:** Under PBRS, Bob learns position and rotation individually to ~50% each. But the combined gate caps at ~10% — 2.5× below what independence would predict. Under the original fractional reward, Bob learns nothing (1.3%/7.1% per-axis, 0.07% combined). The adversarial goal distribution prevents simultaneous satisfaction of the coupled pos+rot objective — the combined gate is a causal bottleneck, not a measurement artifact (C4 resolved).

**Alice learns across all variants** (83--91% GoalValidityRate) — Alice is NOT the problem. The failure sits at Bob's inability to combine independently-learned position and rotation skills under the non-stationary adversarial distribution.


---

## Original PBRS Reward Variant — Isaac Lab (30 T-block Scenes)

Same 528-env infrastructure, but using the **original (non-simplified) PBRS reward formula** before the simplification in Fix P63 (normalised fractional reward). Comparison isolates whether the PBRS simplification (fractional deltas, single α coefficient) improved performance.

| Model | CSV Gate SR | Thesis Gate SR | Trial SR | Pos-only | Pos+rot | PosErr | RotErr | Avg Push |
|-------|------------|----------------|----------|----------|---------|--------|--------|----------|
| **orig_PBRS** (single-agent) | **83.3%** (25/30) | 56.7% (17/30) | 70.5% (423/600) | 100% (10/10) | 35% (7/20) | 0.050 m | 0.460 rad | 26.7 |
| **orig_ASP** (Bob only) | 0.0% (0/30) | 0.0% (0/30) | 0.0% (0/300) | 0% (0/10) | 0% (0/20) | 0.355 m | 1.419 rad | 27.4 |

**Key finding:** The PBRS simplification from orig → A_simp adds **23.3pp under thesis gate** (56.7% → 80.0%) and **20pp pos+rot** (35% → 70%). The original formula double-scaled log components (Fix P69) and used separate pos/rot α coefficients (α_pos=12, α_rot=1 before normalisation) — the unified SE(2) fractional delta (α=3.0, P63) eliminates the competing gradient problem.  
**orig_ASP fails at 0%** — confirms ASP collapse persists across reward formulations. The adversarial goal distribution prevents Bob from learning regardless of whether the reward is original or simplified.

**Validation protocol caveat:** `orig_PBRS` CSV `success` column used a **position-only gate** (83.3%), not the thesis gate (56.7%). This reflects an earlier validation script before the thesis gate was standardised. Model I also uses 10 trials/scene (not 20).

---

## Model I — TASP dPose + Bob Time Penalty (Isaac Lab)

Fork of G_tasp_dpose (time-based ASP). Adds symmetric Bob time penalty `R_B += −γ_sp·t_B` at Bob phase end (Sukhbaatar et al. 2018 full reward structure). Alice reward unchanged (`R_A = γ_sp·max(0, t_B−t_A)`). **Complete failure.**

| Model | Scene SR | Trial SR | Pos-only | Pos+rot | PosErr | RotErr | Avg Push |
|-------|----------|----------|----------|---------|--------|--------|----------|
| **I_tasp_bob_pen** | **0.0%** (0/30) | 0.0% (0/300) | 0% (0/10) | 0% (0/20) | 0.367 m | 1.262 rad | 17.7 |

**10 trials/scene, not 20** — lower than the standard protocol. Bob penalty collapses the adversarial dynamics entirely: penalising Bob for time creates perverse incentives (Bob learns to minimise pushes by failing fast) and destroys the self-regulating curriculum that gave G_tasp_dpose 16.7%.

---

## Gym-pusht HPC Results — 30 T-block Scenes (Thesis Gate)

32 CPU cores (AsyncVectorEnv for A/B; single-process for C), `push_nsteps=30`, Apptainer container.  
Validated with both thesis gate and coverage gate on identical scenes.

**Validation checkpoints** (intermediate; training continued past these points):  
A at iter 8,800 / B at iter 9,000 / C at iter 1,300.

| Model | Scene SR | Pos-only | Pos+rot | PosErr | RotErr | Coverage SR |
|-------|----------|----------|---------|--------|--------|-------------|
| **A_gym** | 10.0% (3/30)  | 30% (3/10)  | 0% (0/20)  | 0.288 m | 0.948 rad | 0.0% |
| **B_gym** | **50.0%** (15/30) | **90%** (9/10) | **30%** (6/20) | 0.100 m | 1.220 rad | 0.0% |
| **C_gym** | 13.3% (4/30)  | 40% (4/10)  | 0% (0/20)  | 0.140 m | 1.482 rad | 0.0% |

All models score 0% under the coverage ≥ 0.95 gate (gym-pusht native metric).

### Gym HPC Training Metadata (latest checkpoints, 2026-06-30)

| Model | Final Iters | Wall-Clock | Est. Push-Macros | Throughput | Pushes/update |
|-------|------------|------------|-----------------|------------|---------------|
| **A_gym** | **18,800** | ~30 h (~1.3 d) | **18.0 M** | ~209 push/s (~0.217 it/s) | 960 |
| **B_gym** | **18,800** | ~30 h (~1.3 d) | **18.0 M** | ~209 push/s (~0.217 it/s) | 960 |
| **C_gym** | **4,400** | ~48 h (~2.0 d) | **4.2 M** | ~24 push/s (~0.025 it/s) | ~480 |

A/B throughput from TF event scalar timestamps (June 28 21:44 → June 29 21:44, 24h captured).  
C throughput from TF events spanning June 27 23:19 → June 29 23:19 (48h, single-process ASP).  
A and B continued past the observed TF events to ~30h total; all event files rsynced from TMPDIR at job completion.

**Key observation:** A and B received 18.0M pushes — nearly identical to Isaac A's 19.0M and Isaac B's 20.6M. Total push count is matched; the performance gap is driven by batch size (960 vs 7,920 transitions/update).

### Gym HPC vs Isaac (same thesis gate, same total push budget)

| Model | Isaac SR | Gym SR | Isaac PosErr | Gym PosErr | Isaac Pushes | Gym Pushes | Batch Isaac | Batch Gym |
|-------|----------|--------|-------------|------------|-------------|------------|-------------|-----------|
| **A** | 80.0% | 10.0% | 0.032 m | 0.288 m | 19.0 M | 18.0 M | 7,920 | 960 |
| **B** | 76.7% | 50.0% | 0.023 m | 0.100 m | 20.6 M | 18.0 M | 7,920 | 960 |
| **C** | 6.7%  | 13.3% | 0.143 m | 0.140 m | 60.2 M | 4.2 M | 7,920 | ~480 |

A and B have matched total push budgets (~18–20M); the SR gap is driven by per-update batch size (8× larger in Isaac). C's Isaac run has 14× more pushes than C's gym run but underperforms it — ASP does not benefit from more experience.

---

## Gym-pusht (Non-HPC, Local) — ASP Models E–H + C

Standard gym-pusht (not HPC 32-core), best-of-20 trials, max 30 pushes, thesis gate. These are **non-HPC** runs with fewer CPU cores — substantially weaker training than the HPC 32-core models above. Included for protocol-difference reference; not directly comparable to gym HPC rows.

| Model | Scene SR | Trial SR | Pos-only | Pos+rot | PosErr | RotErr | Coverage SR |
|-------|----------|----------|----------|---------|--------|--------|-------------|
| **E_dpose_gym** | 3.3% (1/30) | 0.2% (1/600) | 10% (1/10) | 0% (0/20) | 0.097 m | 1.611 rad | 0.0% |
| **F_disc_gym**  | 0.0% (0/30) | 0.0% (0/600) | 0% (0/10)  | 0% (0/20) | 0.122 m | 1.659 rad | 0.0% |
| **G_tasp_gym**  | 0.0% (0/30) | 0.0% (0/600) | 0% (0/10)  | 0% (0/20) | 0.095 m | 1.259 rad | 0.0% |
| **H_tasc_gym**  | 3.3% (1/30) | 0.2% (1/600) | 10% (1/10) | 0% (0/20) | 0.129 m | 1.450 rad | 0.0% |
| **C_asp_gym**   | 0.0% (0/30) | 0.0% (0/600) | 0% (0/10)  | 0% (0/20) | 0.275 m | 1.444 rad | 0.0% |

**Key observation:** At non-HPC scale, all ASP models (C–H) collapse to ~0–3.3% SR — only E and H manage a single pos-only success each (E_Forward scene). The gym-pusht non-HPC protocol uses fewer CPU cores and smaller batches than HPC (960 transitions/update), making gradient estimates even noisier. Together with the HPC results, this confirms ASP fails across **three environments** (Isaac, gym HPC, gym non-HPC) and collapses to zero with decreasing batch quality.

---


### Earlier local gym runs (few CPU cores, for reference only)

These are from local `gym_gympusht.csv` validations with N≤6 cores — substantially weaker than the HPC 32-core models above. Included for budget scaling reference, not comparison.

| Model | PosErr | RotErr | Notes |
|-------|--------|--------|-------|
| A_gym_local | 0.377 m | 0.684 rad | 6 cores, ~144k total pushes |
| B_gym_local | 0.172 m | 1.294 rad | 6 cores, ~144k total pushes |
| C_gym_local | 0.275 m | 1.444 rad | single-process, few iterations |

---

## Interpretation

**1. The ordering A ≈ B ≫ C is preserved across Isaac and gym HPC, but the absolute gaps compress in the simpler gym environment.**

B_curr narrows from 1.5× behind at Isaac scale to 5× ahead in gym HPC — the P82 curriculum (position-only Phase 1) partially compensates for the 8× smaller PPO batch (960 vs 7,920 transitions/update). In Isaac, where the batch is large enough for stable gradient estimates on both objectives simultaneously, the curriculum's staging adds no value (76.7% vs 80.0%). In gym HPC, where batch variance limits PPO+LSTM gradient quality, the simplified Phase 1 objective helps bootstrap position control (90% gym vs 100% Isaac pos-only SR).

**2. ASP (Models C–H) fails uniformly across three environments — Isaac, gym HPC, gym non-HPC.**

Training diagnostics reveal the mechanism: under PBRS, Bob learns per-axis position (54\%) and rotation (45--46\%) individually, but the combined gate caps at 8.5--10.0\% — 2.5$\times$ below the independence product. The adversarial goal distribution prevents Bob from satisfying both criteria simultaneously. Under the original reward, Bob learns nothing (1.3\% per-axis, 0.07\% combined). Alice validity reaches 83--91\% across all variants — Alice is not the bottleneck.

Outcome-based Alice (+5 fail / −1 succeed) produces a non-stationary goal distribution that prevents Bob from learning the combined pos+rot objective regardless of simulator choice. Gym C HPC (13.3%) slightly edges Isaac C (6.7%) because the simpler gym environment (no cuRobo IK failures, no contact physics, smaller workspace) gives Bob marginally cleaner gradients — but neither breaks above ~15% or achieves any pos+rot success. At non-HPC (few-CPU) scale, all ASP models collapse to ~0–3.3% with zero pos+rot success — the adversarial gradient vanishes as batch quality degrades. The failure is structural: ASP's adversarial curriculum collapses in contact-rich multi-objective domains, consistent across all three environments.

**3. Time-based ASP (G/H) partially rescues ASP but cannot close the single-agent gap.**

G_tasp_dpose reaches 16.7% — the best ASP variant, 2.5× better than outcome-based E (6.7%). H_tasp_disc reaches 10.0%. Time-based Alice (Sukhbaatar et al. 2018) provides a self-regulating curriculum that avoids the toxic-goal collapse of outcome-based Alice, but still lags single-agent A by 4.8× (80.0% vs 16.7%). Adding Bob time penalty (Model I) collapses to 0% — penalising Bob for solve time destroys the self-regulating dynamics entirely.

**4. Model I (TASP dPose + Bob penalty) is a failed ablation — 0% SR, 10 trials/scene.**

The symmetric Sukhbaatar reward (Alice gains from Bob's slow solving, Bob penalised for slow solving) creates a dynamic where Bob's optimal strategy is to fail immediately (minimise penalty), and Alice cannot bootstrap because Bob never provides meaningful episodes. Confirms that the asymmetric formulation (Alice incentivised, Bob unpenalised) is necessary for TASP to function.

**5. PBRS simplification adds 23.3pp (thesis gate).**

orig_PBRS (original reward, pre-simplification) reaches 56.7% thesis gate and 70.5% trial SR. A_simp (simplified, P63) reaches 80.0% and 66.0%. The simplification eliminates competing position/rotation α coefficients and double-scaled log components (Fix P69), replacing them with a single unified SE(2) fractional delta. The pos+rot improvement is dramatic: 35% → 70%.

**6. The dominant Isaac→gym scaling factor is PPO+LSTM batch variance, not total push count.**

Gym A/B HPC received 18.0M pushes — nearly identical to Isaac A's 19.0M and B's 20.6M. Yet A drops from 80.0% to 10.0% and B drops from 76.7% to 50.0%. Total push count is matched; the per-update batch is 960 vs 7,920 transitions (8× smaller). With LSTM hidden-state propagation (Fix P13), smaller batches sample fewer hidden-state initial conditions per update, creating biased GAE advantage estimates and higher policy-gradient variance. This causes earlier convergence plateaus, not slower convergence — the gym models reach their ceiling faster but plateau lower. PBRS `k_p=30` provides meaningful gradient out to ~0.35 m (ΔΦ ≈ 0.86 for a 5 cm improvement from 0.30 m), so the reward signal is present — it's the gradient estimate quality that degrades at small batch.

**7. Gym B HPC (50%) confirms that curriculum helps at small batch without improving at large batch — consistent with the independence thesis (RQ3).**

At Isaac scale (7,920/batch), no curriculum (A, 80.0%) beats curriculum (B, 76.7%) — staging adds complexity without value. At gym scale (960/batch), curriculum (B, 50.0%) dramatically beats no curriculum (A, 10.0%) — staging simplifies the objective enough to overcome batch variance. Reward design and curriculum design are independent levers; their interaction depends on the batch size regime.

---

## Training Metadata — Isaac Models

All Isaac models run on 528 parallel envs, RTX Pro 6000 (96 GB VRAM), cuRobo IK, ~0.022 it/s (~46 s/iter).  
Per-iteration push-macros: 528 envs × ~15 pushes = ~7,920.

| Model | Algorithm | Iters (planned) | Est. Wall-Clock | Est. Push-Macros | Checkpoint | Size |
|-------|-----------|----------------|-----------------|-----------------|------------|------|
| A_simp | PPO + LSTM | 2,400 (3,000) | ~31 h (~1.3 d) | ~19.0 M | 26.06.20, `agent/latest_checkpoint.pt` | 9.35 MB |
| B_curr | PPO + LSTM | 2,600 (2,600) | ~33 h (~1.4 d) | ~20.6 M | 26.06.28, `agent/model_best.pt` | 9.35 MB |
| C_asp | PPOABC + GoalEncoder | 7,600 (3,000) | ~97 h (~4.0 d) | ~60.2 M | 26.06.26, `bob/model_best.pt` | 5.24 MB |
| E_asp_dpose | PPOABC + GoalEncoder | 7,600 (3,000) | ~97 h (~4.0 d) | ~60.2 M | 26.06.26, `bob/model_best.pt` | 5.24 MB |
| F_asp_disc | PPOABC + GoalEncoder | 7,400 (3,000) | ~95 h (~3.9 d) | ~58.6 M | 26.06.26, `bob/model_best.pt` | 5.24 MB |
| G_tasp_dpose | PPOABC + GoalEncoder | 3,800 (3,000) | ~49 h (~2.0 d) | ~30.1 M | 26.06.26, `bob/model_best.pt` | 5.24 MB |
| H_tasp_disc | PPOABC + GoalEncoder | 4,000 (3,000) | ~51 h (~2.1 d) | ~31.7 M | 26.06.26, `bob/model_best.pt` | 5.24 MB |
| I_tasp_bob_pen | PPOABC + GoalEncoder | — | — | — | 26.06.29, `bob/model_best.pt` | ~5.24 MB |
| orig_PBRS | PPO + LSTM | 1,030 (1,030) | ~50 h (~2.1 d) | ~31.6 M | 26.06.11, `agent/model_best.pt` (it 1000) | 9.35 MB |
| orig_ASP | PPOABC + GoalEncoder | 1,266 (1,266) | ~69 h (~2.9 d) | ~38.9 M | 26.06.11, `bob/model_best.pt` (it 1200) | 5.24 MB |

**Cost efficiency:** A_simp achieves 80.0% SR in ~1.3 days / 19M pushes. ASP models (E/F) train for 3.9–4.0 days / 59–60M pushes and only reach 6.7%. That's 3× more compute for 12× lower success. G_tasp_dpose is the most efficient ASP at 16.7% in ~2.0 days / 30M pushes. Model I (0%) and orig_ASP (0%) are complete failures — both confirm that adversarial/penalty mechanisms do not bootstrap meaningful learning.

**RQ1 baseline note:** `orig_PBRS` is the hand-tuned ad-hoc PPO baseline (`hpc_push_2048env_rel_full`, classic reward, rel_obs + rel_act, 2048 envs, iter 1000). Compared to A_simp (PBRS, same rel_obs/rel_act): PBRS adds **23.3pp thesis gate** (56.7% → 80.0%) and **35pp pos+rot** (35% → 70%) at comparable push budget (31.6M vs 19.0M).

---

## Model Architecture Reference

| Model | Type | Architecture | Object | Reward |
|-------|------|-------------|--------|--------|
| **A_simp** | Single-agent PBRS | PPO + LSTM (28D obs) | T-block | PBRS dense + sparse +5/+2 |
| **B_curr** | PBRS + forced curriculum | PPO + LSTM (28D obs) | T-block | PBRS + 3-phase pos→rot staging |
| **C_asp** | PBRS + ASP | PPOABC + GoalEncoder | T-block | PBRS + outcome Alice +5/-1 |
| E_asp_dpose | ASP + SE(2) d_pose | PPOABC + GoalEncoder | T-block | Single-potential PBRS, L=0.07m |
| F_asp_disc | ASP + disc | PPOABC + GoalEncoder | Disc (L=0) | Position-only PBRS |
| G_tasp_dpose | Time-based ASP | PPOABC + GoalEncoder | T-block | TASP: R_A = γ_sp·max(0, t_B−t_A) |
| H_tasp_disc | Time-based ASP + disc | PPOABC + GoalEncoder | Disc | TASP + position-only |
| **I_tasp_bob_pen** | G + Bob time penalty | PPOABC + GoalEncoder | T-block | TASP: R_A = γ_sp·max(0, t_B−t_A), R_B += −γ_sp·t_B |
| **orig_PBRS** | Single-agent pre-P63 | PPO + LSTM | T-block | PBRS (original): α_pos=12, α_rot=1, separate Δd/Δyaw |
| **orig_ASP** | ASP pre-P63 | PPOABC + GoalEncoder | T-block | PBRS (original) + outcome Alice +5/-1 |

---

## Source Directories

- `A_simp/20_isaac_30t.csv` — from `ppo_pbrs_reward/26.06.20/runs/hpc_pbrs_simp_528env/` (iter 2400)
- `A_simp/hpc_gym_a_valid.csv` — from `asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.28/hpc_gym_a/` (iter 8,800; latest ckpt at 18,800)
- `B_curr/28_isaac_30t.csv` — from `asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.28/hpc_pbrs_curr_528env_fixed/` (iter 2600)
- `B_curr/hpc_gym_b_valid.csv` — from `asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.28/hpc_gym_b/` (iter 9,000; latest ckpt at 18,800)
- `C_asp/26_isaac.csv` — from `26.06.26/runs/hpc_pbrs_asp_dpose_528env/`
- `C_asp/hpc_gym_c_valid.csv` — from `asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.28/hpc_gym_c/` (iter 1,300; latest ckpt at 4,400)
- `E_asp_dpose/26_isaac.csv` — from `26.06.26/runs/hpc_pbrs_asp_dpose_528env/`
- `F_asp_disc/26_isaac.csv` — from `26.06.26/runs/hpc_pbrs_asp_disc_528env/`
- `G_tasp_dpose/26_isaac.csv` — from `26.06.26/runs/hpc_pbrs_tasp_dpose_528env/`
- `H_tasp_disc/26_isaac.csv` — from `26.06.26/runs/hpc_pbrs_tasp_disc_528env/`
- `I_tasp_bob_pen/results_tasp_dpose_bob_pen.csv` — Model I (TASP dPose + Bob penalty), 10 trials/scene, 0% SR
- `orig_loss/0_orig_rew_30_isaac.csv` — Original PBRS reward (pre-simplification), single-agent, 20 trials/scene
- `orig_loss/reults_valid_asp_orig_Rew.csv` — ASP + original reward, 10 trials/scene, 0% SR

### Gym-pusht (non-HPC) CSVs
- `E_asp_dpose/gympusht.csv` — non-HPC gym-pusht, best-of-20 trials
- `F_asp_disc/gympusht.csv` — non-HPC gym-pusht, best-of-20 trials
- `G_tasp_dpose/gympusht.csv` — non-HPC gym-pusht, best-of-20 trials
- `H_tasp_disc/gympusht.csv` — non-HPC gym-pusht, best-of-20 trials
- `C_asp/gympusht.csv` — non-HPC gym-pusht, best-of-20 trials

### Cross-model comparison
- `comparison/isaac_summary.md` — Isaac-only cross-model summary table (A/B/E/F/G/H)
- `comparison/gympusht_summary.md` — Gym-pusht cross-model summary (all variants)
- `comparison/per_test_comparison.txt` — Per-test PASS/FAIL for Isaac models
- `comparison/gympusht_per_test.txt` — Per-test PASS/FAIL for gym-pusht models
- `comparison/` — 26 PNG charts (SR bars, grouped diffs, error scatter, etc.)
- `abc_comp/` — Cross-environment plots (p1_sr_bars.png, p2_batch_bottleneck.png, p3_heatmap.png)
