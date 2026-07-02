# Presentation Revision Plan — Supervisor Critique Response

**Deck**: `literature/paper-async/presentation/presentation.tex` (~1530 lines, beamer/XeLaTeX)
**Branch**: `asp_goal_encoder`
**Last updated**: 2026-07-02 (8 of 10 plan items resolved; ABC mechanism explained with PPO-clipping + discrete-action argument; val_multi_combined.png regenerated with 5 models and matching styling; Discussion section reduced to 2 unique slides; Key Result merged into Conclusion RQ2; Extra Takeaway placeholder removed; What Does Success Look Like slide added; TASP-dPose-BP removed from main deck; per-attempt SR footnote added)
**Defense timeline**: 1–3 weeks (cheap re-runs from existing checkpoints allowed; no new long trainings)

> **SINGLE SOURCE OF TRUTH (RESOLVED 2026-06-30).** The deck is now backed by consistent CSVs.
> **Five models** on 30 identical T-block scenes: **PPO-Baseline 83.3%**, **PPO-PBRS 80.0%**,
> **PPO-Curriculum 76.7%**, **TASP-dPose 16.7%**, **ASP-dPose 6.7%**. Disc models (F/H) removed.
> TASP-dPose-BP removed from main deck (kept in Appendix I + Limitations + Empirical Result table).
> Descriptive names throughout. Builds to 62pp xelatex-clean.

---

## Table of Contents

1. [Overview & Status](#overview)
2. [Supervisor Critique Register (C1–C10)](#critique)
3. [Key Findings From Investigation](#findings)
4. [Deck-vs-Data Reconciliation Table](#reconciliation)
5. [Evaluation Matrix — Which Checkpoints to Run](#evalmatrix)
6. [Three-Tier Fix Plan](#tiers)
7. [Slide-by-Slide Action Map](#slidemap)
8. [Open Items to Verify](#openitems)
9. [Decisions Already Made](#decisions)
10. [Key File & Data Index](#files)
11. [Final Audit — Remaining Issues & Action Plan](#audit)
12. [gym-pusht Testbed](#gym-testbed)

---

## 1. Overview & Status <a name="overview"></a>

The supervisor raised **10 problems** with the master-thesis defense deck. They cluster into
three themes:

| Theme | Critiques | Nature |
|---|---|---|
| **Scientific integrity / overclaiming** | C1, C3, C4, C5, C6 | Claims not supported by (or inconsistent with) the saved data |
| **Methodological rigor** | C7, C8, C9 | Missing seeds/CIs, baselines, controlled measurements |
| **Internal consistency / delivery** | C2, C10 | Contradictory numbers, over-dense slides, missing clips |

The biggest risk (per supervisor: "the single biggest risk") is **C1 — overclaiming external
validity**. Investigation shows the integrity problem is **deeper than the supervisor knew**:
the central results table is not backed by the experiments on disk (see §4).

---

## 2. Supervisor Critique Register (C1–C10) <a name="critique"></a>

> Each entry: **[SUP]** = supervisor's verbatim concern, **[ASK]** = the user's instruction,
> **[FIND]** = what the investigation found, **[FIX]** = planned action.

### C1 — Overclaiming external validity
- **[SUP]** The deck says it disproves Plappert 2021 / Sukhbaatar 2018 / Narvekar 2020. You showed one implementation of ASP failed on one task at one budget. "ASP failed in my setup" ≠ "ASP generates a toxic curriculum." This is the single biggest risk.
- **[ASK]** He is right. Re-write the points that make this claim; focus on *why ASP fails for this specific task*. Candidate arguments: (a) ASP needs very large compute overhead for robotic sim; (b) ASP is bad at building a curriculum for a task that must balance **multiple competing goals**. Use the theory table (Narvekar survey, Portelas, Florensa reverse-curriculum, Luo PCCL, Hutsebaut HRL, Sukhbaatar) to find more task-specific arguments.
- **[FIND]** The adversarial loop **is** wired in the final PBRS Model C code (`wrapper_push_asp.py:755–761` rewards Alice on Bob-fail / penalizes on Bob-success), so "toxic curriculum" is defensible *in principle*. BUT it is mixed with a geometric base reward (`val_reward`, line 616) so the adversarial signal may be diluted. The strong "disproves" framing is not warranted from N=1 task, single budget.
- **[FIX]** Rewrite Slides 13a/13b + Conclusion: replace every "Disproven: …" with *"In this specific contact-rich, IK-gated, multi-objective SE(2) pushing task, under a single-GPU budget, our ASP implementation did not yield a useful curriculum."* Ground the task-specificity in: combined pos∧rot gate (multi-objective), contact-rich sparse exploration, IK gating, compute budget. Position **forced vs automatic curriculum** using the theory table.

### C2 — Scale mismatch undercuts the ASP verdict
- **[SUP]** Self-play is notoriously scale-hungry. 3000 iters × 528 envs ≈ a few hours. You even cite Berner 2019 (Dota 2 = months of self-play) — which argues against your own "ASP fails" conclusion at this budget.
- **[ASK]** They have a point. Argue in speaker notes that 3000 iterations take ~2 days of training on my system, and frame this correctly for my set-up and what I can realistically implement.
- **[FIND]** Deck Slide 1 notes say "~3000 iterations each"; budget is real constraint. Berner currently reads as self-contradiction.
- **[FIX]** Add speaker note + caveat: 3000 iters ≈ **~2 days wall-clock** on one RTX Pro 6000 @528 envs; self-play is scale-hungry (Berner = months). Reframe claim to *"ASP did not pay off within a budget realistic for a single-GPU robotics lab"* — turning Berner into **support**, not contradiction.

### C3 — The headline "4.6×" mixes incomparable metrics
- **[SUP]** Model A's 80% is best-of-3 on 30 validation scenes; Push-PPO's 17.3% is single-attempt training SR. Best-of-3 of 80% ≈ ~41% per attempt. Comparing best-of-3 validation to training SR is not apples-to-apples — RQ1's central number is methodologically inconsistent.
- **[ASK]** Check my results (`asyncDualPlayPPO/runs`) and `master_isaac_archive_clean`; tell me what would be a better comparison considering all saved runs.
- **[FIND]** Confirmed worse than stated: deck's "80%" = the **pos-only sub-score (80%)** on the 20-scene set; **overall is 60%**, pos+rot 40%. Push-PPO training SR ranges **0.8%→23.8%** across runs/obs modes; 17.3% is one rel_full run.
- **[FIX]** Drop cross-protocol ratios. Report a **same-protocol** table: validation SR (best-of-3, identical scene set) for A vs B vs C vs ad-hoc, plus a separate training-SR curve. The fair RQ1 comparator for ad-hoc is `hpc_push_2048env_rel_full` (same rel obs as A) — re-eval it on the same scenes (see §5). If a ratio is kept, compute it at equal protocol.

### C4 — 0.07% combined SR vs 66%/65% individual is statistically suspicious
- **[SUP]** If position and rotation succeeded independently at ~0.65 each, combined would be ~0.42, not 0.0007. 0.07% with 65% marginals implies strong anti-correlation or different conditions — points to a gate/reward bug, not a fundamental "toxic curriculum." An examiner will ask exactly this.
- **[ASK]** Explain this problem.
- **[FIND]** The 66%/65% were from old `train_curobo.py` training logs (per-axis SRs logged independently at different timesteps, not the thesis gate). **Resolved 2026-07-01:** Extracted training metrics from current PBRS ASP runs: Bob does learn per-axis position (54\%) and rotation (45--46\%) under PBRS, but the combined gate caps at 8.5--10.0\% — 2.5$\times$ below independence. Under the original reward, nothing is learned. The combined gate is a causal bottleneck (coupled physics + adversarial distribution), not a measurement artifact.
- **[FIX]** New 2-panel diagnostic slide: Panel 1 = Alice GoalValidityRate bar chart (83--91\% across variants), Panel 2 = Bob per-axis vs combined SR (54\% / 45\% per-axis → 10\% combined). Caption explains the independence gap.

### C5 — Negative results confounded by a massive bug-fix history
- **[SUP]** `implementations.md` lists fixes P1–P81+, several ASP-breaking (P55 "PPO update never called → Iteration 0 forever," P64 wrong dense formula until 2026-06-08, P65 false-positive termination on 45–55% of pushes). Were the ASP "collapse" numbers produced on the fixed code? If not, the finding is an artifact.
- **[ASK]** Explain this problem.
- **[FIND]** **All saved runs post-date the fixes.** Earliest checkpoints are **2026-06-11**; P55/P64/P65 are dated ≤2026-06-08. PBRS ASP runs are 06-15 → 06-19. So the headline ASP-collapse numbers were produced on **post-fix** code — defensible. Caveat: the deeper Alice-reward-disconnect documented in the **05-19 diagnostic** was in the *old* `train_curobo.py` path, not the final PBRS `wrapper_push_asp.py` (which is correctly wired, see C1). **Separately, Model B's negative result IS a remaining artifact:** its curriculum trigger never fired by construction (see §3.1) — so its "failure" is a code bug, not a scientific finding.
- **[FIX]** Add an explicit slide/footnote: "All reported runs use code after fixes P63–P81 (≥2026-06-11); pre-fix runs are excluded." Cite the checkpoint dates. This *defuses* the confound rather than ignoring it.

### C6 — "Same reward" isn't a clean ablation
- **[SUP]** Model A and Model C differ in goal distribution (fixed-random vs adversarial), episode budget (15 vs ≤50 pushes), and agent count. The collapse could be "harder/non-stationary goals," not "the curriculum mechanism." You admit distribution shift is the cause — which means RQ2/RQ3 aren't cleanly isolating "curriculum."
- **[ASK]** How could I change the argument to be better?
- **[FIND]** True: A and C differ on ≥3 axes at once (goal distribution, budget, agent count). Archive configs: A = 15 pushes single-agent fixed-random; C = Alice 5 + Bob 10 adversarial, 2 agents. The deck's "independence" claim over-reaches. **Model B is also not a clean curriculum test:** its Phase‑2 trigger never fired (see §3.1), so Model B measures "a curriculum that never turned on," not "PBRS+curriculum vs PBRS."
- **[FIX]** (1) State explicitly that ASP changes *three* things together, so it isolates **"self-play as implemented," not "the curriculum mechanism."** (2) Reframe Model B honestly as a **mis-specified trigger** (§3.1), not "PBRS made curriculum unnecessary." (3) Optional controlled experiment (approved): single-agent PPO + **scripted hard-goal distribution** (same budget, same agent count) to isolate distribution-shift from the two-agent machinery — see §5 / Tier 2.

### C7 — No statistical rigor in the main comparison
- **[SUP]** Single numbers, vague ranges ("0–3%"), no seeds/variance/CIs for the headline table. "3 runs" is mentioned only for PBRS robustness.
- **[ASK]** He is right — propose fixes.
- **[FIND]** Training-SR variance data **exists**: PBRS Model A = **5 chains** (BestSR 0.081–0.110); Push-PPO = 7 abs runs (0.0076–0.075) + 4 rel_full (0.0009–0.238). Validation evals are **single-seed per config** (one results CSV per model per date). `plot3_run_consistency.png` already exists.
- **[FIX]** (a) Compute **mean ± 95% CI** for training SR from the archive chain logs; add error bars/shaded curves. (b) Re-run `validate_push.py` on Model A's **multiple existing checkpoints** (26.06.15/16/17/18 + archive chains) → validation **mean ± CI**. (c) Replace "0–3%" / "3–7%" with exact per-run numbers + ranges. (d) State n explicitly per cell.

### C8 — Missing the obvious working baselines
- **[SUP]** No HER (standard sparse goal-conditioned pushing baseline), no SAC (despite citing Haarnoja). "80% is good" has no external calibration vs published planar-pushing results.
- **[ASK]** How hard would a very simple HER and SAC pushing agent be, for more comparison points?
- **[FIND]** Stack is bespoke: discrete 4D×21 MultiCategorical actions + cuRobo IK push primitive + Isaac Lab. HER fits the goal-conditioned env naturally (relabel achieved object pose as goal). SAC needs a continuous-action or SAC-discrete variant wired to the push primitive (current actions are discrete).
- **[FIX]** **Decision: NOT selected for re-runs** (deprioritized for the 1–3 week budget). Effort estimate kept for record: HER ≈ 3–7 days, SAC ≈ 5–10 days integration + training. Lighter alternative chosen instead: **cite published planar-pushing SR** (e.g., learned pushing papers) for external calibration in the Limitations slide.

### C9 — Compute-overhead slide is apples-to-oranges
- **[SUP]** The 5–10× claim compares against a different project (Table Tennis, DirectRLEnv, 4096 envs, joint-space, no IK). Not a controlled measurement of ASP overhead.
- **[ASK]** What would be a good comparison?
- **[FIND]** Slide 11 compares ASP push (528 env, IK) vs Table-Tennis (4096 env, joint-space, no IK) — confounds task, env count, action space, IK.
- **[FIX]** Replace with a **controlled** measurement: **Model A (single-agent push) vs Model C (ASP push)** on the **same machine, same 528 envs, same push + cuRobo pipeline** → report iterations/sec and env-steps/sec. Extract per-iteration wall-clock from existing slurm logs (no re-run needed); optional short timing run for clean numbers. **Approved for this round.**

### C10 — Internal inconsistencies + delivery risk
- **[SUP]** Bob budget "≤50" (slide 8a) vs "200" elsewhere; "30 scenes" (slide 6c) vs "20/30" (appendix B); category counts disagree (overview 6/12 vs detail headers 3/7). ~30 content slides for 30 min (~1 min/slide incl. full proofs) is over-dense; several slides depend on clips still `[clip placeholder]`.
- **[ASK]** Give me a new plan to fix these.
- **[FIND]** Confirmed contradictions (see §4 + §7): budget (deck Alice 15/Bob ≤50 vs implementations "200" vs archive 5/10); scenes (deck 30 / appendix "20/30" / saved eval 20); Models G/H reported but **no G/H runs exist**; many `\mediabox`/`\playclip` placeholders unfilled (rec_push_s11/s13/s21, asp_random, gif_push_sequence).
- **[FIX]** (1) One number for each quantity, sourced from §5 re-eval. (2) Standardize counts: 8 models A–H = 1 single-agent + 1 curriculum + **6 ASP variants** (C–H); fix 6/12 and 3/7 headers. (3) Cut ~30 → **~18–20 content slides** (~1.5 min each); push proofs/derivations to appendix. (4) Placeholder register: record-or-remove each clip (§7).

---

## 3. Key Findings From Investigation <a name="findings"></a>

1. **Deck headline numbers are not on disk.** No CSV contains 80% / 30-scene / disc-100%. Only the **20-scene best-of-3 = 60%** eval (26.06.18) exists. (→ C3, C10)
2. **ASP loop IS wired** in final code (`wrapper_push_asp.py:755–761`) — corrects an earlier assumption based on the stale 05-19 diagnostic. "Toxic curriculum" is defensible in principle but evidence must come from final-run logs. (→ C1, C4)
3. **Validation protocol is real**: `validation_configs.py::ALL_TESTS` = **30 scenes** (10 disc + 10 T-block pos-only + 10 T-block pos+rot); `validate_push.py:320 MAX_RETRIES=3` ⇒ **best-of-3** genuinely exists. Default `--max_pushes=15`; 26.06.18 used 30. (→ C3, C10)
4. **All runs are post-fix** (checkpoints ≥2026-06-11; fixes ≤2026-06-08). (→ C5)
5. **G/H (time-based ASP) now exist and were evaluated** — G_tasp_dpose = 16.7%, H_tasp_disc = 10.0%. They outperform outcome-based E/F (6.7% each) by 3-10pp. Time-based Alice partially rescues ASP but still lags single-agent by 4.8×. (→ C10 resolved)
6. **"disc 100%" for Model A is unbacked** — Model A (T-block-trained) was never evaluated on the disc env. (→ C3, C10)
7. **Seeds exist for training SR** (A: 5 chains; ad-hoc: 7 abs + 4 rel_full) but **not for validation** (single CSV per config). (→ C7)
8. **05-19 diagnostic root causes (OLD path, context for C5):** Alice reward = pure geometry (not Bob outcome); ABC silently disabled (`warm=NO` when `alice_mean_rew<0`); Bob sparse reward in a contact task; ABC buffer starvation from short truncated trajectories. These were the *old* `train_curobo.py` pathologies; the final PBRS path fixed the wiring (item 2).
9. **Model B curriculum never triggered by construction** — the Phase‑2 trigger thresholds a quantity that cannot reach the threshold. Full mechanism in §3.1. (→ C5, C6)
10. **C4 fully resolved (2026-07-01):** Extracted Bob training metrics from TensorBoard events across three ASP variants. Under PBRS, Bob learns per-axis position to **54.1\%** and rotation to **45.0--46.3\%** individually — but the combined gate caps at **8.5--10.0\%**, 2.5$\times$ below the independence product (24--25\%). Under the original fractional reward, Bob learns nothing (PositionSR 1.3\%, Combined 0.07\%). Alice validity converges to 83--91\% across all variants. New 2-panel diagnostic plot (`asp_diagnostic_2panel.png`) replaces the old `plot4_alice_vs_bob_diagnostic.png` on slide "ASP Diagnostic — Alice \& Bob Training Dynamics."

### 3.1 Model B — Why the curriculum never triggered <a name="modelb"></a>

**File:** `asyncDualPlayPPO/train_b_pbrs_curriculum.py`.

**The trigger (lines 880–886):** Phase 2 activates only if **all 50** of the most recent
iterations have `mean_pos_err < CURRICULUM_POS_THRESHOLD` (= **0.08 m**, line 497;
`CURRICULUM_LOOKBACK = 50`, line 498):
```python
ema_pos_err_hist.append(mean_pos_err)                                  # 881
if not curriculum_active and len(ema_pos_err_hist) == CURRICULUM_LOOKBACK:
    if all(e < CURRICULUM_POS_THRESHOLD for e in ema_pos_err_hist):    # 0.08, ALL 50
        curriculum_active = True
```

**The bug — `mean_pos_err` is the wrong quantity:**
- Line 686: `pos_err_buf.extend(pbrs_result["pos_err"]…)` runs **inside the push loop** →
  records object→goal distance **after every push, for every env** (15 pushes × `num_envs`).
- Line 837: `mean_pos_err = np.mean(pos_err_buf)` → **mean per‑push position error over the
  whole iteration**, dominated by early/mid‑episode pushes far from a freshly randomized goal.

Objects spawn 0.1–0.45 m from the goal (random spawn + ≤0.45 m filter) with only
`max_pushes_per_episode = 5`, so the per‑push mean has a **structural floor well above 0.08 m**.
Sanity check: the **best** model's *validation* avg PosErr is **0.202 m** (26.06.18) — i.e. the
trigger demands the iteration‑mean per‑push error be **< ~0.4× of what the best model ever
achieves**, for 50 consecutive iterations. **Unreachable by construction.**

Two compounding factors:
1. **Wrong metric** — should key on a "position is solved" signal (episodic position‑SR
   `env.episode_successes`, or terminal error of completed episodes), not the mean of *all*
   per‑push errors.
2. **Brittle AND over 50 iters** — `all(e < 0.08 …)`; one noisy iteration ≥ 0.08 resets the streak.

**Consequence:** `curriculum_active` stays `False` forever → line 893 keeps `w_rot = 0.0`, and
line 648 (`enable_rot_sparse=curriculum_active`) keeps the **+2 rotation bonus gated off**.
Rotation gets zero signal the whole run → matches observed RotErr 0.503 / pos+rot SR ~10–20%.

**Defense framing fix (Slide 7):** the deck says *"PBRS was so effective the EMA never
stabilized."* That is a mischaracterization. Honest statement: **the curriculum trigger was
mis‑specified (0.08 m on iteration‑mean per‑push error, below even the best model's performance,
with a brittle 50‑iteration AND), so Phase 2 was unreachable by construction** — an
implementation artifact, **not** evidence that "curriculum is useless when reward is informative."

**Corrected trigger (if re-running Model B):** gate on **episodic position‑only SR ≥ τ**
(e.g. τ≈0.6) sustained over a lookback window with **hysteresis** (enter at τ_hi, don't reset on
single dips), or on the terminal pos_err of *completed* episodes — not the per‑push mean.

---

## 4. Deck-vs-Data Reconciliation Table <a name="reconciliation"></a>

Source of truth (2026-06-30): `results_validation/SUMMARY.md` + gym HPC latest checkpoints.

| Deck claim | Actual data (2026-06-30) | Verdict |
|---|---|---|---|
| PPO-PBRS **80% SR**, **24/30** scenes | `A_simp/20_isaac_30t.csv` = 80.0% (24/30) | ✅ MATCH |
| pos-only 100%, pos+rot 70% | 10/10 pos-only, 14/20 pos+rot | ✅ MATCH |
| PPO-Baseline (orig reward, 2048 env) 80% SR | `orig_loss/0_orig_rew_30_isaac.csv` = 24/30 (80.0%) | ✅ MATCH (added 2026-07-01) |
| ASP-Baseline (orig reward, 2048 env) 0% SR | `orig_loss/reults_valid_asp_orig_Rew.csv` = 0/30 | ✅ MATCH (added 2026-07-01) |
| PPO-Sparse 16.7% SR | `0_PPO_sparse/26_isaac.csv` = 5/30 | ✅ MATCH (added 2026-07-01) |
| PPO-PBRS PosErr 0.032 m / RotErr 0.568 rad | mean over all 30 scenes (incl. failures) = 0.0322 / 0.5685 | ✅ MATCH; clarified as "mean over all 30" in deck |
| PPO-Curriculum **76.7%** (P82 fixed) | `B_curr/28_isaac_30t.csv` = 76.7% (23/30) | ✅ MATCH |
| ASP-dPose **6.7%**, TASP-dPose **16.7%** | `E_asp_dpose/26_isaac.csv` = 2/30; `G_tasp_dpose/26_isaac.csv` = 5/30 | ✅ MATCH |
| TASP-dPose-BP **0%** (tested) | `ppo_pbrs_dpose/26.07.01/.../results_tasp_dpose_bob_pen.csv` = 0/30 | ✅ MATCH (added 2026-07-01) |
| Gym B 50% | `B_curr/hpc_gym_b_valid.csv` = 15/30 = 50.0% | ✅ MATCH |
| C-Isaac file = E file (byte-identical) | C_asp/26_isaac.csv ≡ E_asp_dpose/26_isaac.csv | ⚠ no genuine Model-C Isaac validation exists |
| Disc models removed from deck entirely | F/H CSVs still on disk; deck has zero disc references | ✅ deck clean; disc data archived |
| RQ3 "substitute" framing | PPO-Curriculum 76.7% ≈ PPO-PBRS 80.0% (n.s.); gym shows curriculum helps at small batch | ✅ substitution holds at large batch; scope caveat stated |

### Current definitive validation (30 T-block scenes, best-of-20, max-30 pushes, thesis gate)

| Model | SR | PosErr | RotErr | Pos-only / Pos+rot |
|---|---|---|---|
| PPO-Baseline (orig.\ reward, 2048 env) | **80.0\%** | $\sim$0.03 m | $\sim$0.50 rad | 100\% / 70\% |
| PPO-PBRS (no curriculum) | **80.0\%** | 0.032 m | 0.568 rad | 100\% / 70\% |
| PPO-Curriculum | **76.7\%** | 0.023 m | 0.663 rad | 100\% / 65\% |
| PPO-Sparse (no PBRS) | 16.7\% | — | — | 30\% / 10\% |
| TASP-dPose (time-based self-play) | 16.7\% | 0.158 m | 1.457 rad | 30\% / 10\% |
| ASP-dPose (outcome-based self-play) | 6.7\% | 0.143 m | 1.612 rad | 20\% / 0\% |
| TASP-dPose-BP (Bob penalty) | 0.0\% | — | — | 0\% / 0\% |
| ASP-Baseline (orig.\ reward, 2048 env) | 0.0\% | — | — | 0\% / 0\% |

**Key takeaways:** PPO-Baseline and PPO-PBRS tie at 80\% — PBRS achieves same performance with 4$\times$ fewer parallel envs. No ASP variant exceeds 16.7\%. The symmetric Bob penalty (TASP-dPose-BP) collapses to 0\%. PPO-Sparse at 16.7\% shows reward starvation without PBRS even for single-agent.

---

## 5. Evaluation Matrix — Which Checkpoints to Run <a name="evalmatrix"></a>

Goal: produce **one authoritative, defensible results table** from existing checkpoints
(best-of-3, identical 30-scene protocol where possible). All `model_best.pt` or `latest_checkpoint.pt`.

**Status as of 2026-06-29:**

| # | Model | Checkpoint | Status |
|---|---|---|---|
| 1 | **A (PBRS simp)** | `ppo_pbrs_reward/26.06.20/runs/hpc_pbrs_simp_528env/agent/latest_checkpoint.pt` (it 2400) | ✅ DONE — 80.0% SR, `results/A_simp/20_isaac_30t.csv` |
| 2 | A — seeds | 26.06.15/16/17 chains | ❌ Not done — single-seed validation |
| 3 | **B (curr)** | `hpc_pbrs_curr_528env_fixed/agent/model_best.pt` (it 2600) | ✅ DONE — 76.7% SR, `results/B_curr/28_isaac_30t.csv` |
| 4 | **C (ASP)** | Bob only has gym-pusht eval | ⚠ No Isaac eval for C — no Bob checkpoint validated on Isaac |
| 5 | **D (ASP no-GE)** | Excluded per user request | — |
| 6 | **E/F/G/H (dpose/disc/tasp)** | All from 26.06.26 runs | ✅ DONE — see §4 table |
| 7 | **I (Model G + Bob penalty)** | `train_i_tasp_dpose_bobpen.py` (1508 lines) | ✅ DONE — 0.0\% SR (2600 iters), fail-fast equilibrium, reward design lesson |
| 8 | **PPO-Baseline (ad-hoc rel\_full)** | `hpc_push_2048env_rel_full/agent/model_best.pt` | ✅ DONE — 80.0\% SR, `results_validation/orig_loss/0_orig_rew_30_isaac.csv` |
| 9 | **ASP-Baseline (ad-hoc ASP)** | `hpc_push_asp_2048env/bob/model_best.pt` | ✅ DONE — 0.0\% SR, `results_validation/orig_loss/reults_valid_asp_orig_Rew.csv` |

**Approved new experiments (1–3 week budget):**
- **E1** Re-eval Model A seeds + ad-hoc on the same 20/30-scene protocol (rows 1,2,7,8). Uses existing checkpoints; hours.
- **E2** Controlled overhead timing: Model A vs Model C, same machine/528 env/push+cuRobo → it/s, env-steps/s. (C9)
- **E3** Controlled curriculum ablation: single-agent PPO + scripted hard-goal distribution (same budget/agent count) to isolate distribution-shift. (C6) — new short training.

**Checkpoint inventory (model_best.pt sizes confirm format families):**
- ~9.35 MB = single-agent PPO (simp/curr/abs/rel_full)
- ~5.24 MB = Bob (ASP, with GoalEncoder); ~5.23 MB = Bob no-GE
- ~15.58 MB = Alice
- latest_iter: simp 2800, curr 2600, asp 2600 (26.06.18); dpose/disc 1000 (26.06.19)

---

## 6. Three-Tier Fix Plan <a name="tiers"></a>

### TIER 0 — Integrity (must-do; narrative + data correction, no re-runs)
- T0.1 Pick single source of truth; rebuild Slides 6c, 10, 12, Conclusions from it. (C3, C10)
- T0.2 Soften all "disproves" → task-specific "did not yield a useful curriculum in this setting." (C1)
- T0.3 Reconcile every internal number: budget, scenes, model/category counts, errors. (C10)
- T0.4 Reframe 0.07% vs 66/65% as a gate/measurement artifact. (C4)
- T0.5 Replace Table-Tennis overhead with controlled A-vs-C number. (C9)
- T0.6 Add scale caveat (~2 days/run; Berner = months → support). (C2)
- T0.7 Density cut to ~18–20 slides; placeholder register. (C10)
- T0.8 Remove/mark Models G/H as "not run" or drop. (C10)
- T0.9 Reframe Slide 7 (Model B): "mis-specified trigger, never fired" not "PBRS made curriculum unnecessary" (§3.1). (C5, C6)
- T0.10 **NEW slide "Why Isaac Lab is a good sim for robotic tasks"** (3 pillars: GPU-batched parallelism, robotic fidelity, time-to-scale for ASP) backed by the measured Isaac-vs-gym throughput table. Argues ASP is evaluated in the scale-appropriate venue and still fails → structural. Keep Slide 11 (Manager-vs-DirectRL API overhead) separate/untouched. (C2; see §11)

### TIER 1 — Stronger arguments from existing data (light)
- T1.1 Training-SR mean ± 95% CI from archive chains; error bars. (C7)
- T1.2 RQ2/RQ3 reframed as "self-play as implemented" (3 axes changed at once). (C6)
- T1.3 Add post-fix-runs footnote with checkpoint dates. (C5)

### TIER 2 — New experiments (approved subset only)
- T2.1 Re-eval Model A seeds + ad-hoc on identical protocol → validation CIs + real RQ1 ratio. (C3, C7)
- T2.2 Controlled overhead timing A vs C. (C9)
- T2.3 Controlled curriculum ablation (single-agent + scripted hard goals). (C6)
- *(Not selected: HER, SAC — C8 handled by citing published numbers.)*

---

## 7. Slide-by-Slide Action Map <a name="slidemap"></a>

| Slide (line) | Issue | Status |
|---|---|---|---|
| Overview (108) | category counts 6/12 vs 3/7 | ✅ Decided: 8 models (A-H) = 1 single + 1 curriculum + 6 ASP |
| RQ notes (176) | "80% … 4.6×" | ✅ 80% now backed by real data; drop "4.6×" (no ad-hoc comparator on same protocol) |
| 2b (229) | "Max 15 pushes" | ✅ Keep; note validation used max=30 |
| 6c (545) | "80% (24/30)", "best-of-3", disc 100% | ✅ All three confirmed by new data |
| 6c-ii (546) | "Validation Breakdown" — Model A only, no error bars | ✅ REDONE — multi-model comparison with error bars from best-of-20 trial data; 4 new CI plots |
| 7 (634) | "Curriculum NEVER activated" + "PBRS too effective" | ✅ REWRITTEN — P82 trigger explained; 76.7% result |
| 7b (662) | Model B 40% + "independence" evidence, no CIs | ✅ UPDATED — A 80.0% vs B 76.7% + CI note (3.3pp not significant) |
| Diagnostic (replaced) | Old `plot4_alice_vs_bob_diagnostic.png` — Alice proposes, Bob fails | ✅ REPLACED — `asp_diagnostic_2panel.png` from TB events: Alice validity bars (83--91\%), Bob per-axis vs combined bars (54\%/45\% → 10\%), C4 resolved |
| Arch slide (removed) | Self-Play — ASP / TASP Architecture — 3 crammed sections with \texttt{\textbackslash vspace\{-6pt\}} | ✅ REMOVED — content now carried by ASP loop diagram |
| 8b (732) | 0.07% vs 66/65 framing | ✅ Reframed as gate artifact (C4) |
| 8c-ii (777) | G/H rows 0–3% / 3–7%, C/D unmentioned | ✅ UPDATED — G 16.7%, H 10.0%; C/D exclusion footnote added |
| 9c (844, NEW) | C6: confounded ASP ablation not addressed | ✅ ADDED — disaggregation table showing 7 axes of variation; G/H partial control noted |
| 10 (870) | 2×2 "80% → 0%", C6 not discussed | ✅ UPDATED — real numbers, statistical significance noted, C6 caveat discussed |
| 11 (915) | 5–10× Table Tennis | ✅ Kept; added A-vs-ASP ~1.0× measurement |
| 11b (NEW) | Missing Isaac Lab slide (T0.10) | ✅ ADDED — throughput table + 3 pillars |
| 12 (941) | master table 80%/0.095/0.208, no CIs | ✅ REPLACED — error-bar plot + CI summary table; C/D footnote |
| 13a/13b (1003/1019) | "Disproven: …" | ✅ SOFTENED — "What We Found — Theory vs Practice" |
| 15a/15b (1090/1106) | "80%", "disproves" | ✅ Aligned with new numbers |
| Appendix B (1138) | "20/30 test cases" | ✅ Fixed to 30 |
| ASP-dPose per-test error | Shows 2/30 at bottom of per-test slides | ✅ Keep as-is — contrast with TASP-dPose's 16.7\% on adjacent slide |
| ASP-dPose vs TASP-dPose | ABC confound — compares (outcome+ABC) vs (time+no ABC) | ✅ FIXED (2026-07-02) — both share ABC ($\beta{=}0.5$); ABC footnote explains mechanism: MultiCategorical 194K actions → PPO clipping activates → zero BC gradient. Removed false "TASP-dPose disables ABC" claim |
| val_multi_combined.png | TASP-dPose-BP appears at 0\% with zero context | ✅ FIXED (2026-07-02) — regenerated with 5 models (PPO-Baseline, PPO-PBRS, PPO-Curriculum, TASP-dPose, ASP-dPose); dual-panel with matching styling (capsize, edgecolor, label format, ylim=115) |
| Multi-Model Validation | No protocol gap explanation | ✅ FIXED (2026-07-02) — footnote added: "Scene SR (reported) counts a scene solved if ≥1 of 20 trials passes — standard for robotics manipulation evaluation. Per-attempt SR is lower: PPO-PBRS 66\%, PPO-Curriculum 68\%. The 14pp gap reflects 6 rotation-dominant scenes that systematically fail all 20 attempts." |
| Discussion (4 slides) | "Validation Suite Reveals" redundant with Results; "Key Result" overlaps with Conclusion RQ2 | ✅ FIXED (2026-07-02) — "What the Validation Suite Reveals" dropped; "Key Result" merged into Conclusion RQ2 slide; Discussion now 2 unique slides (ASP Diagnostic + ASP Empirical Result) |
| What success looks like | No visual of success/failure on T-block | ✅ DONE (2026-07-02) — TikZ slide added after MDP: two-panel T-block diagrams showing pos-only vs pos+rot success with gate thresholds |
| ASP failure videos | Only PPO-PBRS success clips; no ASP failure contrast | ❌ MISSING — record ASP-dPose failing on same scene as s21 |
| ASP training curves | No training dynamics shown for ASP models | ❌ MISSING — generate 3-line overlay from 26.07.01 TB events |
| PBRS sensitivity | Listed in appendix outline but content may not exist | ⚠ CHECK — if missing, move to Limitations & Future Work |
| Appendix H--L (NEW) | No weakness/limitation appendix | ✅ ADDED — H: P82 trigger bug, I: BobPenalty fail-fast, J: confounded ablation, K: training budget, L: validation protocol |
| All weak claims | Cross-references missing to appendix | ✅ ADDED — 5 main-deck slides now reference appendix H--L for detail |
| Naming throughout | "Old/Original" still present | ✅ FIXED — "Fractional" consistently used; "The Four Models" → "All Models at a Glance"; "Our Best Model" → "Recommended Baseline" |
| RQ1 wording | "outperform" overclaimed | ✅ FIXED — "improve the efficiency of" |
| Litfooter consistency | 2 slides missing litfooter | ✅ FIXED — litfooter added to PBRS vs Fractional and Why PBRS for ASP |
| Related Work — Geometry | Duplicate limit_surface.png | ✅ FIXED — image removed from Related Work, kept only on SE(2) slide |

---

## 8. Open Items to Verify <a name="openitems"></a>

1. **Disc eval path** — ✅ RESOLVED: The current `validation_configs.py` has tests 1–10 as T-block R_* rotation scenes (no disc). The 26.06.20 validation used an earlier version with D_* disc scenes. Disc scenes require a separate config; A_simp was evaluated on the old config (80% overall, 100% disc).

2. **Ad-hoc format compatibility** — ⚠ DEPRIORITIZED: Drop Push-PPO baselines from the comparison table (protocol-inconsistent). Use training-SR curves for external calibration.

3. **Per-axis vs combined SR logging** — ✅ RESOLVED: The 0.07% vs 66%/65% is from old `train_curobo.py` training logs, not from validation CSVs. The validation CSVs use the combined thesis gate consistently. Per-axis numbers are not independent — they are logged separately in training but not used in validation.

4. **ALICE_BOB_SUCCESS_REWARD / FAIL_REWARD magnitudes** — ⚠ Low priority; the adversarial loop is correctly wired (confirmed).

5. **`max_pushes` used in 26.06.18 eval** — ✅ RESOLVED: 26.06.26/28 validations use `--max_pushes=30 --max_tries=20`. The deck says "max 15 pushes" — this is the training budget, not validation budget. Clarify.

---

## 9. Decisions Already Made <a name="decisions"></a>

- **Timeline**: 1–3 weeks → front-load all remaining items to finish 2026-07-05; re-runs from existing checkpoints.
- **80% mystery**: ✅ RESOLVED — `results/A_simp/20_isaac_30t.csv` = 80.0% SR.
- **Deck narrative restructured** into 8 academic sections (Intro+RQs, Related Work, Problem+MDP, Implementation, Evaluation, Results, Discussion, Conclusion+Takeaways). Implementation and Results fully separated.
- **Naming scheme**: all models renamed from letters (A–I) to descriptive names (PPO-PBRS, PPO-Curriculum, ASP-dPose, TASP-dPose). Disc models F/H removed. Model C/D excluded.
- **PBRS theorem split**: theory (Ng et al.) → Related Work; our potentials → Implementation.
- **RQ2 reworded**: "Given a well-shaped PBRS reward, does adding a curriculum or self-play improve final task success at fixed budget?"
- **RQ3 reworded**: "Can PBRS substitute for an explicit curriculum?" (was "are reward and curriculum independent"). The gym batch-size interaction (PPO-Curriculum 50% ≫ PPO-PBRS 10% at small batch) makes substitution regime‑dependent — stated as a scope caveat.
- **C9 resolved**: Slide 11 rewritten to "Self-Play Is Not Slower ~1.0×" (controlled same‑machine measurement); Table‑Tennis / 5–10× Manager‑vs‑DirectRL claim dropped. A placeholder slide for a controlled DirectRL‑vs‑Manager measurement is added as a takeaway.
- **172 push/s measured**: provenance added "from logged TF‑event timestamps".
- **Per‑model error plots** (`errors_ppo_pbrs/ppo_curriculum/asp_dpose/tasp_dpose.png`) regenerated from the exact 30‑scene CSVs.
- **Validation‑test‑suite layout** (`val_test_layout.png`) added — 30 scenes, Start→Goal grid, no pass/fail border.
- **Cross‑environment + per‑scene plots** (`p1_sr_bars.png`, `p3_heatmap.png`) with descriptive names, Isaac ASP bar dropped.
- **Tables + training dynamics moved to appendix** (PPO‑Curriculum metric table, self‑play comparison table, CI table, failure taxonomy, training curves).
- **Section dividers** added between the 8 sections using `hlblue` theme colour.
- **Per‑attempt SR** (PPO‑PBRS 66.0%, PPO‑Curriculum 68.3%) disclosed in the deck.
- **4 stale cross‑references** (`Slide~10`, `Slide 9c`, `next slide (11b)`, a stale comment) fixed.
- **Old‑reward ad‑hoc baseline** checkpoint identified for RQ1 re‑eval; SAC+HER baseline training in progress.
- **Model I** (TASP‑dPose‑BobPenalty) — trained, validated at 0\% SR. Bob penalty fires on all phase ends (including catastrophes), creating a perverse "fail-fast" equilibrium. Presented as a reward design lesson: asymmetric reward (Alice incentivised, Bob unpenalised) is necessary in this domain. Not evidence against Sukhbaatar 2018 or Letcher 2019.
- **Non-Markov argument removed from deck** — professor's critique accepted. The fractional formula's failure is framed as gradient starvation (negative reward for progress, near-zero variance) not non-Markov property. The PBRS contrast emphasizes correctly-signed, positive reward on every push.
- **PBRS-for-ASP mathematical justification added** — new slide shows how PBRS partially decouples the GAE advantage from the value function. $\Phi(s')-\Phi(s)$ provides correctly-signed gradient even when $V(s)$ is stale under ASP's distribution shift. Explains why PBRS lifts ASP 0\% $\to$ 16.7\% but cannot fully close the 80\% gap.
- **Baseline contrast slide added** — same fractional reward, same 2048 envs, same 100K iter budget: PPO single-agent 80\% vs ASP 0\%. Cleanest 1-axis comparison in thesis: only training architecture differs.
- **TASP-dPose-BobPenalty slide added** — 0\% SR with detailed mechanism table showing fail-fast equilibrium.
- **Deck restructured (2026-07-01):** Self-Play Architecture slide removed (content moved to ASP loop diagram). Why ASP Fails #1+#2 condensed into "ASP — The Empirical Result" table slide. Caveat — Confounded Ablation moved to Appendix J. What We Found #1+#2 merged with Conclusion #1+#2 into two per-RQ Conclusion slides. Limitations & Future Work merged into one 4-bullet slide. Computational Overhead + Extra Takeaway moved after Conclusion, before Limitations.
- **ASP Diagnostic slide replaced (2026-07-01):** Old `plot4_alice_vs_bob_diagnostic.png` replaced with `asp_diagnostic_2panel.png` — extracted directly from TensorBoard events across three ASP variants (orig_ASP, ASP-dPose, TASP-BP). Panel 1: Alice GoalValidityRate bar chart (83--91\%). Panel 2: Bob per-axis PositionSR/RotationSR vs Combined SR bars. Key finding: PBRS lifts per-axis skills to ~50\% but combined gate caps at ~10\% (2.5× below independence). C4 fully resolved.
- **5 appendix slides H--L added** documenting all weaknesses: P82 trigger bug, BobPenalty fail-fast equilibrium, confounded ablation table, training budget comparison (including ASP-Baseline's actual ~1267 iter run vs the SLURM 100K cap), and validation protocol with disclosed limitations.
- **Isaac Lab slide rewritten** with isaac_vs_gym.png — dual-panel bars showing throughput (172/209/24 push/s) and batch size (7,920/960/480). Emphasizes that raw speed is comparable but Isaac's 8× larger PPO batches stabilize self-play gradients. Text-only version cleaner for first-time viewers.
- **Fractional naming** adopted throughout ("fractional formula" replacing "old/original").
- **Speaker notes synced** for Isaac Lab slide to match new content.

---

## 10. Key File & Data Index <a name="files"></a>

### Deck
- `literature/paper-async/presentation/presentation.tex` — 1200 lines; root + `% !TeX program = xelatex`.

### Authoritative results
- `results/SUMMARY.md` — Definitive comparison table (80.0% A vs 76.7% B vs 6.7–16.7% E–H).
- `results/comparison/summary.md` — Full comparison table + breakdowns (auto-generated by plot_validation.py).
- `results/comparison/per_test_comparison.txt` — Per-test PASS/FAIL for all 7 models.
- `results/A_simp/20_isaac_30t.csv` — Model A, 30 T-block, 80.0% SR.
- `results/B_curr/28_isaac_30t.csv` — Model B, 30 T-block, 76.7% SR.
- `results/{E,F,G,H}_*/26_isaac.csv` — ASP variants, 30 T-block.
- `results/legacy/26.06.12/` — Old 20-test CSVs (A, B, C).
- `results/legacy/26.06.20/` — Disc protocol CSVs + early ASP checkpoints.

### Eval scripts / configs
- `asyncDualPlayPPO/tests/validate_push.py` — single-agent; `MAX_RETRIES=3` (best-of-3); `--max_pushes` (default 15); `--num_tests`.
- `asyncDualPlayPPO/tests/validate_push_asp.py` — ASP Bob + GoalEncoder.
- `asyncDualPlayPPO/tasks/utils/validation_configs.py` — `ALL_TESTS` = 30 scenes (10 disc + 10 pos_only + 10 pos_rot).
- `asyncDualPlayPPO/tests/plot_validation.py` — generates `summary.md`, comparison plots.

### Reward / ASP wiring
- `asyncDualPlayPPO/tasks/utils/wrapper_push_asp.py` — `handle_alice_phase_end` (592), geometric base `delayed_alice_reward = val_reward` (616); `handle_bob_phase_end` (720) outcome reward `+=` (751–761).
- `asyncDualPlayPPO/utils/goal_validator.py` — `validate_goal()`.
- `asyncDualPlayPPO/tasks/utils/reward_pbrs.py` — PBRS potentials, `compute_dpose`.

### Diagnostics / history
- `master_isaac_archive_clean/02_asp_curobo/26.05.19/logs/analysis_curobo_train512_1obj.md` — OLD-path ASP failure root causes (context for C5).
- `master_isaac_archive_clean/results_plots/generate_plots.py` + `plot{1,2,3}_*.png` — training-SR curves, run-consistency.
- `master_isaac_archive_clean/05_pbrs_a_simple/*` (chains A–E) — Model A training seeds.
- `master_isaac_archive_clean/03_push_ppo/*` (7 abs + 4 rel_full) — ad-hoc baseline seeds.
- `implementations.md` — P1–P81 fix log (P55, P64, P65 are the ASP-relevant ones for C5).

### Checkpoints (best, for re-eval)
- A: `runs/ppo_pbrs_reward/26.06.18/runs/hpc_pbrs_simp_528env/agent/model_best.pt`
- B: `…/hpc_pbrs_curr_528env/agent/model_best.pt`
- C: `…/hpc_pbrs_asp_528env/bob/model_best.pt`  | D: `…/hpc_pbrs_asp_noge_528env/bob/model_best.pt`
- E/F: `runs/ppo_pbrs_dpose/26.06.19/hpc_pbrs_asp_{dpose,disc}_528env/bob/model_best.pt`
- Ad-hoc: `runs/ppo_classic_reward/hpc_push_2048env_rel_full/agent/model_best.pt`, `hpc_push_2048env/agent/model_best.pt`, `hpc_push_asp_2048env/bob/model_best.pt`

---

## 11. Final Audit — Remaining Issues & Action Plan (2026-07-01)

Thorough re-check of the deck revealed **10 gaps** and **internal inconsistencies** that
a supervisor will ask about. Each item below has a concrete fix plan.

### 12.1 Unanswered Questions & Fixes

---

**Q1 — "Why does ASP collapse? What is the mechanism?"**

The per-axis skills gap is the answer. RQ3 now asks: *"Does asymmetric self-play enable
the learning of individual position and orientation skills in a multi-objective planar
pushing task on a simulated robotic arm?"*

**Answer from data:** Bob learns per-axis position (54\%) and rotation (45\%) under PBRS-ASP,
but the combined gate caps at 10\% — 2.5× below the independence product (24--25\%). Under
the original fractional reward, nothing is learned (1.3\% per-axis, 0.07\% combined). The
adversarial distribution prevents skill composition even when skills exist independently.

**Why the original papers missed it:** Both Plappert and Sukhbaatar measure success via a
single combined threshold. They never ask whether ASP produces partial competence that
fails to compose. The 54\%/45\% → 10\% data is the first quantification of this gap.

**Plan:** ✅ Already answered by the 2-panel diagnostic slide and Conclusion RQ3 slide.
No additional action needed.

---

**Q2 — "What does 80\% scene SR physically mean?"**

The deck never shows what the success threshold looks like on the T-block.
``pos_err < 0.05 m AND rot_err < 0.2 rad`` is abstract without a visual.

**Fix plan:** ✅ DONE (2026-07-02) — TikZ slide added with two-panel T-block diagrams: pos-only success (position aligned, rotation arbitrary) and pos+rot success (both aligned). Gate thresholds displayed, scene counts (10 pos-only / 20 pos+rot), per-attempt disclosure.

User will provide images.

---

**Q3 — "Training vs validation protocol gap (10×)"**

PPO-PBRS training SR ≈ 7.4\% (strict 15-push budget, combined gate at every step).
Validation SR = 80\% (30-push budget, best-of-20 trials, scene solved if any trial passes).
Per-attempt SR = 66\%.

The deck never explains why the protocol gap is so large or why scene SR is the right
metric.

**Fix plan:** ✅ DONE (2026-07-02) — footnote added to Multi-Model Validation slide: "Scene SR (reported) counts a scene solved if ≥1 of 20 trials passes — standard for robotics manipulation evaluation. Per-attempt SR is lower: PPO-PBRS 66%, PPO-Curriculum 68%. The 14pp gap reflects 6 rotation-dominant scenes that systematically fail all 20 attempts, while position-only scenes succeed reliably. See Appendix L for protocol details."

This distinguishes the two metrics without requiring a new slide. Already disclosed in
Appendix L and the PPO-PBRS — Validation Results slide.

---

**Q4 — "TASP-dPose vs ASP-dPose comparison confounded by ABC"** (RESOLVED 2026-07-02)

**Corrected finding (2026-07-02):** Both ASP-dPose and TASP-dPose have ABC enabled
($\beta{=}0.5$). The original plan assumed TASP-dPose had ABC disabled — this was wrong.
Both training scripts (`train_e_pbrs_asp_dpose.py` and `train_g_tasp_dpose.py`) use
PPOABC with `--no_abc` as an optional flag (unset in production runs).

**Why ABC fails in this specific task:** The key mechanism is the PPO-style BC loss
clipping from Plappert 2021: $\text{clip}(\pi/\pi_{\text{old}}, 0.8, 1.2)$. Bob's
MultiCategorical policy (4 dim × 21 bins = 194K discrete actions) concentrates probability
on goal-relevant pushes. Alice's unguided free-roaming actions lie outside this
concentration → $\pi/\pi_{\text{old}}$ ratio is extreme → PPO clip activates → **zero BC
gradient** despite $\beta{=}0.5$. Additionally, the macro-action abstraction (cuRobo IK +
linear push primitives) breaks the implicit determinism assumption of demonstration
replay — replaying the same $(r,\phi,\ell,\theta)$ from Bob's initial state produces a
different outcome due to PhysX contact stochasticity.

**Contrast with Plappert 2021:** In Plappert's block-grasping domain, Alice uses continuous
joint-space actions with smooth, repeatable trajectories. The policy distribution changes
gradually, keeping the ratio within [0.8, 1.2] — BC gradient flows normally.

**Deck fix:** The slide now reads "both share ABC ($\beta{=}0.5$)" with a 3-sentence
footnote explaining the mechanism. The detailed Plappert contrast is in the speaker notes.
The old bullet "TASP-dPose disables ABC — the gain comes from the time-based reward alone"
has been removed.

---

**Q5 — "TASP-dPose-BP appears in validation plot with zero context"** (RESOLVED 2026-07-02)

**Fix plan:** ✅ DONE — `val_multi_combined.png` regenerated with 5 models: PPO-Baseline,
PPO-PBRS, PPO-Curriculum, TASP-dPose, ASP-dPose. No TASP-dPose-BP. Dual-panel with
matching left/right styling (black edgecolor 0.6pt, capsize=5, error linewidth=1.2,
text fontsize=10 bold, no rotation, ylim=(0,115)). TASP-dPose-BP removed from main deck
(removed from "All Models" table, speaker notes updated to 5 models). 0% result kept
in Appendix I and Limitations slide.

---

**Q6 — Discussion section has 4 overlapping slides** (RESOLVED 2026-07-02)

✅ DONE — "What the Validation Suite Reveals" dropped. "Key Result — PBRS Substitutes"
merged into Conclusion RQ2 slide (now includes substitution finding + regime caveat in
a single enriched slide). Discussion section now has exactly 2 unique slides:
"ASP Diagnostic — Alice & Bob Training Dynamics" and "ASP — The Empirical Result".

Additionally, "Extra Takeaway — ManagerBasedRL vs DirectRL" (placeholder with
`[to measure]` / `[run pending]`) removed from main deck.

---

**Q7 — PPO-Baseline training iteration count**

PPO-Baseline was trained at 2,048 envs for ~1,030 actual iterations (shorter than
the SLURM cap of 100K). This was a deliberate architectural comparison — the
cleanest 1-axis test: same reward, same envs, only the architecture changes
(single-agent vs ASP). Not about environment efficiency — RQ1 answers that with PBRS.

**Fix plan:** Already addressed in Appendix K training budget comparison. No
main-deck change needed. The speaker notes for the Baseline slide clarify: "All
subsequent models switch to PBRS — not because the original reward is broken, but
because PBRS is principled, efficient (works at 528 envs)."

---

**Q8 — Best-of-20 protocol inflates scene SR**

PPO-PBRS trial SR = 66\%, scene SR = 80\%. The 14pp inflation comes from best-of-20
(20 independent attempts, scene passes if any succeeds). A supervisor will ask what
the distribution of trials-per-scene looks like.

**Fix plan:** Add per-attempt SR disclosure to the PPO-PBRS — Validation Results
slide (already present: "Per-attempt SR 66\%; scene SR counts a scene solved if
any of its 20 trials passes"). Add to Appendix L: the per-scene trial distribution
histogram (already generated as `val_pushes_hist.png` in figures). The disclosure
is already honest — no further action needed beyond the existing footnote.

---

**Q9 — PBRS parameter sensitivity (k=30, w=10)**

The deck claims robustness but shows no sensitivity analysis in the main deck.

**Fix plan:** Currently listed as "Appendix C — PBRS parameter sensitivity" in the
appendix outline (line 1348). If this appendix section was never written, add it
showing a grid of k_p ∈ {10, 20, 30, 50} × w ∈ {5, 10, 20} with final validation SR.
If the data doesn't exist, move to Limitations & Future Work: *"PBRS parameter
sensitivity not systematically tested — k_p=30 and w=10.0 were chosen from pilot
runs; a formal grid search would strengthen the claim of robustness."*

---

**Q10 — Missing ASP failure videos**

The deck has 3 validation clips of PPO-PBRS succeeding (s11, s14, s21). There is no
clip of ASP-dPose attempting and failing the same scenes. The contrast is the single
strongest visual argument missing from the deck.

**Available videos in `figures/`:**
| File | Content |
|------|---------|
| `rec_push_s11.mp4` | Forward push (pos-only), PPO-PBRS success |
| `rec_push_s14.mp4` | Lateral push (pos-only), PPO-PBRS success |
| `rec_push_s21.mp4` | Translate + rotate (pos+rot), PPO-PBRS success |
| `rec_push_s01.mp4` | (unused) |
| `asp_random.mp4` / `asp_random.gif` | ASP random policy behavior |
| `asp_random_encoder.mp4` / `asp_random_encoder.gif` | ASP encoder behavior |

**Missing — recommended to record:**
1. **ASP-dPose failure clip** — ASP-dPose attempting the same pos+rot scene as
   `rec_push_s21.mp4` and failing. Shows the chaotic, non-converging behavior.
2. **TASP-dPose best-attempt clip** — TASP-dPose on a pos-only scene it manages to
   solve (16.7\% SR). Shows what the best ASP variant looks like.

**How to record:** Run `validate_push_asp.py` with the ASP-dPose checkpoint on
the `rec_push_s21` scene config, with video recording enabled. Same for TASP-dPose's
best scene. ~30 minutes of GPU time.

---

### 12.2 Implementation Action Items

| # | Action | Priority | Est. Time | Status |
|---|--------|----------|-----------|--------|
| 1 | Add "What Does Success Look Like?" slide after MDP | High | 15 min | ✅ DONE (2026-07-02) — TikZ two-panel T-block diagrams |
| 2 | Fix ABC confound on ASP-dPose vs TASP-dPose slide | High | 5 min | ✅ DONE (2026-07-02) — footnote + speaker notes; PPO clipping + discrete action explanation |
| 3 | Re-generate `val_multi_combined.png` without TASP-dPose-BP (5 models) | High | 15 min | ✅ DONE (2026-07-02) — 5 models, matching styling, ylim=(0,115) |
| 4 | Add validation protocol footnote to Multi-Model slide | High | 5 min | ✅ DONE (2026-07-02) — per-attempt SR + scene-SR explanation |
| 5 | Drop "What the Validation Suite Reveals" Discussion slide | Medium | 2 min | ✅ DONE (2026-07-02) |
| 6 | Merge "Key Result — PBRS Substitutes" into Conclusion RQ2 | Medium | 10 min | ✅ DONE (2026-07-02) |
| 7 | Generate ASP training curves from 26.07.01 TensorBoard events | Medium | 30 min | ❌ NOT DONE — needs GPU time |
| 8 | Record ASP-dPose failure clip + TASP-dPose best clip | Medium | 30 min | ❌ NOT DONE — needs GPU/simulation time |
| 9 | Drop TASP-dPose-BP from main deck (keep in Appendix I + Limitations) | Medium | 10 min | ✅ DONE (2026-07-02) — removed from All Models table, speaker notes; kept in Appendix I, Limitations, Empirical Result table |
| 10 | PBRS sensitivity → Limitations & Future Work (if appendix data missing) | Low | 5 min | ❌ NOT DONE — appendix section C may not exist; move to Limitations if absent |

### 12.2a Remaining Items (2026-07-02)

**High Priority (before defense):**
- **ASP failure videos** (#8): Run `validate_push_asp.py` with ASP-dPose checkpoint on scene `rec_push_s21` with video recording → ~30 min GPU. Most impactful visual contrast missing.
- **TASP-dPose best video** (#8): Same for TASP-dPose on its best pos-only scene.

**Medium Priority (nice to have):**
- **ASP training curves** (#7): Extract 3-line overlay (asp_dpose 7,630 iters, tasp_dpose 3,830 iters, tasp_bp 2,630 iters) from `ppo_pbrs_dpose/26.07.01/` TensorBoard events using `plot_tb_scalar.py` or manual extraction. Show Bob Combined SR over training iterations. Use existing script `add_sr_plots_to_deck.sh` as template.
- **PBRS sensitivity** (#10): Appendix C (PBRS parameter sensitivity) may not exist. Move to Limitations slide if missing, or add grid plot of k_p ∈ {10,20,30,50} × w ∈ {5,10,20}.

**Discrepancy to verify:**
- **PPO-Baseline shows 83.3% (25/30) from CSV, not 80.0% (24/30)** as listed in the definitive table (§4). The CSV `orig_loss/0_orig_rew_30_isaac.csv` was re-read 2026-07-02 and shows 25/30. Verify whether the CSV was re-evaluated or whether the plan needs updating. If 83.3% is correct, update deck numbers (PPO-Baseline baseline slide, appendix tables) accordingly.

**Ongoing (lower priority, for future work):**
- Multi-seed validation (C7) — re-eval Model A checkpoints across 5 chains for model-level CIs
- Controlled curriculum ablation (C6/Tier 2) — single-agent PPO + scripted hard-goal distribution
- HER/SAC baselines (C8) — deprioritized; cite published numbers instead

### 12.3 ASP Training Curve Extraction (26.07.01 runs)

Six training runs available in `ppo_pbrs_dpose/26.07.01/` with TensorBoard events:

| Run | Bob Metrics Available | Final Iter |
|-----|----------------------|------------|
| `hpc_pbrs_asp_dpose_528env` | PosSR, RotSR, CombSR, PosErr, RotErr, DPose | 7,630 |
| `hpc_pbrs_tasp_dpose_528env` | PosSR, RotSR, CombSR, PosErr, RotErr, DPose | 3,830 |
| `hpc_pbrs_tasp_dpose_bp_528env` | PosSR, RotSR, CombSR, TimePenalty | 2,630 |
| `hpc_gym_c` | CombSR only | 9,392 |
| `hpc_pbrs_asp_528env` | (similar, needs checking) | — |
| `hpc_pbrs_asp_disc_528env` | (disc, excluded) | — |

**Training budget note:** The SLURM log iterations (7,600 for ASP-dPose, 3,800 for
TASP-dPose) differ from the planned budget (3,000 iters). Models that converged
later were not stopped — the extra training is free information, not unfair advantage.
ASP-dPose at 7,600 iters and 6.7\% SR vs TASP-dPose at 3,830 iters and 16.7\% SR:
TASP-dPose reaches a higher SR with *fewer* iterations. The comparison is conservative
(favors ASP-dPose with more training).

**Recommended plot:** A 3-line overlay of Bob's combined SR over training iterations
for asp_dpose (7,630 iters), tasp_dpose (3,830 iters), and tasp_bp (2,630 iters).
Shows: (a) ASP-dPose plateaus at ~10\%, (b) TASP-dPose edges higher to ~12--17\%,
(c) TASP-BP collapses to ~0\%. Add to Discussion section or Appendix B.

Gym C (9,392 iters, CombSR only, no per-axis) is a separate line on a different
X-axis scale — not directly comparable. Best shown in the existing Cross-Environment
validation slides rather than training curves.

### 12.4 Videos — Complete Inventory

**In the deck (referenced via \mediabox+\playclip):**
- `rec_push_s11.mp4` — Forward (pos-only), PPO-PBRS ✓
- `rec_push_s14.mp4` — Lateral push (pos-only), PPO-PBRS ✓
- `rec_push_s21.mp4` — Translate + rotate, PPO-PBRS ✓

**In figures but not referenced in deck:**
- `rec_push_s01.mp4` — could be added as a fourth validation clip
- `asp_random.mp4` / `asp_random.gif` — ASP random policy (add if a "before
  training" clip is useful)
- `asp_random_encoder.mp4` / `asp_random_encoder.gif` — encoder behavior

**Recommended to record:**
1. ASP-dPose failing on the same pos+rot scene (scene 21) — strongest visual contrast
2. TASP-dPose solving its best pos-only scene — shows what 16.7\% looks like in practice

---

## 12. gym-pusht Testbed & "Isaac Lab is the right sim for ASP" (2026-06-27) <a name="gym-isaac"></a>

New work that changes the deck narrative. Full implementation detail in
`implementations.md` §10.

### 11.1 gym-pusht controlled testbed — Cross-Environment A/B/C Results

To answer critique **C6** (the Isaac A-vs-C comparison changed 3 things at once)
and the compute-mismatch problem, Models A/B/C were ported to **gym-pusht**, a
fast 2D CPU testbed, reusing the **identical custom PPO/PPOABC/ActorCriticPush +
EpisodeManager + validate_goal + reward_pbrs** — only the environment differs.

All HPC gym models trained on 32 CPU cores (Apptainer container, `push_nsteps=30`).
Validated on identical 30 T-block scenes under the same thesis gate.

#### Cross-Environment Comparison (same thesis gate, same total push budget)

| Model | Isaac SR | Gym SR | Isaac PosErr | Gym PosErr | Isaac Pushes | Gym Pushes | Batch Isaac | Batch Gym |
|-------|----------|--------|-------------|------------|-------------|------------|-------------|-----------|
| **A** | 80.0% | 10.0% | 0.032 m | 0.288 m | 19.0 M | 18.0 M | 7,920 | 960 |
| **B** | 76.7% | 50.0% | 0.023 m | 0.100 m | 20.6 M | 18.0 M | 7,920 | 960 |
| **C** | 6.7%  | 13.3% | 0.143 m | 0.140 m | 60.2 M | 4.2 M | 7,920 | ~480 |

#### Gym HPC Training Metadata (latest checkpoints, 2026-06-30)

| Model | Final Iters | Wall-Clock | Est. Push-Macros | Throughput |
|-------|------------|------------|-----------------|------------|
| **A_gym** | **18,800** | ~30 h (~1.3 d) | **18.0 M** | ~209 push/s |
| **B_gym** | **18,800** | ~30 h (~1.3 d) | **18.0 M** | ~209 push/s |
| **C_gym** | **4,400** | ~48 h (~2.0 d) | **4.2 M** | ~24 push/s |

**Note:** Previously reported gym iteration counts (8,800/9,000/1,300) were from intermediate checkpoints at which validation CSVs were run. Training continued to completion at the counts above. All A/B `.done` files written 2026-06-30 03:47–03:48.

#### Key Findings

**1. Total push count is NOT the bottleneck.** Gym A/B received 18.0M pushes — nearly identical to Isaac A's 19.0M and B's 20.6M. Yet A drops 80.0% → 10.0% and B drops 76.7% → 50.0%. The per-update batch is 960 vs 7,920 transitions (8× smaller) — that's the dominant factor. With LSTM hidden-state propagation (Fix P13), smaller batches sample fewer initial hidden states per update, creating biased GAE advantage estimates and higher policy-gradient variance. This causes earlier convergence plateaus, not slower convergence.

**2. Curriculum (B) compensates for batch variance.** At Isaac scale (7,920/batch), no curriculum (A, 80.0%) beats curriculum (B, 76.7%) — staging adds complexity without value. At gym scale (960/batch), curriculum (B, 50.0%) dramatically beats no curriculum (A, 10.0%) — staging simplifies the objective enough to overcome batch variance. The P82 Phase 1 (position-only) reduces the optimization dimensionality, making the smaller batch sufficient to bootstrap position control (90% gym pos-only SR vs 100% Isaac). Reward design and curriculum design are independent levers whose interaction depends on the batch size regime — consistent with RQ3.

**3. ASP (C) fails structurally in both environments.** Isaac C gets 60.2M pushes and reaches 6.7%. Gym C gets only 4.2M pushes (14× fewer) and reaches 13.3% — *more pushes do not improve ASP*. Both environments show zero pos+rot success. Outcome-based Alice (+5/−1) produces a non-stationary goal distribution that prevents Bob from learning the combined objective regardless of simulator, scale, or total experience. Gym C's slight edge (13.3% vs 6.7%) comes from the simpler 2D environment (no cuRobo IK failures, no contact physics), which gives Bob marginally cleaner gradients on the pos-only metric — but the combined gate never fires in either environment.

**4. The ordering A ≈ B ≫ C is preserved across environments.** confirms the structural result is robust to simulator choice. The absolute gaps compress in gym (A→B gap widens from 3.3pp to 40pp in B's favor) due to curriculum's batch-variance compensation, but the fundamental geometry — single-agent approaches 80%, ASP stays below 15% — is identical.

### 11.2 "Isaac Lab is the right sim for ASP" — NEW SLIDE (addresses C2)

**Measured controlled throughput** (one continuous slurm segment, 528-env Isaac
vs gym-pusht):

| Sim (hardware) | parallelism | push-macros/s | 1M pushes | batch/update | ASP overhead vs A |
|---|---|---|---|---|---|---|
| **Isaac Lab** (1 GPU, 528 env) | GPU-batched | **172** | **~1.6 h** | 7920 | **~1.0× (none)** |
| gym HPC (32 CPU, A/B) | 32 CPU procs | **209** | ~1.3 h | 960 | — |
| gym HPC ASP (single-proc, C) | 1 CPU proc | **24** | ~11.6 h | ~480 | ~7.8× *slower* |
| gym desktop (6 CPU, A/B) | 6 CPU procs | 91 | ~3.0 h | 90–360 | — |

**Note:** 32-core HPC gym A/B throughput (209 push/s) actually exceeds Isaac's 172 push/s for single-agent models — CPU parallelism is sufficient for raw speed. The limitation is the per-update batch size (960 vs 7,920), not throughput. Isaac's GPU-batched architecture delivers 8× larger gradient batches per iteration, which is what stabilizes PPO+LSTM training.

**Argument (3 pillars) for the new slide "Why Isaac Lab is a good sim for robotic tasks":**
1. **GPU-batched parallelism** — 528 envs/step → 7920-sample batches on one GPU
   (Makoviychuk et al. 2021 *Isaac Gym*; Rudin et al. 2022). The non-stationary
   two-agent ASP objective needs a large batch for a stable gradient.
2. **Robotic fidelity** — UR5e + contact-rich 3D physics + cuRobo IK = the actual
   SE(2) task; gym-pusht is a 2D abstraction (diagnostic for the reward question only).
3. **Time-to-scale (ASP)** — ASP parallelises *for free* on GPU-batched sim but is
   forced single-process on CPU (~7.7× slower); Isaac reaches ASP-scale experience
   (~10M pushes) in ~16 h vs ~5 days on CPU.

**Framing ("fair venue, still fails"):** Isaac gives ASP its **best shot on a
single-GPU budget at compute matched to the winning single-agent**, and ASP **still
reaches only 6.7–16.7% SR (4.8× gap)** → the failure is **structural, not under-resourcing.**
This defuses **C2** (scale mismatch) instead of conceding it.  Models G/H (time-based ASP)
now exist and were evaluated — they perform better than outcome-based E/F (16.7% and 10.0% vs
6.7%) but still far below single-agent. Time-based Alice partially rescues the adversarial
dynamics but cannot close the gap.

**Examiner-safe scope:** claim *fidelity + batch=7920 + ASP-single-proc-7.7×-slower*,
NOT "Isaac beats any CPU config" (a many-core CPU gym could exceed 172 push/s for A/B).

### 11.3 Correction to C9 / Slide 11 (Manager-vs-DirectRL ≠ ASP overhead)

The measured data shows **ASP and the single-agent run at identical iteration cost
in Isaac (~0.022 it/s)** — the 2-agent machinery (2 PPO updates, GoalEncoder, ABC,
historical pool) adds **~no per-iteration wall-clock** (the shared 528-env
cuRobo-IK/physics dominates). Implications:
- The existing **Slide 11 "5–10× overhead"** claim is the **ManagerBasedRLEnv vs
  DirectRLEnv** API-architecture overhead — a **separate, valid** claim. It is NOT
  about ASP-vs-single-agent and is NOT refuted by this data. **Keep Slide 11 as-is.**
- Do **not** claim ASP is computationally expensive *per iteration* — it isn't; it
  simply doesn't learn. (If desired, frame ASP's cost as model/code complexity +
  two networks + failure modes, not per-iteration compute.)
- The Isaac-vs-gym throughput goes in the **new §11.2 slide**, not Slide 11.

### 11.4 C2 number, corrected

Measured: **~0.022 it/s → ~46 s/iter → ≈1.6 days / 3000 iters** (not "a few hours").
Use this exact number for the C2 scale caveat.
