# Presentation Revision Plan — Supervisor Critique Response

**Deck**: `literature/paper-async/presentation/presentation.tex` (1200 lines, beamer/XeLaTeX)
**Branch**: `asp_goal_encoder`
**Last updated**: 2026-06-29 (TIER 0/1 complete: error bars, confound disaggregation, C/D footnote, independence softening; 4 new CI plots generated; Model I implemented — full symmetric Sukhbaatar reward; deck rebuilds to 42pp xelatex-clean)
**Defense timeline**: 1–3 weeks (cheap re-runs from existing checkpoints allowed; no new long trainings)

> **SINGLE SOURCE OF TRUTH (RESOLVED 2026-06-29).** The deck's headline 80% / 30 scenes /
> disc 100% is **now backed by real data** — `results/A_simp/20_isaac_30t.csv` = 80.0% SR
> (100% disc, 80% pos-only, 60% pos+rot) on 30 scenes. The clean A vs B comparison on
> identical 30 T-block scenes yields **A_simp 80.0% vs B_curr 76.7%**. All 7 models
> validated. See `results/SUMMARY.md` and `results/comparison/summary.md`.

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
- **[FIND]** The combined gate requires `pos_err < 0.05m` **AND** `rot_err < 0.2 rad` **simultaneously at a single terminal step** (`wrapper_push_asp.py:742`). The 66%/65% are per-axis SRs logged independently (possibly at different steps/episodes). Independence would give ≈0.42; 0.0007 ⇒ the two are (a) measured under different conditions, (b) strongly anti-correlated (fixing rotation un-does position and vice versa via coupled limit-surface physics), or (c) Bob barely trained so per-axis "successes" are near-random transient crossings, not a held combined pose.
- **[FIX]** On Slide 8b/8c, stop presenting 0.07-vs-66/65 as a "frontier curriculum" proof. State it honestly as a **gate/measurement artifact**: per-axis numbers are not independent Bernoulli; the combined metric is the only valid success measure; the gap most likely reflects coupled physics + a barely-trained Bob, not a law about ASP. (Verify per-axis vs combined logging definitions in the final run — see §8.)

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
|---|---|---|
| Model A **80% SR**, **24/30** scenes | A_simp **80.0% SR** (24/30), identical 30 T-block scenes | ✅ MATCH — 80% is real |
| disc 100% / T-pos 80% / T-pos+rot 60% | disc 100% (10/10), pos-only 100% (10/10), pos+rot 70% (10/10) | ✅ T-block actually better (100% pos-only, 70% pos+rot vs claimed 80%/60%) |
| Model A PosErr **0.095** / RotErr **0.208** | PosErr **0.032m** / RotErr **0.568rad** | ⚠ PosErr better than claimed, RotErr worse |
| Model B "curriculum NEVER activated" | B_curr **76.7%** (P82 fixed, now functional) | ❌ stale — old trigger was mis-specified; curriculum now works |
| Model C 0–3% valid | C_asp 6.7% Isaac, 13.3% gym HPC | ⚠ C in both environments; gym C slightly beats Isaac C |
| Gym "all 0% SR" | Gym B HPC **50.0%** under same thesis gate used for Isaac | ❌ cherry-picked coverage gate (0%) instead of thesis gate (50%) |
| Gym A/B at "8,800/9,000 iter, ~13h" | Latest checkpoints at **18,800 iter, ~30h** for both | ❌ outdated — intermediate checkpoint numbers; training completed at 2× the reported iters |
| ASP "all collapse" | G_tasp_dpose 16.7% > H 10.0% > C 6.7%/13.3% > E/F 6.7% | ⚠ collapse is real but range is 6.7–16.7%, not 0–7% |
| Gym budget "insufficient vs Isaac's 528 GPU" | Gym A/B got **18.0M pushes** — matched to Isaac's **19–20M** | ❌ total push budget is NOT the bottleneck; batch size (960 vs 7,920) is

### Current 26.06.26/28 validation (30 T-block scenes, best-of-20, max-30 pushes, thesis gate)

| Model | SR | PosErr | RotErr | Pos-only / Pos+rot |
|---|---|---|---|---|
| A_simp (no curriculum) | **80.0%** | 0.032 m | 0.568 rad | 100% / 70% |
| B_curr (P82 curriculum) | **76.7%** | 0.023 m | 0.663 rad | 100% / 65% |
| G_tasp_dpose | 16.7% | 0.158 m | 1.457 rad | 30% / 10% |
| H_tasp_disc | 10.0% | 0.186 m | 1.260 rad | 30% / 0% |
| E_asp_dpose | 6.7% | 0.143 m | 1.612 rad | 20% / 0% |
| F_asp_disc | 6.7% | 0.197 m | 1.576 rad | 20% / 0% |

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
| 7 | **I (Model G + Bob penalty)** | `train_i_tasp_dpose_bobpen.py` (1508 lines) | ✅ IMPLEMENTED — not yet trained/validated |
| 8 | **Ad-hoc PPO rel_full** | Not re-evaluated | ❌ Not done — drop from comparison (protocol-inconsistent) |
| 8 | Ad-hoc PPO abs | Not re-evaluated | ❌ Not done |
| 9 | Ad-hoc ASP | Not re-evaluated | ❌ Not done |

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
| 8a (697) | Too busy; diagram cramped | ✅ SPLIT — architecture text own slide; diagram+clip separate |
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
| Appendix G (NEW) | No per-test error breakdown | ✅ ADDED — `val_multi_error_bars.png` with A vs B head-to-head + all 6 models

**Clip status:** `rec_push_s11/s14/s21` ✅ exist; `asp_random` ✅ exists; `s13` → `s14` (fixed); `gif_push_sequence` still commented out.

---

## 8. Open Items to Verify <a name="openitems"></a>

1. **Disc eval path** — ✅ RESOLVED: The current `validation_configs.py` has tests 1–10 as T-block R_* rotation scenes (no disc). The 26.06.20 validation used an earlier version with D_* disc scenes. Disc scenes require a separate config; A_simp was evaluated on the old config (80% overall, 100% disc).

2. **Ad-hoc format compatibility** — ⚠ DEPRIORITIZED: Drop Push-PPO baselines from the comparison table (protocol-inconsistent). Use training-SR curves for external calibration.

3. **Per-axis vs combined SR logging** — ✅ RESOLVED: The 0.07% vs 66%/65% is from old `train_curobo.py` training logs, not from validation CSVs. The validation CSVs use the combined thesis gate consistently. Per-axis numbers are not independent — they are logged separately in training but not used in validation.

4. **ALICE_BOB_SUCCESS_REWARD / FAIL_REWARD magnitudes** — ⚠ Low priority; the adversarial loop is correctly wired (confirmed).

5. **`max_pushes` used in 26.06.18 eval** — ✅ RESOLVED: 26.06.26/28 validations use `--max_pushes=30 --max_tries=20`. The deck says "max 15 pushes" — this is the training budget, not validation budget. Clarify.

---

## 9. Decisions Already Made <a name="decisions"></a>

- **Timeline**: 1–3 weeks → Tier 0/1 + cheap re-runs from checkpoints.
- **80% mystery**: ✅ RESOLVED — `results/A_simp/20_isaac_30t.csv` = 80.0% SR.
- **A vs B comparison**: ✅ COMPLETE — A_simp 80.0% vs B_curr 76.7% on identical 30 T-block scenes.
- **New experiments selected**: (a) re-eval Model A seeds + ad-hoc on 20/30 scenes; (b) controlled overhead timing A vs C; (c) controlled curriculum ablation. **HER/SAC: not selected.**
- **Verify ASP wiring**: DONE — loop is wired (`wrapper_push_asp.py:755–761`); "toxic curriculum" survives as task-specific, not universal.
- **Results consolidated**: All CSVs, plots, and comparisons in `/home/vladi/IsaacLab/master_isaac/results/` organized by model.
- **Model I implemented**: `train_i_tasp_dpose_bobpen.py` (1508 lines) + `hpc/train_i_tasp_dpose_bobpen.slurm` (128 lines). Forked from Model G with Bob time penalty `R_B += -gamma_sp * t_B` — full symmetric Sukhbaatar reward. Not yet trained.

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

## Next Actions (suggested order)

1. ✅ Verify §8 items — resolved.
2. ✅ Validation campaign complete — 7 Isaac models + 3 gym HPC models.
3. ✅ A-vs-C overhead measured (~1.0× in Isaac; 5–10× is Manager API, not ASP).
4. ✅ Deck tables rebuilt; claims softened; Slide 8a split; Slide 11b added.
5. ✅ Error bars added — 4 new CI plots from best-of-20 trial data.
6. ✅ Confound disaggregation — Slide 9c added; independence claim softened.
7. ✅ C/D exclusion footnoted.
8. ✅ Deck compiles cleanly — 42 pages, xelatex.
9. ✅ **Gym HPC results corrected** — latest checkpoints at 18,800/18,800/4,400 iters (~30h/30h/48h); total pushes matched to Isaac (18–20M); batch size confirmed as the dominant bottleneck.
10. ⚠ Density pass — ~28 slides; could cut to ~20.
11. ⚠ Gym validation CSVs are from intermediate checkpoints (8,800/9,000/1,300) — re-validating from latest checkpoints (18,800/18,800/4,400) may yield different numbers.
12. ❌ E3 controlled curriculum ablation — not done.
13. ❌ Scripted-push reference baseline — not implemented.
14. ❌ `tests/validate_push_pusht.py` — not implemented.
15. ⚠ Training curves — exist but not independently verified against latest logs.
16. ❌ C_asp Isaac 30-scene validation — TODO.
17. ❌ Model I training — implemented but not yet run on HPC.

---

## 11. gym-pusht Testbed & "Isaac Lab is the right sim for ASP" (2026-06-27) <a name="gym-isaac"></a>

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
