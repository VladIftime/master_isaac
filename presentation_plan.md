# Presentation Revision Plan — Supervisor Critique Response

**Deck**: `literature/paper-async/presentation/presentation.tex` (1200 lines, beamer/XeLaTeX)
**Branch**: `asp_goal_encoder`
**Last updated**: 2026-06-25
**Defense timeline**: 1–3 weeks (cheap re-runs from existing checkpoints allowed; no new long trainings)

> **SINGLE SOURCE OF TRUTH (decision pending).** The deck's headline numbers
> (80% / 30 scenes / best-of-3 / disc 100%) are **not present in any saved CSV**.
> The only complete saved eval is `runs/ppo_pbrs_reward/26.06.18/comparison_plots/summary.md`
> = **60% / 20 T-block scenes / best-of-3**. Until we regenerate a full 30-scene eval,
> treat the 26.06.18 numbers as ground truth and flag everything else as "to-regenerate".

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
5. **G/H (time-based ASP) were never run** — only `asp_dpose` (=E) and `asp_disc` (=F) runs exist. Deck reports G/H numbers anyway. (→ C10)
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

Source of truth = `runs/ppo_pbrs_reward/26.06.18/comparison_plots/summary.md` + `per_test_comparison.txt`.

| Deck claim | Saved data (26.06.18) | Verdict |
|---|---|---|
| Model A **80% SR**, **24/30** scenes | PPO-Base **60.0% SR**, **12/20** scenes | ❌ mismatch — "80%" = pos-only sub-score |
| **30** held-out scenes, **best-of-3** | **20** T-block scenes evaluated (best-of-3 protocol exists for 30) | ⚠ disc scenes not run |
| disc 100% / T-pos 80% / T-pos+rot 60% | No disc; **Pos-only 80% / Pos+rot 40%** (T-block) | ❌ mismatch |
| Model A PosErr **0.095** / RotErr **0.208** | PosErr **0.202 m** / RotErr **0.893 rad** | ❌ mismatch |
| Model B **40%** | PPO-Curriculum **15%** (40% only on 26.06.15) | ❌ stale |
| Model C 0.07% train, 0–3% valid | ASP **0.0%** valid (20 scenes) | ⚠ "0–3%" vague |
| Models **G/H** = 0–3% / 3–7% | **No G/H runs exist** | ❌ unbacked |
| Push-PPO baseline **17.3%** | ad-hoc runs 0.8%→23.8%; 17.3% = one rel_full run | ⚠ cherry-picked |
| Model A **training SR 8.75%** | PBRS-A peak training SR ≈ **9.4%** | ✅ ~matches |

### Saved 26.06.18 validation (20 T-block scenes, best-of-3, max-30 pushes)

| Model | SR | PosErr | RotErr | Easy/Med/Hard | Pos-only / Pos+rot |
|---|---|---|---|---|---|
| PPO-Base (A) | **60.0%** | 0.202 m | 0.893 rad | 100/50/50 | 80 / 40 |
| PPO-Curriculum (B) | 15.0% | 0.154 m | 1.754 rad | 0/33/10 | 10 / 20 |
| PPO-ASP (C) | 0.0% | 0.457 m | 1.554 rad | 0/0/0 | 0 / 0 |
| PPO-ASP-NGE (D) | 0.0% | 0.478 m | 1.442 rad | 0/0/0 | 0 / 0 |

---

## 5. Evaluation Matrix — Which Checkpoints to Run <a name="evalmatrix"></a>

Goal: produce **one authoritative, defensible results table** from existing checkpoints
(best-of-3, identical 30-scene protocol where possible). All `model_best.pt`.

| # | Model | Checkpoint | Eval script | Scenes | Priority | Purpose |
|---|---|---|---|---|---|---|
| 1 | **A (PBRS simp)** | `runs/ppo_pbrs_reward/26.06.18/runs/hpc_pbrs_simp_528env/agent/model_best.pt` (it 2800) | `validate_push.py` | 30 (disc 1–10 + tblock 11–30) | ★ | Reproduce/replace 80% headline |
| 2 | A — seeds | 26.06.15/16/17 `simp/agent/model_best.pt` + archive chains A–E | `validate_push.py` | 30 | ★ | Validation **mean ± CI** (C7) |
| 3 | **B (curr)** | `26.06.18/.../hpc_pbrs_curr_528env/agent/model_best.pt` (it 2600) | `validate_push.py` | 30 | high | parity with A |
| 4 | **C (ASP)** | `26.06.18/.../hpc_pbrs_asp_528env/bob/model_best.pt` (it 2600) | `validate_push_asp.py` | 30 | high | Bob + GoalEncoder |
| 5 | **D (ASP no-GE)** | `26.06.18/.../hpc_pbrs_asp_noge_528env/bob/model_best.pt` | `validate_push_asp.py` | 30 | high | GoalEncoder ablation |
| 6 | **E/F (dpose/disc)** | `26.06.19/.../{asp_dpose,asp_disc}/bob/model_best.pt` (it 1000) | `validate_push_asp.py` | 30 | med | only ASP-variant runs that exist |
| 7 | **Ad-hoc PPO rel_full** | `runs/ppo_classic_reward/hpc_push_2048env_rel_full/agent/model_best.pt` | `validate_push.py` | 30 | ★ | fairest ad-hoc comparator (C3) |
| 8 | Ad-hoc PPO abs | `runs/ppo_classic_reward/hpc_push_2048env/agent/model_best.pt` | `validate_push.py` | 30 | low | secondary |
| 9 | Ad-hoc ASP | `runs/ppo_classic_reward/hpc_push_asp_2048env/bob/model_best.pt` | `validate_push_asp.py` | 30 | med | 2×2 "ad-hoc + ASP" cell |

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

| Slide (line) | Issue | Action |
|---|---|---|
| Overview (108) | category counts 6/12 vs 3/7 | standardize to 8 models / 6 ASP variants |
| RQ notes (176) | "80% … 4.6×" | replace with same-protocol numbers |
| 2b (229) | "Max 15 pushes" | keep; ensure consistency w/ ASP budget statement |
| 6c (545) | "80% (24/30)", "best-of-3", disc 100% | rebuild from 30-scene re-eval (or state 60%/20) |
| 7 (634) | "Curriculum NEVER activated" + "PBRS too effective → EMA never stabilizes" cause | reframe: **mis-specified trigger, unreachable by construction** (§3.1), not "PBRS made curriculum unnecessary" |
| 7b (662) | Model B 40% + "independence" evidence | use 15% (saved); state Model B is **not** a clean curriculum test (trigger never fired) |
| 8a (697) | Alice 15 / Bob ≤50 | reconcile with archive (Alice 5 / Bob 10) |
| 8b (732) | 0.07% vs 66/65 framing | reframe as gate artifact (C4) |
| 8c-ii (777) | G/H rows | mark "not run" or remove |
| 10 (870) | 2×2 "80% → 0%" | use authoritative numbers |
| 11 (915) | 5–10× Table Tennis | controlled A-vs-C timing |
| 12 (941) | master table 80%/0.095/0.208 | rebuild from re-eval |
| 13a/13b (1003/1019) | "Disproven: …" | soften to task-specific |
| 15a/15b (1090/1106) | "80%", "disproves" | align with new numbers |
| Appendix B (1138) | "20/30 test cases" | fix to actual count |

**Clip placeholder register** (`\mediabox`/`\playclip`): `rec_push_s11/s13/s21.mp4`, `asp_random.mp4`, `gif_push_sequence.gif` (commented), `*_key.png` stills. Decision per clip: **record** (run `record_push_video.py` on Model A best ckpt) **or remove**.

---

## 8. Open Items to Verify <a name="openitems"></a>

1. **Disc eval path** — `validate_push.py` builds one `target_object` at startup; disc scenes (1–10) need the disc env (`CylinderCfg`). Confirm whether the script takes a `--disc`/disc-config flag or needs a separate launch. Determines if "disc 100%" for Model A is obtainable.
2. **Ad-hoc format compatibility** — `ppo_classic_reward` 2048-env runs predate the 4D×21 / rel-mode refactors (P34, P50, P62). Confirm `validate_push.py` can load them with matching flags; else RQ1 falls back to training-SR curves only.
3. **Per-axis vs combined SR logging** — confirm in the final PBRS-C run logs how `Metrics/Bob/PositionSR` / `RotationSR` / combined are computed (same step? same episode?) to substantiate the C4 explanation.
4. **ALICE_BOB_SUCCESS_REWARD / FAIL_REWARD magnitudes** — read `reward_utils` to confirm the adversarial signal magnitude vs the geometric base (dilution check for C1/C4).
5. **`max_pushes` used in 26.06.18 eval** — summary shows up to 30 pushes; script default is 15. Confirm the flag used so re-eval matches.

---

## 9. Decisions Already Made <a name="decisions"></a>

- **Timeline**: 1–3 weeks → Tier 0/1 + cheap re-runs from checkpoints.
- **80% mystery**: search disk + regenerate 30-scene eval; treat 26.06.18 (60%/20) as interim truth.
- **New experiments selected**: (a) re-eval Model A seeds + ad-hoc on 20/30 scenes; (b) controlled overhead timing A vs C; (c) controlled curriculum ablation. **HER/SAC: not selected.**
- **Verify ASP wiring**: DONE — loop is wired (`wrapper_push_asp.py:755–761`); "toxic curriculum" survives as task-specific, not universal.

---

## 10. Key File & Data Index <a name="files"></a>

### Deck
- `literature/paper-async/presentation/presentation.tex` — 1200 lines; root + `% !TeX program = xelatex`.

### Authoritative results
- `asyncDualPlayPPO/runs/ppo_pbrs_reward/26.06.18/comparison_plots/summary.md` — 60%/20-scene table.
- `…/26.06.18/comparison_plots/per_test_comparison.txt` — per-test PASS/FAIL.
- `…/26.06.18/runs/hpc_pbrs_{simp,curr,asp,asp_noge}_528env/results_*.csv` — raw eval CSVs.
- `…/26.06.15/hpc_pbrs_{simp,curr,asp}_528env/results_*.csv` — earlier (simp 50%, curr 40%, asp 0%).
- `…/ppo_pbrs_dpose/26.06.19/hpc_pbrs_asp_{dpose,disc}_528env/results_asp_disk.csv` — E/F disc evals.

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

1. Verify §8 open items (disc flag, ad-hoc compat, logging defs, reward magnitudes, max_pushes).
2. Write a batch eval script over the §5 matrix → produce one authoritative `results/*.csv`.
3. Extract A-vs-C overhead timing from slurm logs (C9).
4. Rebuild deck tables (Slides 6c, 8c, 10, 12, 15) + soften claims (Slides 13) from new numbers.
5. Density pass: cut to ~18–20 slides; record or remove clip placeholders.
6. (If time) run E3 controlled curriculum ablation.
