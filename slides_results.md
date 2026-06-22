# Slides — Push-T Validation Results

Held-out validation: 30 unseen scenes per model (10 disc = position only; 10
T-block position-only; 10 T-block position+rotation), run by
`asyncDualPlayPPO/tests/validate_push.py` (single-agent) and
`validate_push_asp.py` (self-play Bob). Success = pos < 0.05 m **and**
rot < 0.2 rad (T-block) or pos < 0.05 m (disc). Charts:
`asyncDualPlayPPO/data_analysis/plots/why_pbrs/generate_results_plots.py`.

| Model | Overall | Disc | Pos-only | Pos+Rot |
|---|---|---|---|---|
| Diffusion Policy (imitation, demos) | 53 % | 30 % | 90 % | 40 % |
| **PBRS single-agent (RL, no demos, no curriculum)** | **80 %** | **100 %** | 80 % | **60 %** |
| ASP (disc) — same PBRS reward + self-play | 3 % | 0 % | 10 % | 0 % |
| time-ASP (disc) — same reward + self-play | 3 % | 0 % | 10 % | 0 % |
| time-ASP (T-block) — same reward + self-play | 0 % | 0 % | 0 % | 0 % |

---

## Slide 1 — Reward-driven RL beats imitation on Push-T

![DP vs PBRS](asyncDualPlayPPO/data_analysis/plots/why_pbrs/results_dp_vs_pbrs.png)

**Bullets**
- **30 unseen held-out scenes** (disc = position; T-block position-only; T-block position+rotation).
- **PBRS single-agent (no demos, no curriculum) 80 %** vs **Diffusion Policy (trained on demonstrations) 53 %.**
- **Wins where it's hard:** unseen disc **100 % vs 30 %** (RL transfers to a new object; imitation doesn't); coupled pose **60 % vs 40 %.**
- **Controls orientation even when unscored:** on position-only tests PBRS keeps the T aligned (< 0.2 rad) while Diffusion Policy leaves it rotated up to ~2.8 rad — the coupled `d_pose` reward at work.

**Speaker notes (~45 s):** Thirty start-and-goal scenes the policy never trained on — easy, medium, hard, including the symmetric disc and the full position-plus-angle T-block. Our single-agent PBRS agent, trained from scratch with no demonstrations and no curriculum, solves eighty percent; Diffusion Policy — state-of-the-art imitation, trained on expert demos of this exact task — gets fifty-three. The gap is biggest where the task is hard: a hundred versus thirty on the disc, an object the imitation policy never saw, and sixty versus forty on the coupled goals. One detail makes the point — on position-only scenes Diffusion Policy parks the block at almost any angle, while PBRS keeps it aligned even though angle isn't scored there, because the single pose-distance reward couples them exactly like the physics does.

---

## Slide 2 — The reward is sound; the self-play curriculum is the open problem

![Self-play collapse](asyncDualPlayPPO/data_analysis/plots/why_pbrs/results_selfplay_collapse.png)

**Bullets**
- **Same PBRS reward, four self-play curricula** (asymmetric + time-based, disc & T-block) → **0–3 %.**
- These are **not near-misses** — final errors 0.2–0.8 m; the solver never learned to push toward goals.
- Reward identical to the 80 % agent ⇒ the failure is the **curriculum**, not the reward.
- **Takeaway:** PBRS is the right *reward* — even, and especially, with no curriculum; automatic self-play curriculum is a separate, still-open problem.

**Speaker notes (~40 s):** For honesty about scope: we ran the *same* PBRS reward under self-play — Alice-and-Bob and its time-based variant, on both objects. Every version collapses to zero-to-three percent, and these aren't fine-tuning gaps — the block ends up twenty to eighty centimetres off; the solver never learned to approach the goal. Since the reward is byte-for-byte the one that scored eighty, the reward can't be the problem — it's the self-play curriculum, whose shifting goal distribution starves the learner.

---

## Slide 3 — Discussion: why ASP loses, what Diffusion Policy tells us

**Bullets**
- **ASP and single-agent PBRS share the identical solver reward** — so 80 % → 0–3 % is caused by the *curriculum*, not the reward.
- **ASP swaps a stationary goal distribution for a moving one:** Alice is still learning, so Bob chases a non-stationary target (a two-player Markov game that breaks PPO's trust region); the curriculum never stabilises at Bob's frontier — Alice swings to *impossible* goals (logged collapse: Bob SR 29 % → 6.5 %).
- **The GoalEncoder is exonerated:** ablating it (Model D) still gives ~0 %, isolating the failure to ASP dynamics, not architecture.
- **Diffusion Policy shows imitation's two weak spots — generalisation and orientation:** it copies the trained T-block (90 % pos-only) but fails the unseen disc (30 %) and the coupled pose (40 %); reward-driven RL optimises the objective itself, so it transfers (disc 100 %) and controls yaw (pos+rot 60 %).
- **For this task, the reward is the lever:** a fixed randomised goal distribution is already rich enough — automatic curriculum (ASP) is unnecessary *and* destabilising, and demonstrations (DP) under-perform reward.

**Speaker notes (~70 s):** Two things to take away. First, why self-play loses even though it reuses the winning reward. Single-agent PBRS trains against a fixed, filtered goal distribution — one stable task, a clean gradient, eighty percent. ASP replaces that with goals proposed by Alice, who is herself still learning, so the solver is chasing a moving target; it's a non-stationary two-player game that violates PPO's trust region, and the curriculum never settles at the frontier of the solver's ability. Our own logs show Alice swinging to impossible goals while the solver's success rate falls from twenty-nine to six percent. We even ablated the GoalEncoder to rule out the architecture — it still collapses, so it really is the self-play dynamics. The point for the thesis is that PBRS and ASP solve different problems: PBRS fixes a sparse, mis-shaped reward; ASP tries to fix *where goals come from*. On a one-arm primitive-push task with a single object, that second problem barely exists — random goals already cover it — so the curriculum machinery costs more than it gives.

Second, what Diffusion Policy adds. It's the state-of-the-art imitation method and the origin of Push-T, so beating it is meaningful. It's strong on the object it was shown — ninety percent on position — but it exposes imitation's two classic weaknesses: it doesn't generalise to the unseen disc, thirty versus our hundred, and it doesn't truly control the coupled orientation, forty versus our sixty, often leaving the block badly rotated. Reward-driven RL optimises the actual goal, so it both transfers to new geometry and learns to fix the angle. Put together: for this task the right reward on a plain agent is necessary and sufficient — automatic curriculum is neither, and demonstrations under-perform it.
