# Master Thesis Writing Plan

**Title (working):** Why Asymmetric Self-Play Fails at Contact-Rich Robotic Pushing:
A Regime Analysis of a Method That Succeeds Elsewhere
**Author:** Vlad Iftime (s3426394), MSc Artificial Intelligence, University of Groningen
**Root document:** `main.tex` (six chapters via `\include`, plus appendices)

> This plan is the single reference for writing the thesis. It fixes the chapter
> structure, the content and literature of each section, the figures and tables to
> place, the scientific-integrity guardrails, and the house style. The existing
> chapter files hold earlier draft text; that text was written against a
> single-seed, single-batch result set and is to be treated as **provisional**.
> The final numbers come from the job-array campaign described in
> `implementations.md` §14, not from the earlier `SUMMARY.md`.

## 0. Central Thesis (the one sentence)

Asymmetric Self-Play (ASP) succeeds in the settings prior work used, and fails in
contact-rich robotic planar pushing under a single-Graphical-Processing-Unit
budget. This thesis isolates *why*, by identifying the axes on which the present
regime differs from the successful ones and testing each axis directly. The
single-agent models are the anchor that proves the task is solvable, so the ASP
failure is a genuine finding and not task-impossibility.

**Three contributions, in priority order:**
1. **Primary (ASP failure mechanism).** ASP learns the individual position and
   orientation sub-skills but does not compose them under the coupled success
   gate; the coupled multi-objective gate, not the reward or the encoder, is the
   dominant cause. Supported by the disc-vs-T-block contrast, the per-axis-vs-
   combined diagnostic, the value-bias measurement, the ABC clip-saturation
   measurement, and the demonstration-achievability (Alice-replay) diagnostic.
2. **Secondary (regime contrast).** A structured comparison of this regime against
   the ASP successes (Sukhbaatar 2018, Plappert 2021, and the adversarial-goal
   family) along five axes: goal coupling, reward formulation, imitation signal,
   goal representation, and compute scale. Each axis is both a literature claim
   and an experiment.
3. **Supporting (anchored task solvability).** Potential-Based Reward Shaping
    (PBRS) and the improvement-reward baseline both converge on this task,
    confirming the ASP underperformance is a genuine regime finding rather than
    evidence that the task is unsolvable. The single-agent results anchor the
    thesis and give the practical recommendation that a well-shaped dense reward
    is the first investment for this class of task.

**Sources of truth**
- Final numbers: the job-array campaign outputs collated by the seed collator
  (`implementations.md` §14); every reported figure traces there once the
  campaign completes. Until then, pending results are marked `\TBD` in red.
- Build history and campaign design: `implementations.md` (fixes P1–P82; §14).
- Citations: `literature/paper-async/categories.md` and `model_mapping.md`.
- Figures: `literature/paper-async/presentation/figures/`.

---

## 0.1 Chapter Map (matches `main.tex` `\include` order)

| # | File | Chapter |
|---|------|---------|
| 1 | `introduction.tex` | Introduction |
| 2 | `literature-review.tex` | Background & Related Work |
| 3 | `methods.tex` | Methods (task, MDP, reward, models) |
| 4 | `experiments.tex` | Experimental Setup & Evaluation Protocol |
| 5 | `results.tex` | Results |
| 6 | `conclusion.tex` | Discussion & Conclusion |
| A–L | `appendix.tex` | Appendices |

The Methods chapter holds the MDP, the reward, all model descriptions, and the
campaign design; the Experiments chapter holds the training configuration, the
validation protocol, and the metrics. Results and Discussion are split so the
ASP-failure analysis, which is the thesis's primary contribution, has room to
develop across the regime axes.

---

## 1. Writing Style & Punctuation Guide

### 1.1 Register
- Third person throughout. Past tense for work performed ("the agent was
  trained"); present tense for standing facts ("PBRS preserves the optimal
  policy"). No contractions.
- First-person plural ("we") only in the Contributions paragraph and the
  Conclusion, and used sparingly there.

### 1.2 Rule #1 — Consistency above everything
Any choice made once is applied identically across the entire document: model
names, metric names, capitalisation, hyphenation, unit spacing, citation style,
table column order, and decimal precision. When two valid spellings exist, pick
one and never alternate (always "self-play", never "selfplay"; always "success
rate", never "success-rate"). A short internal style ledger is kept while
writing to enforce this.

### 1.3 Abbreviations
- Spell out in full at first use with the acronym in parentheses, then use the
  acronym consistently thereafter: Graphical Processing Unit (GPU),
  Reinforcement Learning (RL), Deep Reinforcement Learning (DRL), Proximal
  Policy Optimization (PPO), Generalized Advantage Estimation (GAE),
  Potential-Based Reward Shaping (PBRS), Asymmetric Self-Play (ASP), Time-based
  ASP (TASP), Behavioural Cloning (BC), Alice Behavioural Cloning (ABC), Long
  Short-Term Memory (LSTM), Inverse Kinematics (IK), Success Rate (SR),
  Special Euclidean group SE(2), Markov Decision Process (MDP).
- Define once per document. The Abstract is self-contained: acronyms defined
  there are defined again at first use in the body.
- Maintain a running acronym list while drafting to guarantee no term is
  redefined or left unexpanded.

### 1.4 Punctuation
- Prefer colons (elaboration and lists), semicolons (tightly linked independent
  clauses), and commas (parenthetical asides) over the deck's heavy em-dashes.
- Units: `0.05\,\text{m}`, `0.2\,\text{rad}`, `80.0\%`, `4$\times$`;
  non-breaking thin space before units; never begin a sentence with a numeral.
- Numbers with four or more digits use a thousands separator (`2{,}048`).

### 1.4a Diction — precise, literal verbs over colloquial synonyms
Academic prose values a plain, exact word used consistently over a colloquial or
figurative synonym reached for the sake of variety. As a hard rule, colloquial or
metaphorical word choices appear **at most once in ten** sentences; the default is
the direct, literal term. Concretely:
- Do not write that a method "buys" efficiency, that a reward "yields" a return,
  that a curriculum "outpaces" the learner, or that a bug "bites"; write that a
  method "reduces" the required parallelism, that a reward "produces" a return,
  that the curriculum proposes goals "faster than the learner masters them", and
  so on.
- Avoid figurative verbs and idioms in expository text: "lock onto", "close the
  gap", "at odds with", "double-edged", "ready-made", "has room to breathe",
  "stresses any method", "sharpens the comparison", "plagues", "bootstraps"
  (unless used as a defined technical term). Replace each with a literal
  statement of what happens.
- Prefer one fixed verb for a recurring relationship rather than a rotating set
  of near-synonyms. For example, use "reduces" consistently for a quantitative
  decrease rather than alternating among "cuts", "shrinks", "trims", "lowers".
- This rule concerns \emph{diction only}; it does not override the
  paragraph-variety rule in Section~1.5, which governs paragraph \emph{structure}.
  Structural variety is desired; lexical variety-for-its-own-sake is not.

### 1.4b Diction — no code identifiers in prose
Never write source-code identifiers in the body text, and in particular never a
name containing an underscore (for example `train_e_pbrs_asp_dpose.py`,
`compute_pbrs_reward`, `dpose_threshold`, `Metrics/Bob/ValueError`,
`char_length`, `num_envs`). Refer to every quantity, file, or flag by a
descriptive English phrase instead: "the time-based self-play training script",
"the potential-based reward function", "the success-distance threshold", "the
mean absolute value error logged during training", "the characteristic length",
"the number of parallel environments". Mathematical symbols already defined in
the text (for example $k_p$, $w$, $d_{\text{pose}}$, $\Phi$) are permitted;
raw variable or file names are not. Where a specific flag must be named for
reproducibility, describe its effect in words and, if unavoidable, relegate the
literal token to a footnote or an appendix listing, never the running prose.

### 1.4c Sentence length and rhythm
Aim for a mean sentence length of about 18 words with a wide spread, a standard
deviation near 8.7, so that short and long sentences alternate. Rhythm variety
matters more than any hard ceiling: a paragraph of uniform mid-length sentences
reads worse than one that mixes a six-word sentence with a thirty-word one. Long
sentences are therefore welcome where they carry a single connected idea; a
sentence may run past 25 words when splitting it would break the thought, but
avoid a run of several long sentences in a row. Treat 35 words as a soft upper
guideline, and split anything longer unless it is a genuinely unavoidable list.
After every rewrite of a chapter, run the sentence-length checker
(`tools/style_check.py`) on the affected file. Confirm the mean sits near 18, the
standard deviation is high enough to show variety (target ~8.7, at least ~7), and
no sentence is gratuitously long. The checker also reports the total word count
and the number of unique words per file, tracked to keep vocabulary varied
without padding.

### 1.5 Paragraph variety (explicit anti-uniformity rule)
Do **not** reuse a single paragraph template; that produces predictable,
machine-sounding prose. Rotate deliberately among these paragraph types:
- *Claim* — asserts a finding, then substantiates it.
- *Literature-support* — synthesises what prior work established, no new claim.
- *Elaboration* — extends the previous paragraph with an example, corner case,
  or a second angle on the same point.
- *Comparative* — sets two models, regimes, or papers side by side.
- *Mechanistic* — walks through a causal chain (e.g. why the value-function bias
  survives under ASP).
- *Transitional* — short, bridges one theme to the next.

Vary sentence length within paragraphs. Avoid a fixed "topic sentence → three
supports → implication" cadence on every paragraph.

### 1.6 Cross-referencing
Use `\ref`/`\autoref` liberally to avoid repetition: forward-reference results
from Methods, back-reference the MDP from Results, and point to appendices for
derivations, bug post-mortems, and confound analyses. Every figure and table is
referenced in the prose at least once before it appears.

### 1.7 Integrity guardrails
- No "disproves Plappert/Sukhbaatar". ASP *succeeds* in prior work, including
  robotic manipulation (Plappert 2021). Every negative result is scoped to this
  regime: "in this contact-rich, IK-gated, multi-objective SE(2) pushing task,
  under a single-GPU budget, with discrete macro-actions". The contribution is
  the regime contrast, not a refutation.
- State the sample size and protocol at every quantitative claim: 30 held-out
  scenes, best-of-20 stochastically-sampled trials, up to 30 pushes, combined
  gate `pos < 0.05\,m` AND `rot < 0.2\,rad`, and the seed count (mean ± 95\,\%
  confidence interval).
- Both validators sample actions from the policy (not arg-max), so best-of-20 is
  20 genuine draws for every model; single-agent and self-play are compared under
  one protocol.
- Canonical ASP is ABC-on ($\beta = 0.5$); the no-ABC variant is an ablation, not
  the headline model.
- Distinguish scene SR from trial (per-attempt) SR when a headline number appears.
- ASP differs from single-agent on several axes at once; state which single-axis
  comparisons isolate a factor (disc vs T-block; outcome vs time-based Alice;
  ABC on vs off; goal encoder on vs off) and which do not.
- Frame the improvement reward's failure as gradient starvation and near-zero
  variance, not as non-Markovianity.
- Pending results are written as `\TBD` (red) until a seed-mean is available; no
  placeholder number is stated as if measured. Headline comparisons report the
  number of seeds behind every mean: 3--5 seeds for models with confidence
  intervals; 2 seeds noted as an informed estimate with a stated limitation; 1
  seed treated as qualitative pilot evidence. All reported numbers carry the
  seed count, the CI (where computed), and the validation protocol.

---

## 2. The Experimental Campaign (source of the final results)

The final results come from the job-array campaign in `implementations.md` §14,
re-scoped here for the ASP-failure thesis. Every training run is budget-matched:
total experience is held at about 23.8 million pushes, with the iteration count
set inversely proportional to the number of parallel environments, so the
per-update batch size is the only variable that changes across a sweep. Both
validators sample actions; canonical ASP is ABC-on.

### 2.0 Model roster (already implemented)

| Tag | What it is | Role in this thesis |
|---|---|---|
| PPO-Baseline | single agent, improvement reward | anchor + efficiency baseline |
| PBRS (A) | single agent, potential-based reward | solvable anchor; efficiency star |
| Curriculum (B) | single agent, PBRS + pos→rot staging | RQ2 bridge (structure vs reward) |
| ASP-dPose (E) | self-play, outcome Alice, SE(2) reward | primary ASP-failure subject |
| TASP-dPose (G) | self-play, time-based Alice | primary ASP-failure subject |
| ASP-disc (F) | self-play, disc, position-only | coupled-gate isolation + ABC control |
| TASP-disc (H) | self-play, time-based, disc | coupled-gate isolation |
| Bob-penalty (I) | self-play, full symmetric reward | reward-structure evidence |
| gym A/B/C | CPU cross-environment counterparts | regime/robustness check |

### 2.1 Campaign phases (re-scoped to the ASP-failure thesis)

| Phase | Question it answers | Design | Seeds |
|---|---|---|---|
| **Anchor + efficiency** | Is the task solvable, and how few envs does PBRS need? | PPO-Baseline and PBRS across {256, 512, 1024, 2048} envs | 3 |
| **Seed CIs** | What is each model's success with confidence intervals? | Baseline, PBRS, Curriculum, E, G at 528 | 5 |
| **Coupled-gate isolation** | Does ASP succeed when the gate is position-only? | disc F and disc-TASP H at 528, vs T-block E/G | 5 |
| **ASP scale** | Does more compute rescue ASP? | E and G across {256, 512, 1024, 2048} envs | 3 |
| **Component ablations** | Which ASP component matters? | E and G with ABC off, and with goal encoder off | 3 |
| **ABC positive control** | Does ABC help when the task is easy? | disc F with ABC on *and* ABC off | 3 |
| **Reward-structure** | Does the symmetric reward collapse? | Bob-penalty I at 528 | 5 |
| **Cross-environment** | Does the ordering hold in a second simulator? | gym A/B/C across two batch sizes | 3 |
| **Reward sensitivity** | Is the PBRS reward robust to its parameters? | PBRS over a $k_p \times w$ grid (appendix) | 1 |

### 2.2 The regime-contrast axes (the "why others succeeded" backbone)

Each axis is a hypothesis for the ASP failure and a claim about why prior work
succeeded. The literature review argues the axis; the campaign tests it.

| Axis | ASP successes (prior work) | This regime | Isolating experiment |
|---|---|---|---|
| Goal coupling | single reachable state; often position-dominant (Sukhbaatar 2018) | strict coupled pos∧rot SE(2) gate | disc F/H vs T-block E/G |
| Reward formulation | time-based (Sukhbaatar) or outcome (Plappert) | both evaluated | E (outcome) vs G (time-based) |
| Imitation signal | ABC on smooth continuous demos (Plappert 2021) | ABC on discrete macro-action demos | E/G with ABC on vs off + clip logging |
| Goal representation | learned goal embedding (Sukhbaatar 2018) | 8-D GoalEncoder latent | E/G with encoder on vs off |
| Compute scale | distributed (Plappert); months (Berner 2019) | single GPU, ~1.6 days | the ASP scale sweep |
| Symmetric reward | full $R_A/R_B$ (Sukhbaatar) | Bob time-penalty | Bob-penalty I vs G |
| Demo achievability | guaranteed by construction; reversible/repeatable | PhysX contact non-repeatable | Alice-replay diagnostic (§2.3) |

### 2.3 New diagnostic — demonstration achievability (Alice-replay)

ASP's theoretical guarantee is that Alice's own trajectory is a valid solution to
the goal she sets, so every goal is achievable by construction (Sukhbaatar 2018,
Plappert 2021). In contact-rich pushing with discrete macro-actions and
stochastic PhysX contact, replaying Alice's exact action sequence from Bob's reset
does not reproduce Alice's outcome. The diagnostic measures the **Alice-replay
success rate**: how often replaying the goal-setting agent's action sequence, from
the goal-solving agent's start state, actually reaches the proposed goal. A low
replay rate shows the ASP premise itself breaks in this domain, which is a
failure at the level of the assumption rather than the outcome. This is computable
offline from saved checkpoints and needs no new training.

### 2.4 Mandatory code fixes before the campaign is valid
- Both validators sample actions (the self-play validator currently arg-maxes);
  otherwise the single-agent-vs-ASP comparison runs under two protocols.
- Canonical ASP training runs with ABC on; the no-ABC flag is used only for the
  ablation arm.
- The disc positive control gains an ABC-off arm so it has a counterfactual.
- Log gradient-update count per configuration, and state the batch-size vs
  update-count confound wherever the scale sweeps are reported.

---

## 3. Figure & Table Inventory (asset → chapter)

Figures live in `literature/paper-async/presentation/figures/`; copy needed
files into the thesis `images/` directory keeping original filenames. Existing
figures below were generated from the earlier single-seed run set and will be
**regenerated from the campaign outputs** (seed means with confidence intervals)
before final submission.

### 3.1 New figures required by the ASP-failure thesis (generate from campaign)

These are the load-bearing figures of the reframed thesis. Each is produced from
the campaign outputs once they exist.

| Figure | Chapter | Purpose | Source |
|---|---|---|---|
| Disc vs T-block SR (position-only gate vs coupled gate), per ASP variant, seed means + CI | 5 Results | The coupled-gate isolation: ASP on disc vs T-block | coupled-gate isolation phase |
| ASP scale curve: combined SR and value-error vs env count, E and G, 3 seeds + CI | 5 Results | "Does scale rescue ASP?" | ASP scale phase |
| Per-axis vs combined SR bars, ASP variants, seed means + CI | 5 Results | Skills learned individually, not composed | seed-CI + training logs |
| Alice-replay success rate (demo achievability) vs single-agent | 5 Results / 6 Discussion | ASP premise breaks under non-repeatable contact | Alice-replay diagnostic |
| ABC clip-fraction and BC-ratio over training, E and G | 5 Results / 6 Discussion | Imitation signal suppressed in discrete action space | ABC instrumentation |
| ABC on-vs-off SR, disc vs T-block | 5 Results | ABC helps when easy, not when coupled | positive control |
| Regime-contrast table (the five axes) | 2 Related Work | Why prior ASP succeeded, why this fails | literature + campaign |
| PBRS efficiency: SR vs env count, PBRS vs improvement reward, 3 seeds + CI | 5 Results | The 4×-fewer-environments result | anchor + efficiency phase |

### 3.2 Existing figures (regenerate with campaign data)

| Asset | Chapter | Role |
|---|---|---|
| `diagram_task_schematic.png` | 3 Methods | UR5e + table + T-block + goal setup |
| `diagram_limit_surface.png` | 3 Methods / App. A | Limit-surface coupling of translation and rotation |
| `asp_loop_diagram.png` (or `.pdf`) | 3 Methods | Alice↔Bob cycle |
| `a1_potential_landscape.png` | 3 Methods | Exponential potential shape |
| `a2_gradient_comparison.png` | 3 Methods | PBRS vs improvement-reward gradient |
| `a3_episode_simulation.png` | 3 Methods | Per-push reward, 5-push episode |
| `a4_cosine_distance.png` | 3 Methods | Cosine vs modular angular metric |
| `val_test_layout.png` / `val_test_layout_top10.png` | 4 Experimental Setup | 30-scene held-out suite |
| `isaac_vs_gym.png` | 4 Experimental Setup | Throughput vs PPO batch size |
| `val_multi_combined.png` | 5 Results | Cross-model overall + breakdown SR |
| `val_multi_error_bars.png` | 5 Results / App. G | Per-scene SR with 95% CIs |
| `asp_diagnostic_2panel.png` | 5 Results / 6 Discussion | Alice validity; Bob per-axis vs combined |
| `bob_skills_vs_combined.png` | 5 Results | Per-axis vs combined composition gap |
| `p1_sr_bars.png`, `p3_heatmap.png` | 5 Results | Isaac vs gym ordering; per-scene heatmap |
| per-model error plots (PBRS, Curriculum, E, G) | 5 Results | Per-scene position and orientation error |
| `rec_push_s11/s14/s21_key.png` | 5 Results | Success keyframes (videos referenced) |
| `cmp_Success_Rate.png`, `cmp_Value_Loss.png`, `cmp_Mean_Reward.png`, `cmp_Policy_Loss.png` | App. B | Training dynamics |
| `modelB_curriculum_stuck.png` | App. | Curriculum-trigger bug |
| `asp_sparse_starvation.png`, `asp_dead_gradient.png` | 6 Discussion / App. | Reward-starvation diagnostics |
| `plot3_failure_taxonomy.png` | 5 Results / App. | Position × rotation success rectangle |

---

## 4. Per-Chapter Plans

### Chapter 1 — Introduction (`introduction.tex`, revise)
- **1.1 Motivation.** Service robotics and clutter manipulation (bin picking);
  pushing as a foundational pre-grasp skill; the appeal of automatic curricula
  and self-play for learning such skills without hand-designed goal
  distributions. Keep the existing service-robotics opening.
- **1.2 The puzzle.** ASP is a proven method, including in robotic manipulation
  (Plappert 2021). Yet in this contact-rich multi-objective pushing task it fails,
  while a plain single-agent shaped policy succeeds. The thesis asks why.
- **1.3 Research questions.**
  - RQ1: Does ASP, as applied successfully in prior robotic manipulation, succeed
    in this contact-rich multi-objective pushing task under a single-GPU budget?
  - RQ2: Which regime difference from the ASP successes (goal coupling, reward
    formulation, imitation signal, goal representation, or compute scale) causes
    the underperformance?
  - RQ3: What is the underlying mechanism — where, in the learning dynamics, does
    ASP break?
- **1.4 Contributions.** State the three from Section 0 (ASP-failure mechanism;
  regime contrast; anchored task solvability). Preview the disc-vs-T-block
  isolation and the Alice-replay diagnostic as the two most distinctive results.
- **1.5 Thesis outline.**
- Guardrail: ASP succeeds elsewhere; the contribution is the regime analysis, not
  a refutation.

### Chapter 2 — Background & Related Work (`literature-review.tex`, revise)
Keep the RL/PPO/SAC exposition and the reward-shaping, pushing-mechanics, and
SE(2) sections. The self-play section becomes the centre of gravity.
- **2.1 RL and DRL foundations** (existing).
- **2.2 Potential-based reward shaping** (existing): Ng 1999, Grześ 2017,
  Grześ & Kudenko 2009, Devlin & Kudenko 2012, Harutyunyan 2015.
- **2.3 Curriculum learning** (existing, trimmed): Narvekar 2020, Portelas 2020,
  Florensa 2017, Luo 2020.
- **2.4 Asymmetric self-play and adversarial-goal generation (expanded).** This is
  the backbone. For each of the six core papers, state the setting it succeeded
  in and the mechanism it relied on: Sukhbaatar 2018 (time-based reward, reversible
  gridworld, single reachable state); Plappert 2021 (outcome reward, ABC on
  continuous demos, distributed compute, robotic manipulation success); Dennis
  2020 PAIRED, Campero 2021 AMIGo, Durugkar 2021 AIM (adversarial-goal family,
  regret/teacher-student); Berner 2019 (self-play succeeds at very large scale).
- **2.5 The regime-contrast table.** The five-axis comparison from Section 2.2,
  presented as a table with prose: what each successful setting had that this one
  lacks. This is the literature deliverable that motivates every experiment.
- **2.6 Goal-conditioned and hierarchical RL** (trimmed to what is used): goal
  embeddings (Sukhbaatar 2018) for the GoalEncoder; HER (Andrychowicz 2017);
  information bottleneck (Tishby 2015, Goyal 2019) for the latent.
- **2.7 Behavioural cloning and imitation** (for ABC): Torabi 2018, Florence 2022,
  Hester 2018.
- **2.8 Mechanics and geometry of planar pushing** (existing): Mason 1986,
  Goyal 1991, Lynch & Mason 1996, Howe & Cutkosky 1996, Yu 2016, Stüber 2020;
  Park 1995, Urain 2023; equivariance (van der Pol 2020, Huang 2022, Nguyen 2024).
- **2.9 Robotic manipulation context** (brief): Mohammed 2022, Zeng 2018,
  push-grasp synergy (Kasaei 2024, Mokhtar 2022).

### Chapter 3 — Methods (`methods.tex`, revise)
Keep the existing environment, MDP, reward, SE(2), and model sections. Add the
campaign, validation, and instrumentation subsections (already drafted); update
the model list so ASP-dPose is ABC-on canonical.
- **3.1 Simulation environment** (existing).
- **3.2 Planar-pushing MDP** (existing).
- **3.3 Why the improvement reward fails** (existing).
- **3.4 Potential-based reward shaping** (existing).
- **3.5 SE(2) unified distance** (existing).
- **3.6 Models.** Update ASP-dPose to ABC-on; note the disc variants (F, H) and
  the Bob-penalty variant (I) as first-class subjects, not appendix items.
- **3.7 Why PBRS assists but does not solve ASP** (existing GAE decomposition;
  framed as the hypothesis the instrumentation tests).
- **3.8 Experimental campaign overview** (existing; update to the re-scoped phases
  of Section 2.1, ABC-on canonical, both validators sampling).
- **3.9 Validation protocol** (existing; state both validators sample).
- **3.10 Training instrumentation** (existing value-error + ABC-clip logging; add
  the Alice-replay diagnostic of Section 2.3).

### Chapter 4 — Experimental Setup (`experiments.tex`, revise)
- **4.1 Tools and infrastructure** (existing).
- **4.2 Training configuration and the campaign.** Per-phase tables from
  Section 2.1: models, environment counts, seeds, iteration counts, job counts.
  State budget matching and the batch-vs-update confound.
- **4.3 Validation protocol** (existing; both validators sample; 30 scenes,
  best-of-20, combined gate; seed-mean and CI reporting).
- **4.4 Cross-environment protocol** (existing).
- **4.5 Off-policy baseline (SAC + HER).** Report honestly: either converged, or
  demoted to "attempted"; use published planar-pushing numbers for calibration.
- **4.6 Metrics.** Scene SR, trial SR, per-axis SR, combined SR, value error,
  ABC clip-fraction and BC-ratio, Alice-replay rate, average pushes.

### Chapter 5 — Results (`results.tex`, full rewrite, reorganised around ASP failure)
- **5.1 The task is solvable (anchor).** Single-agent PBRS reaches its success
  level with confidence intervals; PPO-Baseline matches it at more environments.
  Establishes that ASP's failure is not task-impossibility. `\TBD` until campaign.
- **5.2 RQ (supporting): reward efficiency.** PBRS vs improvement reward across
  {256…2048} envs; the roughly-4×-fewer-environments result with CIs.
- **5.3 RQ1: ASP fails here.** E and G scene SR with CIs, far below single-agent,
  under the unified sampling protocol. State the gap and its significance.
- **5.4 RQ2 axis 1 — goal coupling (the key result).** disc F/H (position-only
  gate) vs T-block E/G (coupled gate). If ASP succeeds on the disc and fails on
  the T-block, the coupled multi-objective gate is isolated as the dominant cause.
- **5.5 RQ2 axis 2 — reward formulation.** E (outcome) vs G (time-based).
- **5.6 RQ2 axis 3 — imitation.** ABC on vs off (E and G); the positive control
  (disc ABC on vs off) showing ABC helps when the task is easy.
- **5.7 RQ2 axis 4 — goal representation.** Goal encoder on vs off (E and G).
- **5.8 RQ2 axis 5 — compute scale.** ASP scale sweep; combined SR flat (or not)
  as environments grow; value error vs env count.
- **5.9 RQ3: the mechanism.** Per-axis vs combined SR (skills learned, not
  composed); value-bias measurement under the shifting goal distribution;
  ABC clip-saturation measurement; the Alice-replay diagnostic (the ASP premise
  breaks under non-repeatable contact).
- **5.10 Cross-environment confirmation.** gym A/B/C; the ordering holds; the
  batch-regime nuance for curriculum.
- Guardrail: every number carries seeds + protocol; `\TBD` where pending.

### Chapter 6 — Discussion & Conclusion (`conclusion.tex`, revise)
- **6.1 Synthesis: why ASP fails here and succeeds elsewhere.** Walk the five
  axes, concluding which one(s) the data implicates most strongly (expected:
  coupled gate + non-repeatable demonstrations, with compute ruled out by the
  scale sweep).
- **6.2 Relation to prior work** (existing, updated): Ng confirmed; Plappert's ABC
  clip mechanism explains the discrete-action failure; Sukhbaatar's reversibility
  assumption broken by contact; Berner's scale story tested and ruled out here.
- **6.3 Contributions revisited**, mapped to the three of Section 0.
- **6.4 Limitations.** Best-of-20 inflates scene SR; ASP differs on several axes
  though single-axis isolations are provided; PBRS parameter sensitivity is a
  coarse single-seed grid; SAC+HER status; sim-only.
- **6.5 Future work.** Slower-changing goal proposer to reduce value bias;
  unclipped or continuous-parameter ABC; relaxed/partial-credit gate; zero-sum
  stabilisation of the symmetric reward; sim-to-real; and the push-grasp synergy
  application that motivates the task.

### Appendices (`appendix.tex`, revise)
- **A** Push mechanics: limit surface and characteristic length.
- **B** Full training curves for all models (seed means).
- **C** PBRS parameter sensitivity ($k_p \times w$ grid).
- **D** Validation scene descriptions (30 scenes; disc mirror).
- **E** cuRobo IK pipeline and workspace bounds.
- **F** Observation-space specifications (all variants).
- **G** Per-scene error bars with confidence intervals.
- **H** Curriculum-trigger correction.
- **I** Bob-penalty fail-fast equilibrium (reward-design lesson).
- **J** Regime-contrast disaggregation and the axes ASP varies on at once.
- **K** Training-budget and update-count table (batch-vs-update confound).
- **L** Alice-replay diagnostic: method and full results.
- **M** SAC + HER configuration and status.

---

## 6. Data Analysis

This section describes what data to extract from the completed training runs,
what to look for in each output, what figures to generate, and how the headline
results table should be laid out.

### 6.1 Data sources

| Source | Format | Location | Content |
|---|---|---|---|
| Validation CSV | one CSV per run dir | `final_results_thesis/<exp>/validation_results_*.csv` | per-scene: `test_index, test_type, success (0/1), pushes_used, pos_err, rot_err` (30 scenes × 20 trials) |
| TensorBoard event files | `.tfevents` | `final_results_thesis/<exp>/summary/` | training curves: `Metrics/Bob/CombinedSR, PositionSR, RotationSR, ValueError, ABC/ClipFraction, Diagnostics/GradientUpdates` |
| Alice-replay CSV | to generate offline | via `tests/alice_replay.py` on each saved checkpoint | per-replay: `success, pushes, pos_err, rot_err, stop_reason` — measures whether Alice's own actions reach her proposed goals |
| Collated summary | CSV | produced by `extras/collate_seeds.py` | per-model mean ± 95 % CI for overall scene SR, per-test-type SR, per-difficulty SR |

### 6.2 TB tag availability by model

The TensorBoard tags differ between single-agent and self-play scripts, and
between the d_pose (E, G, I) and legacy (C, D) variants. The cross-model
comparison relies on the following shared tags:

| Tag | base/pbrsA | pbrsE | pbrsG | discF | taspI | Purpose |
|---|---|---|---|---|---|---|
| `Metrics/SuccessRate` | yes | — | — | — | — | single-agent combined SR |
| `Metrics/Bob/CombinedSR` | — | yes | —* | — | yes | self-play combined-gate SR (d_pose only) |
| `Metrics/Bob/PositionSR` | — | yes | yes | yes | yes | per-axis position SR |
| `Metrics/Bob/RotationSR` | — | yes | yes | yes | yes | per-axis orientation SR |
| `Metrics/Bob/ValueError` | pbrsA only | yes | —* | — | — | value-bias measurement |
| `ABC/ClipFraction` | — | yes* | —* | yes* | —* | imitation-signal suppression |
| `Diagnostics/GradientUpdates` | yes | yes | —* | — | — | update-count confound |

\* Conditionally logged (only when ABC is enabled, or only in the d_pose scripts).

### 6.3 Headline results table

A single table showing every model with its environment count, number of seeds,
and scene success rate with a confidence interval, plus a column flagging the
research question or comparison each row serves.

| Model | Envs | Seeds | Scene SR ± CI | Trial SR | Pos SR | Rot SR | Answers |
|---|---|---|---|---|---|---|---|
| PPO-Baseline (base) | 528 | 3 | \TBD | \TBD | — | — | RQ1 anchor |
| PPO-Baseline (base) | 256 | 2 | \TBD* | \TBD | — | — | RQ1 anchor |
| PPO-PBRS (pbrsA) | 1024 | 3 | \TBD | \TBD | — | — | RQ1 anchor |
| PPO-Curriculum (pbrsB) | 1024 | 2 | \TBD* | \TBD | — | — | RQ1 anchor |
| ASP-dPose (pbrsE) | 528 | 5 | \TBD | \TBD | \TBD | \TBD | RQ1 primary subject |
| TASP-dPose (pbrsG) | 528 | 5 | \TBD | \TBD | \TBD | \TBD | RQ1, RQ2 reward |
| ASP-disc (discF) | 528 | 5 | \TBD | \TBD | — | — | RQ2 coupling isolation |
| Bob-penalty (taspI) | 528 | 5 | \TBD | \TBD | \TBD | \TBD | RQ2 reward structure |
| E — ABC off | 528 | 3 | \TBD | \TBD | — | — | RQ2 imitation |
| G — ABC off | 528 | 3 | \TBD | \TBD | — | — | RQ2 imitation |
| E — encoder off | 528 | 3 | \TBD | \TBD | — | — | RQ2 goal representation |
| G — encoder off | 528 | 3 | \TBD | \TBD | — | — | RQ2 goal representation |
| E — scale | 256, 2 048 | 3 | \TBD | \TBD | — | — | RQ2 scale |
| G — scale | 256, 2 048 | 3 | \TBD | \TBD | — | — | RQ2 scale |
| gym A (PBRS) | 64 | 1 | \TBD* | \TBD | — | — | cross-environment |
| gym B (Curriculum) | 64 | 1 | \TBD* | \TBD | — | — | cross-environment |
| gym C (ASP) | 64 | 1 | \TBD* | \TBD | — | — | cross-environment |

Rows marked \TBD* carry a footnote that fewer than three seeds were available
(2 seeds for an informed estimate; 1 seed for qualitative evidence). Per-axis
SR for ASP models is the training-final value from TensorBoard (last iteration
of the `Metrics/Bob/PositionSR` / `Metrics/Bob/RotationSR` curve, averaged
across seeds). Single-agent models log a single `Metrics/SuccessRate`, not per-
axis. The disc models (discF) lack rotation entirely, so only the overall SR is
relevant.

### 6.4 Figure inventory

All figures are generated with a new standalone script `tools/plot_results.py`,
following the conventions of `tools/plot_push_primitive.py` (matplotlib, Agg
backend, ColorBrewer palette, DPI 150, PDF + PNG output to `images/`).

**Figure 1 — head-to-head scene SR with CIs (RQ1).**
Grouped bar chart, one bar per headline model (base@528, pbrsA@1024, pbrsB@1024,
pbrsE@528, pbrsG@528, discF@528, taspI@528) with CI whiskers. The single-agent
anchors cluster at the top; E and G sit far below; discF sits near the anchors.
Data source: `collated_summary.csv`.

**Figure 2 — coupling isolation (RQ2 axis 1, the flagship).**
Two-panel figure. Left: discF scene SR with 5-seed CI alongside E and G with
5-seed CI. Right: per-test-type SR breakdown (discF `pos_only` vs E/G
`pos_rot`), showing discF solves what E/G cannot. Data source: `collated_summary.csv`
per-test-type columns.

**Figure 3 — scale sweep (RQ2 axis 5).**
Line plot: E and G scene SR vs environment count (256, 528, 2 048) with 3-seed
CI error bars. Second panel: value error vs environment count (E only, since
pbrsG does not log `ValueError`). Data source: `collated_summary.csv` for SR;
TensorBoard event files for value error.

**Figure 4 — component ablations (RQ2 axes 3–4).**
Grouped bar chart: E canonical (ABC-on, 5-seed) vs E ABC-off (3-seed) vs E
encoder-off (3-seed), same for G. Inset bar chart: discF ABC-on vs ABC-off
(positive control). Data source: `collated_summary.csv` — collate_seeds.py
parses the `_noabc` and `_noge` name suffixes as separate model rows.

**Figure 5 — per-axis vs combined (RQ3).**
Grouped bar chart: for E and G, show PositionSR, RotationSR, and CombinedSR
(training-final values from TensorBoard, averaged across seeds). The gap between
the independence product (PositionSR × RotationSR) and the measured CombinedSR
is the composition gap — the mechanism of the underperformance. Data source:
TensorBoard event files (`Metrics/Bob/PositionSR`, `Metrics/Bob/RotationSR`,
`Metrics/Bob/CombinedSR`).

**Figure 6 — training-time diagnostics (RQ2 axis 3, RQ3).**
Multi-panel time-series plot from TensorBoard:
- (a) `ABC/ClipFraction` over training for E and G (ABC-on canonical only);
- (b) `Metrics/Bob/ValueError` over training, pbrsA (single-agent) vs E
  (self-play) overlaid on one axis (pbrsG does not log this tag);
- (c) `Metrics/Bob/CombinedSR` (E and G) alongside `Metrics/SuccessRate`
  (base, pbrsA) over the full training-iteration axis.

Data source: TensorBoard event files.

**Figure 7 — Alice-replay (RQ3, RQ2 axis 7).**
Bar chart: replay success rate — the share of Alice's goal-proposing trajectories
that reach her own goal when replayed from Bob's start state. One bar for E,
one for G, one for discF. Data source: `tests/alice_replay.py` output CSVs,
aggregated across seeds. A low rate means the ASP premise (every goal achievable
by construction) is broken by stochastic PhysX contact. No new training
required — operates offline on saved checkpoints.

**Figure 8 — cross-environment (supplementary).**
Bar chart: gym A/B/C scene SR at envs 64, single-seed. Qualitative; confirms the
ordering (single-agent ≈ curriculum ≫ ASP) survives the second simulator. Data
source: `collated_summary.csv` gym rows.

### 6.5 Execution order

1. Run `extras/collate_seeds.py` on the done runs — get headline numbers.
2. Run `tests/alice_replay.py` on E, G, discF checkpoints — get replay rates
   (offline, no training needed).
3. After all running jobs finish (~20 h), re-run the collator on the full set.
4. Extract TensorBoard curves for per-axis-vs-combined, ABC clip-fraction, and
   value error using `tensorboard.backend.event_processing`.
5. Run `tools/plot_results.py` — reads `collated_summary.csv` and TB event
   files, outputs all eight figures to `images/`.
6. Write the Results chapter, slotting each figure into the corresponding
    section (Figure 1→§5.3 RQ1, Figure 2→§5.4 RQ2 axis 1, Figures 3–4→§5.5–5.8
    RQ2 axes 2–5, Figures 5–7→§5.9 RQ3 mechanism, Figure 8→§5.10 cross-env).

The earlier campaign-level execution order (code fixes, submission, collation,
drafting order, style checks) is superseded by the data-analysis pipeline above
and by the current chapter-by-chapter drafting sequence begun in Section 4.
The validation and collation steps are now covered by Section 6.5
items 1–3; the figure regeneration by items 4–5; the chapter drafting order
remains Methods → Experiments → Results → Introduction/Abstract →
Discussion/Conclusion → Appendices.
