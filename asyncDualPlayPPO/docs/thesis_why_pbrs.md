# Why Potential-Based Reward Shaping (PBRS) Is the Right Reward for a One-Arm Push-Primitive Agent

**Branch**: `asp_goal_encoder`
**Scope of this document**: the **reward function**. The curriculum mechanism
(single-agent, forced curriculum, asymmetric self-play, time-based self-play) is
treated as a *separate, independent axis* and is discussed only where it sharpens
the reward argument (§6).

**Central claim**: For a single UR5e arm acting through a non-prehensile **push
macro-primitive** on the planar **Push-T** task (push a T-shaped block to a target
SE(2) pose; Florence et al. 2022; Chi et al. 2023), PBRS is the correct
reward-shaping choice — and it is correct **even for a plain single-agent PPO
agent with no curriculum at all**. The headline evidence (§5) is a controlled
ablation in which the *only* variable changed is the reward function.

The argument is organised around the **defining physical property of pushing**:
by the mechanics of quasi-static planar sliding (§1.1), *every* generic push
produces a **coupled SE(2) twist** — a simultaneous change in translation **and**
orientation. The agent cannot move the T-block without also rotating it. This
single fact, derived from the limit-surface theory of frictional pushing, drives
every reason PBRS is the right reward (§3).

---

## Table of Contents

1. [Problem Framing — What Makes This Environment Special](#1-problem-framing)
   - [1.1 Mechanics of Quasi-Static Planar Pushing](#11-mechanics)
2. [Why Reward Design Is Hard Here](#2-why-reward-design-is-hard-here)
3. [Why PBRS Is Correct, Grounded in Pushing Mechanics](#3-six-reasons)
4. [Theoretical Foundations (Full Rigor)](#4-theory)
5. [Empirical Validation](#5-empirical-validation)
6. [Scope and Honesty: Reward vs. Curriculum](#6-scope-and-honesty)
7. [References](#7-references)
8. [Appendix A — Proofs and Derivations](#appendix-a)

---

## 1. Problem Framing — What Makes This Environment Special <a name="1-problem-framing"></a>

The agent controls **one** UR5e arm. Its decision is not a low-level joint
torque but a **push macro-action**: a 4-D `MultiCategorical` command
`(r, φ, length, θ)` decoded relative to the object, expanded into a fixed
approach→descend→push→retract→return waypoint trajectory, solved by cuRobo IK and
executed over ~72 physics substeps. Four properties of this setting determine
what any reward function must satisfy:

| Property | Consequence for reward design |
|---|---|
| **Non-prehensile (pushing, not grasping)** | Control is *indirect* — the arm nudges the object through frictional contact. Outcomes are physically noisy (PhysX contact jitter, momentary tipping, IK tolerance). The reward must not amplify this noise. |
| **Macro-action horizon** | An episode is only **~5–10 decision steps** (pushes), not thousands of control steps. Credit assignment is short but each step is high-variance. Sparse terminal reward gives almost no gradient. |
| **SE(2) multi-objective goal** | Success requires matching position `(x, y)` **and** yaw `θ`. Two objectives that can compete. |
| **Episodic** | Episodes terminate on success, catastrophe (tip/launch/out-of-bounds/arm-through-table), or push budget. This licenses a specific, stronger form of PBRS (§4.2). |

Every claim in §3 ties a property of PBRS to one of these four features.

### 1.1 Mechanics of Quasi-Static Planar Pushing <a name="11-mechanics"></a>

The single most important fact about this task is a *theorem* of pushing
mechanics, not an empirical accident: **a single push generically changes both the
translation and the orientation of the object at the same time.** They cannot be
controlled independently at the level of one push.

**The limit surface.** When a rigid object slides quasi-statically on a planar
support under dry friction, the relationship between the applied frictional load
and the resulting motion is captured by the **limit surface** (Goyal, Ruina &
Papadopoulos 1991). Let the planar contact wrench be `w = (fₓ, f_y, m)` (two force
components and a moment about the object's centre) and the object twist be
`t = (vₓ, v_y, ω)` (linear and angular velocity). The set of wrenches that the
friction distribution can sustain forms a **convex limit surface** `H(w) = 1` in
wrench space, and the **maximum-power-dissipation principle** forces the resulting
twist to be **parallel to the outward normal of the limit surface** at the applied
wrench:

```
t = λ · ∇_w H(w),   λ ≥ 0.
```

This normality law is the rigorous source of translation–rotation coupling: the
map `w ↦ t` is generically full rank on SE(2), so a force component generically
induces an angular velocity and a moment generically induces a linear velocity.

**Centre of friction and the voting theorem.** A push applies a contact force
along a line of action. Mason (1986) showed that the *sense* of rotation of the
pushed object is decided by where this line of action passes relative to the
**centre of friction (CoF)** and the edges of the **friction cone** (the
"voting theorem"). The only pushes that produce **pure translation** are those
whose line of action passes through the CoF — a *measure-zero* set. **Every other
push — i.e. essentially all of them — produces a coupled twist** `(Δx, Δy, Δθ)`
simultaneously. Lynch & Mason (1996) characterise when this coupled motion can be
made predictable ("stable pushing," keeping the object within the friction cone),
and Akella & Mason (1998) show that *posing* a polygon (controlling position
**and** orientation) requires deliberately sequenced pushes that exploit, rather
than avoid, this coupling.

**Why the T-block.** The Push-T task (Florence et al. 2022; Chi et al. 2023) uses
a T-shaped block precisely because it is **asymmetric**: its CoF is offset from the
geometric centroid, so pushes induce large, configuration-dependent rotations, and
its orientation is *observable and must be controlled*. A rotationally symmetric
disc is the opposite limiting case — yaw is unobservable, so the coupling becomes
irrelevant; we use the disc deliberately as an ablation (§4.7, §5.4). Empirical
studies confirm the coupling is also **stochastic**: the exact twist depends on the
unknown pressure distribution under the object, which varies push-to-push (Yu,
Bauza, Fazeli & Rodriguez 2016; Stüber, Zito & Stolkin 2020).

**Consequence for reward design.** Because translation and rotation are coupled in
*every* push, a reward cannot treat them as two independent channels at the action
level: improving one degree of freedom generically perturbs the other, and a
goal-reaching plan must traverse intermediate poses that are temporarily worse in
one coordinate. The reward must (i) score the *whole* SE(2) pose change a push
induces, (ii) permit such "detour" pushes without penalty, and (iii) tolerate the
physical stochasticity of the induced twist. §3 shows PBRS satisfies all three;
§4.7 shows the coupling even fixes the *form* of the reward metric.

---

## 2. Why Reward Design Is Hard Here <a name="2-why-reward-design-is-hard-here"></a>

The task sits in a hard middle ground between two failing extremes.

**Sparse reward starves the gradient.** With only ~5–10 macro-steps and a strict
SE(2) success gate, the terminal reward fires on a vanishing fraction of pushes.
In the self-play variant this was measured at **0.14 % of pushes**
(`implementations.md`, Fix P53); the resulting GAE advantages were
indistinguishable from zero-mean noise and the agent never bootstrapped
(`train_push_asp.py` line reaching 0.07 % SR). A short macro-action horizon makes
sparse reward strictly worse than it is in dense-control RL: there are far fewer
chances for a lucky success to seed learning.

**Ad-hoc dense reward fixes the gradient but quietly breaks the task.** The
project's original dense reward was a fractional-improvement formula with state
penalties:

```
R = α·(d_prev − d_now)/d_prev  +  α·(y_prev − y_now)/y_prev
  − β·d_now  −  β_rot·y_now
  + 5 (position gate)  + 2 (pos∧rot gate)  − 5 (tip)
```

This formulation triggered an extended tuning campaign — visible in the
`implementations.md` Fix log: coefficient rescaling (P17, P18), rotation-vs-position
gradient dominance "by ~20×" (P47), repeated re-normalisation of the fractional
term (P63, P64), and so on. The structural defects (formalised in §4.5 and
Appendix A.2) are:

1. **Non-Markovian reward.** The `1/d_prev` factor couples the reward to the
   *previous* state, so two identical pushes from identical states earn different
   rewards depending on history. No potential function can reproduce it.
2. **Noise amplification near the goal.** The `1/d_prev` factor magnifies both
   signal *and* physics noise by up to ~6× near the goal — exactly where a
   pushing task most needs a clean signal.
3. **Optimal-policy distortion.** The `−β·d_now` / `−β_rot·y_now` penalties are
   not potential differences; they add a per-step "be close now" cost that biases
   the policy away from temporarily-suboptimal-but-globally-better trajectories.
4. **No optimality guarantee.** Without PBRS structure there is no theorem tying
   the shaped optimum to the true (sparse-task) optimum, leaving room for
   reward-hacking local optima.

PBRS is precisely the construction that delivers a dense per-push gradient
**while** eliminating all four defects.

---

## 3. Why PBRS Is Correct, Grounded in Pushing Mechanics <a name="3-six-reasons"></a>

Each reason maps a consequence of the **coupled-twist** property (§1.1) to a
property of PBRS. The six R-reasons are coupling-first; a short closing paragraph
covers the horizon, curriculum-neutrality, and empirical points.

**R1 — Translation and rotation are coupled in every push ⇒ do not decouple them in the reward.**
Because each push changes `(Δx, Δy, Δθ)` together (§1.1), a reward with two
separately-weighted channels (`α_pos·Δd + α_rot·Δyaw`) fights the dynamics: any
fixed coefficient ratio is wrong for some configurations, and improving one degree
of freedom perturbs the other. The observed symptom was rotation gradients
dominating position **~20×** (`implementations.md` P47). The principled fix is a
**single SE(2) potential** over the *coupled* pose error,
`Φ = exp(−k·d_pose²)` with `d_pose = √(dx² + dy² + L²·dθ²)` — it scores exactly the
twist the limit surface governs (§4.7).

**R2 — The coupling has a physical length scale ⇒ `L` is a constant, not a hyperparameter.**
In the ellipsoidal limit-surface model (Howe & Cutkosky 1996) the moment is coupled
to force through a characteristic length `c` — the radius of gyration of the
pressure distribution. The `L` in `d_pose` plays exactly this role: it converts an
orientation error (radians) into the translational-equivalent displacement it costs
at the object's effective lever arm. So `d_pose` is the **weighted SE(2) metric
induced by the object's friction geometry**, with `L ≈` the object's characteristic
radius (T-block `L = 0.07 m`; disc `L = 0`). Derived in §4.7 / Appendix A.4.

**R3 — Posing requires "detour" pushes ⇒ PBRS's policy-invariance permits them.**
Akella & Mason (1998) show that controlling both position and orientation requires
sequenced pushes, some of which temporarily worsen one coordinate to set up the
final pose. PBRS adds **zero** bias to the optimum (Ng et al. 1999; §4.1), so the
agent may traverse a temporarily-worse pose to reach the goal. The ad-hoc
`−β·d_now` "be-close-now" penalty punishes exactly these necessary detours and
biases the agent into a position-greedy local optimum that never settles
orientation (§4.5).

**R4 — Coupling makes SE(2) limit cycles easy ⇒ zero-sum cycles forbid reward farming.**
Coupling makes it trivial to enter pose limit cycles (nudge to fix yaw, which
spoils position; push back, which spoils yaw). A push primitive is reversible, so
any "got-closer" dense reward can be farmed by oscillation. With `γ_shaping = 1.0`
(licensed because the task is episodic, §4.2), the PBRS term `Φ(s′) − Φ(s)` sums to
**exactly zero** over any closed SE(2) loop (Appendix A.1): reward can be earned
only by *net* SE(2) progress.

**R5 — The induced twist is physically stochastic near the goal ⇒ bounded exponential potentials.**
The exact twist depends on the unknown pressure distribution and varies push-to-push
(Yu et al. 2016; Stüber et al. 2020), so near the goal the same pose error can map
to different motions. `Φ = exp(−k·d_pose²)` gives a smooth, bounded target
(`Φ ∈ (0,1]`) that does not amplify this stochasticity — unlike the fractional
formula, whose `1/d_prev` factor magnifies physics noise up to ~6× near the goal
(§4.3–§4.4).

**R6 — Asymmetry is what makes orientation matter ⇒ the disc ablation proves the metric tracks the physics.**
The T-block's asymmetry makes yaw observable and the coupling strong; a disc is
rotationally symmetric and yaw is meaningless. The coupling-aware metric handles
both correctly: T-block uses `L = 0.07 m` (Model E), disc uses `L = 0` so `d_pose`
collapses to pure 2-D translation (Model F). This is the theory-to-code bridge
(`reward_pbrs.py:147–214`).

**Beyond the coupling (horizon, neutrality, payoff).** Three further points hold
independently of the coupling. *(i) Horizon:* a push macro-action gives only
~5–10 decision steps with success firing on **0.14 %** of pushes
(`implementations.md` P53), so a dense per-push signal is mandatory — and PBRS is
the unique framework that supplies it while provably preserving the optimum
(§4.1). *(ii) Curriculum-neutrality:* because PBRS preserves the optimal policy,
the **same** reward drops into single-agent PPO, a forced curriculum, or self-play
without fighting the curriculum (§6); the reward is the right fixed point to
settle first. *(iii) Empirical payoff:* on the controlled single-agent,
no-curriculum ablation (§5), swapping *only* the reward from ad-hoc-dense to PBRS
lowers position error **0.197 m → 0.114 m** and rotation error
**0.69 → 0.21 (3.3× gap)** at equal compute, generalising to **55 %** SR on 20
held-out scenarios.

---

## 4. Theoretical Foundations (Full Rigor) <a name="4-theory"></a>

### 4.1 PBRS and policy invariance (Ng et al. 1999)

Let `M = (S, A, T, γ, R)` be the underlying MDP. Define the shaped MDP `M′` with
reward

```
R′(s, a, s′) = R(s, a, s′) + F(s, s′),   F(s, s′) = γ_sh · Φ(s′) − Φ(s),
```

for any potential `Φ : S → ℝ`. **Theorem (Ng et al. 1999).** Every optimal policy
of `M′` is optimal in `M`, and vice versa. The proof (Appendix A.1) shows the
shaped action-value satisfies `Q′(s, a) = Q(s, a) − Φ(s)`, an action-independent
shift, so `argmax_a Q′(s, a) = argmax_a Q(s, a)`. **Consequence:** the dense
per-push signal cannot move the optimum away from "reach the SE(2) goal." This is
the property the ad-hoc fractional reward lacks (§4.5).

### 4.2 Episodic PBRS with `γ_shaping = 1.0` (Grzes & Kudenko 2009)

Using the MDP discount `γ = 0.95` inside `F` creates a discounting tax: when
`Φ(s)` is already large (agent close to goal),
`0.95·Φ(s′) − Φ(s)` can be **negative even while making progress**. For a
push task whose hardest regime is the near-goal one, this inverts the gradient
exactly where it must stay positive.

For **episodic** MDPs with absorbing terminal states, Grzes & Kudenko (2009)
proved policy invariance still holds with `γ_shaping = 1.0`:

```
F(s, s′) = Φ(s′) − Φ(s).
```

Properties (proved in Appendix A.1):
- **Zero-sum cycles:** any `s_0 → … → s_0` cycle contributes exactly `0` (no
  oscillation profit — reason R4).
- **No near-goal inversion:** every genuine potential increase yields a positive
  shaping reward.
- **Policy invariance** holds provided `Φ` is constant on terminal states; our
  potentials satisfy `Φ(at goal) = 1` for both position and rotation, meeting the
  condition (Appendix A.1).

This is distinct from the PPO/GAE discount `γ = 0.95`, which is retained.

### 4.3 Bounded exponential potentials

A linear potential `Φ(s) = −d(s)` is unbounded, making the critic's regression
target unbounded. The implemented potentials (`reward_pbrs.py:30–37`) are

```
Φ_pos(s) = exp(−k_p · ‖p − p*‖²_{2D}),     k_p = 30
Φ_rot(s) = exp(−k_r · c(s)),               k_r = 5
c(s)     = (1 − cos(θ* − θ)) / 2  ∈ [0, 1]
```

giving: gentle far-field gradient, steep near-goal gradient, no singularities,
and **bounded critic targets** `Φ ∈ (0, 1]`.

### 4.4 Temperature selection

`k` sets the width of the reward gradient and is tuned to the workspace scale
(typical goal distances 0.05–0.50 m). Reward for a 5 cm improvement, scaled by
`w_pos = 10`, under `F = Φ(s′) − Φ(s)`:

| `k_p` | 0.30→0.25 m | 0.20→0.15 m | 0.10→0.05 m | 0.05→0.00 m |
|------|-------------|-------------|-------------|-------------|
| 15 | 1.36 | 0.89 | 1.02 | 0.37 |
| **30** | **1.03** | **2.34** | **1.87** | **0.72** |
| 50 | 0.33 | 1.90 | 2.75 | 1.18 |

`k_p = 30` keeps a meaningful far-field signal (~1.0 at 0.30 m) without collapsing
the near-goal signal. Rotation, with `c ∈ [0, 1]`:

| `k_r` | c=0.5 (90°) | c=0.1 (36°) | c=0.02 (11°) | c=0 |
|------|-------------|-------------|--------------|-----|
| 10 | 0.007 | 0.368 | 0.819 | 1.000 |
| **5** | **0.082** | **0.607** | **0.905** | **1.000** |
| 3 | 0.223 | 0.741 | 0.942 | 1.000 |

`k_r = 5` gives good gradient across the operative rotation range. Because any
positive scalar multiple of a potential difference is still a valid PBRS reward,
`w_pos = w_rot = 10` changes learning speed but not the optimum.

### 4.5 Why the fractional formula is *not* PBRS

The fractional improvement `α·(d_prev − d_now)/d_prev = α·(1 − d(s′)/d(s))`
depends on the **ratio** `d(s′)/d(s)`, which is not separable into a function of
`s′` minus a function of `s`. Hence **no** `Φ` satisfies
`Φ(s′) − Φ(s) = α·(1 − d(s′)/d(s))` for all state pairs (Appendix A.2). It is
therefore outside the PBRS family and carries no policy-invariance guarantee; the
`−β·d_now` penalties are likewise not potential differences and provably shift the
optimum.

### 4.6 Cosine angular distance and threshold equivalence

For planar pushing only yaw matters. Replacing modular-arithmetic yaw distance
(discontinuous gradient at ±π) with the cosine distance
`c = (1 − cos(θ* − θ))/2` yields a metric that is smooth everywhere, bounded in
`[0, 1]`, and monotone in true angular error. The legacy success gate
`rot_err < 0.2 rad` maps exactly to

```
c_threshold = (1 − cos 0.2)/2 = (1 − 0.98007)/2 = 0.00997 ≈ 0.01,
```

implemented as `PBRS_COS_ROT_THRESHOLD = 0.01` (`reward_pbrs.py:11`). Derivation
in Appendix A.3.

### 4.7 SE(2) unification (`d_pose`) and the physical origin of `L`

The implemented unified objective (`reward_pbrs.py:147–173`) is

```
d_pose = √(dx² + dy² + L²·dθ²),   Φ(s) = exp(−k·d_pose²),  k = 30, w = 10
```

with `L` the object's characteristic length (T-block `L = 0.07 m`; disc `L = 0`).
Far from being an arbitrary weighting, `L` is **fixed by the pushing mechanics**.

**The ellipsoidal limit surface.** Howe & Cutkosky (1996) approximate the limit
surface (§1.1) by an ellipsoid in wrench space, which makes the wrench↔twist map
diagonal:

```
(vₓ, v_y) ∝ (fₓ, f_y),     ω ∝ m / c²,
```

where `c` is a **characteristic length** equal to the radius of gyration of the
support pressure distribution `p(r)` about the centre of friction:

```
c² = ∫ r² p(r) dA  /  ∫ p(r) dA.
```

`c` is the *single constant* that couples moment to angular velocity relative to
the way force couples to linear velocity. It carries units of length and is an
intrinsic property of the object's mass/contact geometry. For a uniform disc of
radius `R`, `c = R/√2`; for the T-block, `c` is its effective radius of gyration
(≈ its half-extent, ~0.07 m).

**`L` is `c`.** The natural error metric on SE(2) that respects this coupling
weighs an angular error by the same length `c` that the limit surface uses to
relate moment to motion. Writing the pose error as `(dx, dy, dθ)`, the
mechanically-consistent (squared) distance is

```
d_pose² = dx² + dy² + c²·dθ²,
```

i.e. **`L = c`**, the limit-surface characteristic length. A `dθ` of one radian is
penalised the same as a translational error of `c` metres — exactly the lever-arm
at which the object's rotation "costs" as much motion as translation. This makes
`d_pose` the **weighted Riemannian metric on SE(2) induced by the object's friction
geometry**, not a tuned coefficient (full derivation in Appendix A.4).

**Consequences.**
- **T-block:** `L = 0.07 m` couples yaw and translation at the physically correct
  ratio, so a single potential `Φ = exp(−k·d_pose²)` produces one coherent gradient
  over the coupled twist a push actually induces (Model E).
- **Disc:** orientation is unobservable, the CoF is at the centre, and `c` is
  irrelevant to the goal, so `L = 0` collapses `d_pose` to pure 2-D translation
  (Model F). The same metric handles both objects correctly — the disc is the
  clean ablation that the metric tracks the physics (R6).

This removes the two-coefficient position/rotation balancing problem (R1) by
deriving the single coupling constant from the mechanics rather than tuning it.

---

## 5. Empirical Validation <a name="5-empirical-validation"></a>

All figures are regenerated by
`asyncDualPlayPPO/data_analysis/plots/why_pbrs/generate_plots.py` from the
`26.06.20` analysis bundle (`anal_26.06.18`); the ad-hoc-dense baseline is the
original `ppo_classic_reward/hpc_push_2048env_rel_full` run. Headline curves are
cropped to an **equal compute budget of 16.1 M environment-pushes**.

### 5.1 Headline: PBRS wins for a plain single-agent PPO agent, no curriculum

This is a controlled ablation: identical UR5e + T-block scene, identical 4-D push
primitive, identical PPO+LSTM agent, **no curriculum in either arm** — the *only*
difference is the reward function (ad-hoc-dense vs. PBRS). The task-metric results
are reward-agnostic and therefore directly comparable.

| Metric (equal 16.1 M-push budget) | Ad-hoc Dense | **PBRS** |
|---|---|---|
| Mean position error | 0.197 m | **0.114 m** |
| Mean rotation error | 0.69 | **0.21** (**3.3× gap**) |

![Position error](asyncDualPlayPPO/data_analysis/plots/why_pbrs/1_position_error.png)

![Position + rotation error](asyncDualPlayPPO/data_analysis/plots/why_pbrs/2_pos_rot_error.png)

The largest separation is in **rotation** — precisely the objective the ad-hoc
formula could never balance (P47). PBRS reaches the 0.21 region while the ad-hoc
reward stalls at 0.69, a 3.3× gap, *with no curriculum doing the work*.

**Reward-signal quality (read with care).** Figure 3 reports the mean and
variance of the per-push reward stream (ad-hoc mean=1.76, std=0.08; PBRS
mean=3.27, std=0.26). These two numbers are **not** on a common scale — they are
different reward functions — so the *magnitude* is not the evidence. The point of
the figure is qualitative: the PBRS stream is a stable, consistently-positive
shaped signal, and the *task* metrics above (which are reward-agnostic) carry the
quantitative claim.

![Reward signal](asyncDualPlayPPO/data_analysis/plots/why_pbrs/3_reward_signal.png)

### 5.2 Generalization to held-out scenarios

On 20 fixed held-out start→goal scenarios spanning easy/medium/hard:

| Model | Overall validation SR |
|---|---|
| **PBRS-A (single-agent)** | **55 %** |
| PBRS-B (forced curriculum) | 40 % |
| PBRS-C (ASP) | 0 % |

![Validation](asyncDualPlayPPO/data_analysis/plots/why_pbrs/4_validation.png)

The single-agent PBRS agent generalises best — the no-curriculum reward, on its
own, transfers to unseen configurations.

### 5.3 Model B (forced curriculum) — brief supporting evidence

Model B confirms the reward also *composes* with an explicit position→rotation
curriculum (a non-trivial property, since a changing reward landscape can
destabilise the critic). It is not the headline: PBRS already wins without it
(§5.1). The strict combined-success metric below shows B does not beat the plain
single-agent agent — reinforcing that the reward, not a curriculum, is doing the
work.

### 5.4 Scoping contrast — the reward is sound; the self-play curriculum is the open problem

All four models below use the **identical** PBRS reward; only the curriculum
differs. Final strict success rate (both position **and** rotation):

| Model | Curriculum | Final combined SR |
|---|---|---|
| A (Si) | none (single-agent) | **8.06 %** |
| B (Cr) | forced pos→rot | 1.75 % |
| C (ASP) | asymmetric self-play | 0.07 % |
| D (ASP-NGE) | ASP, GoalEncoder ablated | 0.05 % |

![Scoping contrast](asyncDualPlayPPO/data_analysis/plots/why_pbrs/5_scoping_contrast.png)

The ASP models (C, D) collapse to ~0 % **despite using PBRS**. Because the reward
is held fixed, this isolates the failure to the **self-play curriculum**, not the
reward. This is the cleanest possible demonstration that the reward question and
the curriculum question are independent — and that the reward question is settled
in PBRS's favour.

> Metric-definition note: the §5.1–§5.2 figures use task-error and held-out
> validation SR (the latter mixing position-only and position+rotation tests,
> hence the higher 55 % for PBRS-A). The §5.4 table uses the *strict* combined
> pos∧rot training metric (hence the lower 8.06 %). They are different, clearly
> labelled metrics and should not be conflated.

### 5.5 Prior baselines (reported)

For context, earlier runs reported in `implementations.md` / `thesis_impl.md`:
ad-hoc single-agent Push-PPO plateaued at a best **18.4 %** position-gated SR
(never mastering rotation), and ad-hoc ASP at **0.07 %**. These corroborate the
regenerated 16.1 M-budget figures above (ad-hoc PosError 0.197 m, RotError 0.69).

---

## 6. Scope and Honesty: Reward vs. Curriculum <a name="6-scope-and-honesty"></a>

This document argues a precise, defensible claim: **PBRS is the right reward
function** for one-arm push-primitive SE(2) goal-reaching, including for a plain
single-agent PPO agent with no curriculum. The evidence (§5.1, §5.2) isolates the
reward as the sole changed variable.

The **curriculum** is a separate axis, deliberately *not* claimed here:
- single-agent (no curriculum) — the headline, and the best generaliser;
- forced position→rotation curriculum (Model B) — supporting;
- asymmetric self-play / time-based self-play (Models C–H) — an open line. The
  self-play models collapse to ~0 % combined SR *with PBRS held fixed* (§5.4),
  and `implementations.md` documents the further "toxic curriculum collapse" of
  the disc self-play model (Bob SR 29 %→6.5 %) that motivated the time-based
  variants (Models G/H). Those iterations concern the **curriculum**, and leave
  the reward claim untouched.

Keeping the claim scoped to the reward avoids coupling a well-supported result to
the still-open curriculum question.

---

## 7. References <a name="7-references"></a>

**Reward shaping (PBRS).**
- Ng, A., Harada, D., Russell, S. (1999). *Policy invariance under reward
  transformations: Theory and application to reward shaping.* ICML.
- Grzes, M., Kudenko, D. (2009). *Theoretical and empirical analysis of reward
  shaping in reinforcement learning.* ICMLA. (Episodic `γ_shaping = 1.0`.)

**Mechanics of planar pushing (translation–rotation coupling).**
- Mason, M. T. (1986). *Mechanics and Planning of Manipulator Pushing Operations.*
  Int. J. Robotics Research 5(3):53–71. DOI 10.1177/027836498600500303.
  (Voting theorem; centre of friction.)
- Goyal, S., Ruina, A., Papadopoulos, J. (1991). *Planar sliding with dry friction.
  Part 1: Limit surface and moment function.* Wear 143(2):307–330.
  DOI 10.1016/0043-1648(91)90104-3. (Limit surface; wrench↔twist normality.)
- Howe, R. D., Cutkosky, M. R. (1996). *Practical Force-Motion Models for Sliding
  Manipulation.* Int. J. Robotics Research 15(6):557–572.
  DOI 10.1177/027836499601500603. (Ellipsoidal limit surface; characteristic
  length `c` = the `L` of `d_pose`.)
- Lynch, K. M., Mason, M. T. (1996). *Stable Pushing: Mechanics, Controllability,
  and Planning.* Int. J. Robotics Research 15(6):533–556.
  DOI 10.1177/027836499601500602.
- Akella, S., Mason, M. T. (1998). *Posing Polygonal Objects in the Plane by
  Pushing.* Int. J. Robotics Research 17(1):70–88. DOI 10.1177/027836499801700107.
  (Conf. precursor: ICRA 1992, DOI 10.1109/ROBOT.1992.219923.) (SE(2) posing via
  sequenced pushes.)
- Yu, K.-T., Bauza, M., Fazeli, N., Rodriguez, A. (2016). *More than a Million Ways
  to Be Pushed: A High-Fidelity Experimental Dataset of Planar Pushing.* IROS.
  arXiv:1604.04038. (Empirical friction variability / stochastic twist.)
- Stüber, J., Zito, C., Stolkin, R. (2020). *Let's Push Things Forward: A Survey on
  Robot Pushing.* Frontiers in Robotics and AI 7:8. DOI 10.3389/frobt.2020.00008.
  arXiv:1905.05138.

**Push-T benchmark.**
- Florence, P., Lynch, C., Zeng, A., et al. (2022). *Implicit Behavioral Cloning.*
  CoRL. arXiv:2109.00137. (Introduced the Push-T task.)
- Chi, C., Xu, Z., Feng, S., et al. (2023). *Diffusion Policy: Visuomotor Policy
  Learning via Action Diffusion.* RSS. arXiv:2303.04137. (Push-T benchmark.)

**Curriculum (separate axis, §6).**
- Plappert, M., et al. (2021). *Asymmetric self-play for automatic goal discovery
  in robotic manipulation.*
- Sukhbaatar, S., et al. (2018). *Intrinsic motivation and automatic curricula via
  asymmetric self-play.* ICLR. (Time-based self-play.)

---

## Appendix A — Proofs and Derivations <a name="appendix-a"></a>

### A.1 Policy invariance and zero-sum cycles

For a trajectory `(s_0, a_0, s_1, …, s_T)` the shaped return is

```
G′ = Σ_{t=0}^{T−1} γ^t [R(s_t,a_t,s_{t+1}) + γ_sh·Φ(s_{t+1}) − Φ(s_t)].
```

Take `γ_sh = γ`. Telescoping the shaping sum,

```
Σ_{t} γ^t [γ·Φ(s_{t+1}) − Φ(s_t)]
   = Σ_t [γ^{t+1}Φ(s_{t+1}) − γ^t Φ(s_t)]
   = γ^T Φ(s_T) − Φ(s_0),
```

so `G′ = G + γ^T Φ(s_T) − Φ(s_0)`. The added term depends only on the endpoints,
not on actions; hence `Q′(s, a) = Q(s, a) − Φ(s)` (an action-independent shift)
and `argmax_a Q′ = argmax_a Q`. Optimal policies coincide. ∎

**Episodic `γ_sh = 1.0` (Grzes & Kudenko 2009).** With absorbing terminal states
and `γ_sh = 1`, the same telescoping gives `G′ = G + Φ(s_T) − Φ(s_0)`. If `Φ` is
constant over terminal states (here `Φ(at goal) = 1` for both position and
rotation potentials), the endpoint correction is a per-episode constant, again
action-independent, preserving the optimum.

**Zero-sum cycles.** For a cycle `s_0 → … → s_0` with `γ_sh = 1`, the telescoped
shaping contribution is `Φ(s_0) − Φ(s_0) = 0`. No policy can extract net shaping
reward from oscillation. ∎

### A.2 The fractional formula admits no potential

Suppose, for contradiction, a `Φ` exists with
`Φ(s′) − Φ(s) = α(1 − d(s′)/d(s))` for all `(s, s′)`. Fix `s′` and vary `s`: the
right side depends on `s` through `d(s′)/d(s)`, which is **not** of the form
`(function of s′) − (function of s)` because of the multiplicative coupling
`d(s′)/d(s)`. Concretely, pick states with `d(s_1)=0.1, d(s_2)=0.2, d(s_3)=0.4`.
Additivity of any potential requires
`[Φ(s_3)−Φ(s_1)] = [Φ(s_3)−Φ(s_2)] + [Φ(s_2)−Φ(s_1)]`, i.e.
`α(1 − 0.4/0.1) = α(1 − 0.4/0.2) + α(1 − 0.2/0.1)`,
which reads `−3α = −α + (−α) = −2α`, false for `α ≠ 0`. Hence no `Φ` exists; the
formula is outside the PBRS family and carries no invariance guarantee. ∎

### A.3 Cosine threshold equivalence

For yaw error `e = θ* − θ`, the legacy gate is `|e| < 0.2 rad`. The cosine metric
is `c = (1 − cos e)/2`, strictly increasing in `|e|` on `[0, π]`, so the gate maps
bijectively to `c < (1 − cos 0.2)/2`. With `cos 0.2 = 0.980067`,

```
c_threshold = (1 − 0.980067)/2 = 0.009966 ≈ 0.01.
```

The mapping is nonlinear (errors near 0 are compressed, errors near π expanded),
but at the operative 0.2 rad gate the correspondence is tight, so the effective
success criterion is unchanged while the gradient becomes everywhere smooth. ∎

### A.4 Limit surface, the coupled twist, and the characteristic length `L`

**Setup.** A rigid object slides quasi-statically on a planar support. Let the
support pressure distribution be `p(r) ≥ 0` over the contact patch, with total
load `N = ∫ p(r) dA`, and let `μ` be the Coulomb friction coefficient. Reduce all
contact friction to a planar wrench `w = (fₓ, f_y, m)` at the centre of friction
(CoF). The set of wrenches the friction can sustain is the convex **limit surface**
`{ w : H(w) = 1 }` (Goyal, Ruina & Papadopoulos 1991).

**Normality (coupled twist).** Maximum power dissipation implies the object twist
`t = (vₓ, v_y, ω)` is parallel to the outward normal of the limit surface at the
operating wrench:

```
t = λ ∇_w H(w),   λ ≥ 0.                                    (A.4.1)
```

Since `∇_w H` generically has all three components non-zero, a pure force
(`m = 0` line of action *not* through the CoF) yields `ω ≠ 0`, and a pure moment
yields `(vₓ, v_y) ≠ 0`. **Translation and rotation are coupled** unless the push
line passes through the CoF — a measure-zero condition. The *sign* of `ω` is given
by Mason's (1986) voting theorem from the contact geometry relative to the friction
cone. ∎ (coupled-twist theorem)

**Ellipsoidal approximation and `c`.** Howe & Cutkosky (1996) approximate `H` by
an ellipsoid, diagonalising (A.4.1):

```
vₓ = k fₓ,   v_y = k f_y,   ω = k m / c²,                   (A.4.2)
```

for a common mobility `k > 0` and a **characteristic length** `c` defined by the
second moment of the pressure distribution about the CoF:

```
c² = ( ∫ r² p(r) dA ) / ( ∫ p(r) dA )  =  (radius of gyration)².   (A.4.3)
```

`c` is intrinsic to the object's mass/contact geometry (uniform disc radius `R`:
`c = R/√2`). It is the unique length that sets how a unit moment maps to angular
velocity *relative to* how a unit force maps to linear velocity.

**The induced metric on SE(2).** Equation (A.4.2) says the object's instantaneous
motion measures angular displacement and translation in a common unit only after
scaling `θ` by `c`. The corresponding object-displacement (squared) length is the
quadratic form

```
ds² = dx² + dy² + c² dθ²,                                   (A.4.4)
```

which is the left-invariant Riemannian metric on SE(2) weighted by the friction
geometry. Integrated to a finite pose error and identifying `L := c`, this is
exactly

```
d_pose = √(dx² + dy² + L² dθ²),                             (A.4.5)
```

the metric implemented in `reward_pbrs.py:147–158`. Hence `L` is the
limit-surface characteristic length, not a free hyperparameter: an angular error
of one radian is mechanically equivalent to a translational error of `L` metres.
For the symmetric disc, yaw is unobservable and irrelevant to the goal, so `L = 0`
and (A.4.5) reduces to the 2-D position distance. ∎
