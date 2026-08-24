# Dual-Arm ASP + HRL: Technical Overview

## Core Idea

Two training phases produce a dual-arm system trained entirely without external reward:

1. **Phase 1** — Train a single-arm sub-policy via Asymmetric Self-Play (ASP)
2. **Phase 2** — Train a manager policy via Meta-ASP over the frozen sub-policy

---

## Phase 1: Single-Arm ASP

**Agents:** Alice and Bob, each a separate PPO policy with shared architecture.

**Game:** Alice acts for `T_A` steps from `s_0`, reaching state `s*`. Bob starts from `s_0` and must reach `s*` within `T_B` steps.

**Rewards:**
```
R_Bob  = -γ · t_B               (minimise steps)
R_Alice = γ · max(0, t_B - t_A)  (propose goals just beyond Bob)
```

**Goal Encoder `E`:** Maps `(s*, s_t)` → `g ∈ R^K` (K=8–16). Two variants:
- *Difference:* `E(s*, s_t) = φ(s*) - φ(s_t)`
- *Absolute:* `E(s*, s_t) = φ(s*)`

Bob's policy: `a_t = π_B(s_t, E(s*, s_t), r_id)`

**Alice Behavioral Cloning (ABC):** When Bob fails, Alice's trajectory is added to a BC buffer. Bob's loss:
```
L_Bob = L_PPO + β · L_ABC
```
Only failed-goal demonstrations are kept (no demos for already-mastered goals).

**Role Identifier `r_id ∈ {0,1}`:** Allows one shared policy to serve as either arm at deployment. During Phase 1, one slot is zero-padded; `r_id` tells the policy which observation block is "self". Randomly sampled each episode to force the policy to learn to use it.

**Multi-goal episodes:** Up to `N_G=5` goals per episode. Bob's final state becomes the next start state, enabling deep workspace exploration.

**Phase 1 output:** Frozen `π_B*` and encoder `E*`. Alice is discarded.

---

## Phase 2: Manager via Meta-ASP

**Manager `π_M`:** Maps joint bimanual state → joint goal pair `[g^(1), g^(2)] ∈ R^(2K)`.

Architecture: shared MLP trunk → two separate output heads (one per arm). Shared trunk forces joint reasoning before goal assignment.

**Meta-ASP game:** Mirrors arm-level ASP one level up.
- *meta-Alice* issues goal sequences to the frozen arms to rearrange the workspace to `s*_ws`
- *meta-Bob* (= `π_M`) starts from the same `s_0` and must replicate `s*_ws`
- meta-Alice's trajectory is a valid demo → enables ABC at manager level

**Reward:**
```
R_meta-Bob  = 1 if workspace reaches s*_ws, else 0
R_meta-Alice = 1 - R_meta-Bob
```

**Manager loss:**
```
L_M = L_PPO + β_M · L_ABC + λ · L_collision + μ · L_conflict
```

**Physics penalties (via IsaacLab/PhysX):**

*Arm-arm collision:*
```
L_collision = (1/T_C) · Σ_t max_j ||f_t^(j)||   # net contact forces between arm links
```

*Object conflict* (both end-effectors near same object):
```
L_conflict = 1[ ∃k : D(ee1, o_k) < d_prox ∧ D(ee2, o_k) < d_prox ]
```

Collision weight `λ` is annealed: `λ(t) = λ_max · exp(-t/τ) + λ_min`.

**Manager goal period:** `T_C = T_A`. Manager re-issues goals every `T_C` steps; arms execute frozen sub-policies with fixed goals in between.

---

## Observation Structure

**Arm sub-policy input:**
```
o_t = [own_joints(6), other_joints(6, zero-padded in Phase 1),
       goal_embedding(K), r_id(1)]
```

**Manager input:**
```
[arm1_joints(6), arm2_joints(6), ee1_pos(3), ee2_pos(3),
 object_states(M×13), task_context(Z)]
```

---

## Key Numbers

| Param | Value |
|---|---|
| `T_A` (Alice steps) | 100 |
| `T_B` (Bob steps) | 200 |
| `T_C` (manager period) | = T_A |
| `N_G` (goals/episode) | 5 |
| Goal dim `K` | 8–16 |
| PPO clip `ε` | 0.2 |
| ABC weight `β` | 0.5 |
| Proximity threshold | 0.08 m |
| Parallel envs | 4096 |
