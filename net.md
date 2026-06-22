# Network & Pipeline Reference

Architecture reference for the push-primitive PBRS models (current research), the two push
baselines, and the reference cuRobo-IK ASP implementation. Chronological history is in
[`implementations.md`](implementations.md); usage is in [`README.md`](README.md).

All policies use a **MultiCategorical** head and an **LSTM** trunk. Observations use ZYX
Euler angles (quaternions from Isaac Sim are converted in `observations.py`).

---

## 1. Push-primitive PBRS models (A–H) — current research

### Action space — 4D × 21 bins, object-relative
Decoded by `action_push_relative.py`:

| Dim | Param | Range | Meaning |
|---|---|---|---|
| 0 | `r` | 0.02–0.08 m (T-block) / 0.06–0.12 m (disc) | radial approach offset from object center |
| 1 | `φ` | [−π, π] | approach angle in object frame |
| 2 | `length` | 0–0.20 m | push distance (0 = hold) |
| 3 | `θ` | [−π, π] | push direction in world frame |

`Xs = obj_x + r·cos(obj_yaw+φ)`, `Ys = obj_y + r·sin(obj_yaw+φ)`,
`Xf = Xs + length·cos θ`, `Yf = Ys + length·sin θ`. `r` near the object guarantees contact;
world-frame `θ` makes goal-reaching translation trivial. `compute_push_waypoints()` expands
this into ~72 IK substeps per push.

### Reward — Potential-Based Reward Shaping (Ng et al. 1999)
Dense per-push reward `F = Φ(s′) − Φ(s)`, `γ_shaping = 1.0` (episodic). Preserves the
optimal policy and produces zero-sum cycles (no reward for loitering).

- **Models A–D** (separate potentials): `Φ = w_pos·exp(−k_p·d_pos²) + w_rot·exp(−k_r·d_rot²)`,
  `k_p = 30, k_r = 5, w_pos = w_rot = 10`; `d_rot` is cosine angular distance.
- **Models E–H** (SE(2) metric): `d_pose = √(dx² + dy² + L²·dθ²)`, `Φ = w·exp(−k·d_pose²)`,
  `L = 0.07 m` (T-block) or `0` (disc — rotation observed but unrewarded).

Layered sparse: `+5` position gate, `+2` full threshold (terminates), `−10` for
launched / tipped / out-of-workspace / through-table.

### Time-based Alice reward (Models G/H, Sukhbaatar 2018)
`R_A = γ_sp · max(0, t_B − t_A)`, `γ_sp = 0.5` — Alice is rewarded for goals she builds
quickly (`t_A`) that Bob takes long to solve (`t_B`), self-regulating the curriculum.

### Observations (Alice / Bob; single-agent A,B use Bob's layout)
| Component | Dims | Alice | Bob | Contents |
|---|---|---|---|---|
| `ee_pose` | 6 | ✓ | ✓ | EE `[x,y,z, roll,pitch,yaw]` (Euler, env-local; no gripper) |
| `obj_state` | 14 | ✓ | ✓ | `[pos(3), euler(3), linvel(3), angvel(3), ee_dist, contact]` |
| `goal_pose` | 6 | — | ✓ | desired `[pos(3), euler(3)]` |
| `goal_dist` | 2 | — | ✓ | slot, repurposed per mode (below) |
| **Total** | | **20D** | **28D** | |

`goal_dist(2)` slot: baseline `[pos_dist, rot_dist]` · `--rel-obs` `[goal_xy − obj_xy]` ·
`d_pose` (E–H) `[d_pose, bearing]`, `bearing = atan2(dy, dx)`.

### Model matrix
| Model | Agents | Object | Reward metric | Alice reward | GoalEncoder |
|---|---|---|---|---|---|
| A | single | T-block | PBRS pos + rot | — | — |
| B | single | T-block | PBRS pos + rot (pos→rot curriculum) | — | — |
| C | Alice+Bob | T-block | PBRS pos + rot | outcome (+5/−1) | yes |
| D | Alice+Bob | T-block | PBRS pos + rot | outcome | ablated |
| E | Alice+Bob | T-block | `d_pose` (L=0.07) | outcome | yes |
| F | Alice+Bob | disc | `d_pose` (L=0) | outcome | yes |
| G | Alice+Bob | T-block | `d_pose` (L=0.07) | time-based | yes |
| H | Alice+Bob | disc | `d_pose` (L=0) | time-based | yes |

### Networks
- **Single-agent (A, B)** — `ActorCriticPush` (`module_push.py`): flat MLP
  `obs→512→256` → `LSTMCell(256)` → actor `Linear(256→4×21)`; separate MLP critic.
- **ASP Bob (C, E–H)** — `ActorCritic` (`module.py`): per-object PI-encoder (`14→512→512`,
  max-pool, LayerNorm) + GoalEncoder latent injected additively into the first trunk layer;
  same LSTM + 4×21 head. Model **D** drops the GoalEncoder (PI-encoder reads the full
  per-object chunk). **Alice** uses the same trunk with no goal input.

### Files
`train_a_pbrs_simple.py` … `train_h_tasp_disc.py`; envs `tasks/push_task_curobo.py`
(T-block) / `push_task_curobo_disc.py` (disc); wrappers `tasks/utils/wrapper_push.py`
(A, B) / `wrapper_push_asp.py` (C–H); reward `tasks/utils/reward_pbrs.py`; one
`hpc/train_[a-h]_*.slurm` per model.

---

## 2. Push baselines (`train_push.py`, `train_push_asp.py`)

Same 4D × 21 push action and ~72-substep primitive as above.

- **Push-PPO** (`train_push.py`): single agent, 28D obs
  `[ee_pose(6) | obj_state(14) | goal_pose(6) | goal_dist(2)]` (30D with `--rel-obs`),
  flat MLP + LSTM, normalised fractional-improvement reward (Fix P63). No ASP/ABC.
- **Push-ASP** (`train_push_asp.py`): ASP two-phase loop with the object-relative push
  primitive. Alice obs 20D `[ee_pose(6) | obj_state(14)]`; Bob obs 25D adds
  `rel_goal(5) = [Δx, Δy, rel_yaw, pos_dist, rot_dist]`. Bob uses the GoalEncoder + LSTM.

The object-relative parameterization raises contact probability per push from ~2%
(absolute) to ~95%+, which is what lets Alice bootstrap.

---

## 3. Reference cuRobo-IK ASP (`train_curobo.py`)

Step-based EE-delta ASP — the canonical cuRobo-IK-in-Isaac-Lab example, not a paper result.

### Observations (Alice / Bob; N = num_objects)
| Component | Dims | Alice | Bob | Contents |
|---|---|---|---|---|
| `robot_state` | 7 | ✓ | ✓ | `ee_pose(6 Euler)` + `gripper(1)` |
| `obj_state` | 14×N | ✓ | ✓ | `[pos(3), euler(3), linvel(3), angvel(3), ee_dist, contact]` |
| `goal_pose` | 6×N | — | ✓ | desired `[pos(3), euler(3)]` |
| `dist` | 2×N | — | ✓ | `[pos_dist, rot_dist]` |
| **Total (N=1 / N=2)** | | **21D / 35D** | **29D / 51D** | |

Bob's per-object data is interleaved as 22D chunks `[obj_state(14) | goal_pose(6) | dist(2)]`.

### Action space — 6D × 11 bins (bin 5 = zero)
Dims 0–2: XYZ delta `(bin−5)/5 · max_delta_m` (0.04 m); dims 3–4: Rx,Ry delta
(0.05 rad, clamped ±0.1); dim 5: sticky gripper (close / hold / open). Targets are
**integrated** (`ee_target_local += Δ`, clamped to the workspace) and solved to absolute
joint positions by cuRobo.

### Network
PI-encoder over objects (`14→512→512`, max-pool, LayerNorm) concatenated with
`robot_state` → trunk `Linear(519→512)→256→128` → `LSTMCell` → actor `Linear(→6×11)`;
critic is a separate MLP on the raw observation. **Bob** adds a GoalEncoder φ-MLP
(`Linear(6→64)→Tanh→Linear(64→8)`, difference variant `φ(goal)−φ(current)`, sum-pooled)
injected additively into the first trunk layer: `h₁ = ReLU(LN(W·enc + Wg·g))`.

### Per-step IK pipeline
Decode bins → integrate EE position/orientation targets → TCP offset correction
(finger-midpoint vs `wrist_3_link`) → `ik_solver.solve_batch()` warm-started from the
previous command → EMA joint smoothing (α=0.2). cuRobo must be imported **before**
`AppLauncher` (torch version lock). IK failure: Alice terminates (−1, arm locked); Bob
reverts the target and holds. Fail rate logged as `Metrics/IKFailRate`.

### Training loop
Two-phase rollout (Alice 100 steps, Bob 100–200 steps). 80% current policy / 20%
historical snapshot per agent. On phase transition or done, IK accumulators re-anchor to
physics state and LSTM hidden states zero. Alice: standard PPO. Bob: PPOABC =
PPO + clipped ABC imitation loss (β=0.5, sequential LSTM eval, GoalEncoder detached) +
auxiliary GoalEncoder distance loss. Failed-Bob Alice trajectories feed a sliding-window
`GPUDemonstrationBuffer`. Checkpoints save Alice/Bob weights+optimizer, ABC buffer,
episode manager, and train state.

### Paper vs current (key deltas)
| | Plappert 2021 | Current |
|---|---|---|
| Robot state | joint angles (6D) | EE Euler pose + gripper (7D) |
| Goal encoding | raw PI embedding | GoalEncoder → 8D latent, additive injection |
| PI pooling | sum-pool | max-pool (DeepSets) |
| Action | continuous Gaussian | MultiCategorical 6×11 |
| IK failure | OSC (always valid) | Alice terminate −1 / Bob hold |
