# Implementation Record — ASP + GoalEncoder + Push-PPO Baseline

**Branch**: `asp_goal_encoder`  
**Last updated**: 2026-05-22 (Fixes P39–P44: push obs 29→28D, waypoint loop death, exploded state, zero penalty, dynamic minibatches, elbow-IK no-terminate)

---

## Table of Contents

1. [Overview & Comparison Targets](#overview)
2. [Development Timeline](#timeline)
3. [ASP + GoalEncoder Implementation](#asp-goalencoder)
4. [cuRobo IK Integration](#curobo-ik)
5. [HPC Setup & Run Guide](#hpc-setup)
6. [Push-PPO Baseline](#push-ppo)
7. [Push Primitive Test](#push-primitive-test)

---

## 1. Overview & Comparison Targets <a name="overview"></a>

The project trains Alice+Bob PPO with **Asymmetric Self-Play** (Plappert et al. 2021) augmented
with a **GoalEncoder** (Sukhbaatar et al. 2018) on two UR5e arms in Isaac Lab. Three controller
variants exist: **RMPFlow** (`train.py`), **DifferentialIK** (`train_diffik.py`), and
**cuRobo IK** (`train_curobo.py`). cuRobo is the primary/recommended variant.

A fourth script, `train_push.py`, implements a single-agent **Push-PPO Baseline** for comparison.

### Comparison Targets

| # | Approach | Script |
|---|----------|--------|
| 1 | **ASP + GoalEncoder** (Plappert + Sukhbaatar) | `train_curobo.py` |
| 2 | **ASP + GoalEncoder (Charlie)** — same as above | `train_curobo.py` |
| 3 | **Push-PPO Baseline** — single-agent PPO with push primitive | `train_push.py` |

### Stack Versions

| Component | Version |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 (commit `6c151ea`) |
| **cuRobo** | **0.7.5** — last release with `IKSolver`/`solve_batch`/`Pose` API |
| PyTorch | 2.7.0+cu128 |
| Python | 3.11.5 |
| Container (HPC) | `nvcr.io/nvidia/isaac-lab:2.3.0` (Apptainer `.sif`) |
| GPU (HPC) | RTX Pro 6000 (96 GB VRAM) |

> **Why v0.7.5?** The `0.8.x` series renamed `IKSolver`, `Pose`, `TensorDeviceType`.
> Using a newer tag will fail at import unless you update the import block in
> `train_curobo.py` and `tests/test_curobo_follow_target.py`.

---

## 2. Development Timeline <a name="timeline"></a>

| Date | Milestone |
|------|-----------|
| 2026-04-01 | Initial HPC scripts, README, fixed early tests |
| 2026-04-02 | Increased Bob displacement threshold; HPC GPU-buffer reload on reset |
| 2026-04-07 | Fixed ABC LSTM hidden-state propagation; removed excess logs for speed |
| 2026-04-08 | ABC test pass; GoalEncoder+LSTM test pass; video recording; physics crash fix |
| 2026-04-09 | Entropy fix; one-object mode; improved SLURM/logging |
| 2026-04-10 | Training visually stable |
| 2026-04-13 | Full pipeline (Alice→goal→Bob) end-to-end working |
| 2026-04-14 | New diagnostic test pass; HPC diagnostic SLURM; fixed trajectory chaining |
| 2026-04-15 | Test 2 (Alice sandbox) passes; diagnostic wrapper added |
| 2026-04-16 | Alice dense reward → potential-based; entropy annealing fixed |
| 2026-04-17 | All 4 diagnostic tests pass |
| 2026-04-20 | GoalEncoder axis-angle → ZYX Euler fix; HPC runs stable |
| 2026-04-21 | Profiler added; 2× speed improvement; Alice reward pass |
| 2026-04-25 | abc_coef + entropy moved to SR-coupled controllers; GPU buffer sliding window |
| 2026-05-03 | Habrok HPC integration; fixed validation transition |
| 2026-05-04 | cuRobo IK test (`test_curobo_follow_target.py`) working; ball-following demo; controller support; gripper open/close |
| 2026-05-05 | `train_curobo.py` initial cuRobo training integration; gripper test forks |
| 2026-05-06 | Fixed workspace clamp / IK fail-rate bugs; HPC cuRobo overlay |
| 2026-05-07 | Removed ABC warmup gate (Fix 9); all diagnostics pass with cuRobo |
| 2026-05-08 | SR-coupled controllers reverted (Fixes 1 & 2); cuRobo install docs; all tests green |
| 2026-05-11 | Push primitive refactor (no yaw/grip/spin, fixed TCP offset), rotation reward function, goal yaw randomization |
| 2026-05-11 | Reward function: position+rotation improvement (Akella & Mason 1998), no distance penalty |
| 2026-05-13 | T-block set as first scenario object; placeholder inertia removed from all objects (objects now spin) |
| 2026-05-13 | Robot lag fix: cube/cylinder/rect/triangle removed from push test scene; T-block only; approach height raised to 0.40 m, 115 substeps total |
| 2026-05-13 | T-block switched to sole ASP task object; target_object→t_shape.usda, spawn (0.0,0.5,0.05); goal_ghost→T-block shape; EE home Y=0.50 added |
| 2026-05-13 | Goal validation: z_max=0.05 rejects airborne T-block goals; out_of_zone goals now fully invalid (was accepted with penalty); Bob off-table handler pays Alice +5, hides ghost, random-safe resets objects |
| 2026-05-13 | ABC debug per-step prints removed; per-episode [ALICE END] and [BOB END] logging with start/final positions, orientations, errors |
| 2026-05-13 | Training log analyzer (analyze_training.py): --log-dir required; cuRobo compatibility confirmed |
| 2026-05-13 | Alice IK fail → immediate episode end with -1 penalty; arm locked in place on fail; overrides wrapper's -3 |
| 2026-05-13 | Too-easy goal filter: after resetting objects for Bob, if Bob starts within success threshold (pos<0.05m AND rot<0.2rad), goal rejected with -3 penalty |
| 2026-05-13 | All documentation (implementations.md, net.md, README.md) updated for T-block scene, 1-object obs dims, goal validation, IK fail handling |
| 2026-05-13 | Push-PPO Fix P1: module_push.py trunk init gain 0.01→sqrt(2), actor_head only keeps 0.01 |
| 2026-05-13 | Push-PPO Fix P2: PUSH_SUCCESS_THRESHOLD_ROT 0.035→0.2 rad (1.1%→6.4% of rotation space) |
| 2026-05-13 | Push-PPO Fix P3: removed per-push LSTM zeroing in train_push.py; hidden state propagates within episode, zeroed only at done boundaries |
| 2026-05-13 | Push-PPO Fix P4: mid-trajectory terminated envs produced garbage rewards (post-reset obs in compute_push_reward); zeroed for those envs |
| 2026-05-13 | Push-PPO Fix P5: sr_buf maxlen 200→push_nsteps×num_envs; was dropping first 80% of each rollout's successes from the SR metric |
| 2026-05-14 | Push-PPO Fix P6: num_bins 11→21 — bin resolution 0.06m→0.03m, now below the 0.05m position success threshold. Actor output 66→126 dims |
| 2026-05-14 | Push-PPO Fix P7: dim 4 (yaw, previously decoded but silently dropped) now drives EE Z-rotation during push phase. Phase 4 quat interpolates tool-down→yaw-rotated; Phase 5 retract keeps final yaw. `compute_push_waypoints` signature gains `yaw` parameter |
| 2026-05-14 | Push-PPO Fix P8: max_pushes_per_episode 3→5 — gives policy room to recover from bad pushes and commit to precision approach |
| 2026-05-15 | Push-PPO Fix P9: max_yaw π→1.0 rad — at π, yaw forced IK into elbow-forward branch at extreme angles; ±1.0 rad (±57°) keeps IK in elbow-up branch and gives 0.1 rad/bin precision (below 0.2 rad success threshold) |
| 2026-05-15 | Push-PPO Fix P10: decode_push_action default num_bins 11→21 (mismatch with train_push.py override); PushConfig duplicate `num_bins` line removed |
| 2026-05-15 | Push-PPO Fix P11: checkpoint resume support — `--resume_iteration` and `--resume_best_sr` args; `ppo.py` save/load now includes optimizer state dict for proper momentum resumption; backward-compatible with old plain-state-dict checkpoints |
| 2026-05-15 | Push-PPO Fix P12: ent_coef 0.01→0.002 — `ent_coef=0.01 × ~18 nats entropy = −0.18` dominated surrogate loss (~0.02), killing gradient signal. γ 0.998→0.95 — at 0.998 all 32 rollout steps got ~94% weight (GAE horizon ~19 steps), making every push look equally (un)important. +RotationSR metric for rotation-only success rate (rot_err<0.2 rad, independent of position gate) |
| 2026-05-15 | Push-PPO Fix P13 (LSTM hidden-state propagation): `evaluate()` previously used zero-init LSTM, creating spurious `π_new/π_old` ratio for pushes 2–5. Now `act_with_hidden` returns `h_in`, stored in `RolloutStorage`, yielded during mini-batch update and passed to `evaluate()` — PPO ratio now reflects genuine weight changes, not LSTM amnesia. Files: `module_push.py`, `storage.py`, `ppo.py`, `train_push.py` |
| 2026-05-15 | Push-PPO Fix P14 (rotation sub-bonus): `+2` bonus when `pos_err < 0.05 AND rot_err < 0.2`. Keeps the `+5` position-only gate (5.7% SR floor) and layers a priority-driven curriculum: primary spatial objective → secondary rotation polishing. `wrapper_push.py:24,224-232` |
| 2026-05-15 | Push-PPO Fix P15 (yaw-isolated rotation reward): replaced `max(|roll|,|pitch|,|yaw|)` with `_yaw_distance_rad()` for the dense improvement term. Planar pushing should track only Z-axis rotation; roll/pitch wobble during translation was a noisy contaminant. Full max-Euler is kept for `rot_err` metric and tip-over detection. `wrapper_push.py:62-75,197-198,212` |
| 2026-05-15 | Push-PPO Fix P16 (tip-over termination): if `abs(roll) > 0.3` or `abs(pitch) > 0.3` (object unrecoverably tipped), episode terminates with −5 penalty. Prunes garbage transitions from the PPO buffer and teaches safe `push_dz` constraints. `wrapper_push.py:30,241-243,286-288` |
| 2026-05-15 | Push-PPO Fix P17 (continuous rotation penalty): added `−PUSH_DENSE_ROT_BETA × yaw_err` (β_rot=0.25) to the dense reward, mirroring the positional penalty `−β·d_now`. Provides per-push urgency to fix orientation, preventing the agent from loitering after achieving position. `wrapper_push.py:28,214` |
| 2026-05-17 | Push-PPO Fix P18 (reward coefficient scaling, final): α 10→12 (1.2×), γ 2→5 (2.5×). Initial attempt at α=30,γ=10 caused catastrophic value function instability on fresh training (Val loss 52→5760→11105 — GAE chain reactions from noisy value predictions amplified by 5–16× larger return variance). α=12,γ=5 provides 2.4× wider reward gap vs original while keeping returns within the critic's initial fit range (expected Val loss ~12 at iter 0). Fresh training stable. `wrapper_push.py:25-26` |
| 2026-05-18 | ASP LSTM hidden-state propagation fix (same as Push-PPO Fix P13): `module.py.evaluate()` now accepts `hidden_state` parameter; Alice and Bob pre-action hidden states captured, zeroed for hist/non-active envs, stored via `storage.add_transitions()`, yielded during PPO mini-batch update, and passed to `evaluate()` — PPO ratio for ASP reflects genuine weight changes. Files: `module.py`, `ppo_abc.py`, `train_curobo.py` |
| 2026-05-18 | New ASP rotation metrics: `Metrics/Alice/RotChgRoll`, `Metrics/Alice/RotChgPitch`, `Metrics/Alice/RotChgYaw` track per-axis orientation change Alice introduces in goals. `Metrics/Bob/PositionSR`, `Metrics/Bob/RotationSR` provide position-only and rotation-only SR independent of combined success — matches Push-PPO baseline pattern. Buffer population fix: `bob_pos_err_buf`/`bob_rot_err_buf` were dead (never pop'd). Iteration summary prints `[AliceRot]` and `[BobSR]` lines. Files: `train_curobo.py` |
| 2026-05-18 | Log analyzer updated: `analyze_training.py` parses new `[AliceRot]` and `[BobSR]` log lines; CSV output gains `alice_rot_roll/pitch/yaw`, `pos_sr`, `rot_sr`, `pos_err`, `rot_err` fields; combined and separate plots include Alice rotation change (3-panel roll/pitch/yaw) and Bob PosSR/RotSR panels. |
| 2026-05-18 | **ASP reward rules fixed**: `ent_coef` 0.05→0.005 (YAML) — entropy bonus was 35× surrogate loss at 0.05 × 14 nats max ≈ 0.7, now 3.5× at 0.005 × 14 ≈ 0.07. Both Alice and Bob inherit the lower value since they share `ppo_continuous.yaml`. Minimum-displacement penalty added to `validate_goal()`: goals with max displacement 0.05–0.10 m get −1 "shallow" penalty instead of +1. Goals remain valid for Bob (he still practices). Alice is pushed to create goals with meaningful displacement (>0.10 m for +1). `goal_validator.py`, `wrapper.py` |
| 2026-05-19 | **Bob dense delta reward reverted (Fix P27)** — v5 (dense) killed Alice's emergent curriculum compared to v1 (sparse). `wrapper.py` |
| 2026-05-19 | **Why the dense reward was reverted** — v1 (sparse-only, `asp_curobo_v1.log`) showed Alice's avg 3D displacement growing from 0.037m to 0.120m over 90 iterations, with not-moved dropping 84%→37% — Alice was learning. Bob SR stagnated at 4–5% but the adversarial curriculum was emerging. v5 (dense delta, `asp_curobo_v5.log`) showed Alice stuck at 0.047m with 80–90% not-moved for all 13 iterations. Root cause: the per-step `Φ(s')−Φ(s)` delta was zero-mean noise (±0.02 with 50% of steps producing exactly 0.0); sparse rewards (+1/−1/+5) fired on only ~10% of episodes; the combined reward stream produced GAE advantages indistinguishable from noise, starving both Bob's PPO and Alice's delayed outcome rewards of any learnable gradient. Sparse-only `{+1/−1/+5}` restored. |
| 2026-05-19 | **Phase-end progress reward for Bob (Fix P28)** — episodic feedback mirrors Alice's structure. `r_progress = clamp(w_pos·(init−final)/init + w_rot·(init−final)/init, −1, +1)` paid once at Bob termination. `bob_timesteps` 200→100 halves credit-assignment horizon. |
| 2026-05-18 | **Bob rotation control improved**: `max_delta_rot` 0.05→0.10 rad/step (2.9°→5.7°) and Rx/Ry clamp 0.05→0.10 rad — doubled EE tilt range per step so Bob can apply more torque to rotate objects through contact. `train_curobo.py:279,301-303` |
| 2026-05-19 | Push-PPO Fix P29 (critic output gain): `module_push.py` critic output `Linear(128→1)` had gain=1.0, producing initial value predictions ~±5–10. At 512 envs × 32 pushes = 16,384 transitions, the GAE backward pass amplified this noise into returns of magnitude 1000+, causing Val loss explosions (356k+). Reduced to 0.01 — matches actor head, initial V ≈ 0.057. `module_push.py:83` |
| 2026-05-19 | Push-PPO Fix P30 (reward clamp + out-of-bounds kill) |
| 2026-05-20 | **Fix P31**: ABC deadlock diagnosed — Alice action entropy too high for ABC to bootstrap Bob; `--diag_alice_shaping` promoted from diagnostic to training flag; new HPC script `train_curobo_shaping.slurm` |
| 2026-05-20 | **Fix P32 (Push-PPO speed)**: Push substeps scaled 115→76 (~1.5× faster rollouts). CUDA-synced wall-clock profiler added to `train_push.py` — reports per-iteration timing for 7 sections: `agent`, `decode`, `ik`, `physics`, `reward`, `store`, `ppo`. `action_push.py:16-24`, `train_push.py:41-42,372-393,435+` |
| 2026-05-20 | **Fix P33 (cuRobo IK tuning)**: Profiler revealed cuRobo `solve_batch` at 65ms/call (69% of iteration), 100× slower than expected. Root cause: default `n_iters=100`, `inner_iters=25` per env with sequential `n_problems=1` loop. LBFGS reduced to `n_iters=30, inner_iters=10` — IK dropped 65→18ms/call (3.6×). MPPI particle_optimizer left untouched to avoid CUDA graph shape errors. `train_push.py:211-214` |
| 2026-05-21 | **Fix P34 (4D action space)**: Push-PPO action space redesigned from 6D (offset_x, offset_y, push_dx, push_dy, yaw, push_dz) to 4D (Xs, Ys, length, theta). Xs/Ys = push start in world coords, length ∈ [0, 0.20] m, theta ∈ [−π, π] rad. Push endpoint: Xf=Xs+len·cosθ, Yf=Ys+len·sinθ. Gripper always closed — no engage/release phases. 5 phases now (approach, descend, push, retract, return) = 72 substeps. Actor head 126→84 dims (4×21). `action_push.py`, `train_push.py`, `module_push.py` |
| 2026-05-21 | **Fix P35 (push debug markers)**: Green sphere at (Xs,Ys), red sphere at (Xf,Yf), blue cylinder arrow from start→end on table surface. Three independent `VisualizationMarkers`, updated every push. `train_push.py:269-334` |
| 2026-05-21 | **Fix P36 (per-env debug logging)**: When `num_envs ≤ 5`, each push logs per-env bins and decoded (Xs, Ys, length, θ, Xf, Yf). `train_push.py:238,549-556` |
| 2026-05-21 | **Fix P37 (length limit)**: Push length clamped to [0, 0.20] m (was [0, 0.30]). `action_push.py:130,152,158` |
| 2026-05-21 | **Fix P38 (profiler removed)**: Inline CUDA-synced profiler removed from `train_push.py` — replaced by per-push marker + per-env debug logging for visibility. `import time` also removed. |
| 2026-05-22 | **Fix P39 (gripper removed from push obs)**: Gripper carries no useful signal (always closed in push primitive). Removed from `PushPolicyCfg` observation terms. `_OBS_ROBOT_DIM` 7→6, total obs 29D→28D. `push_task_curobo.py`, `wrapper_push.py`, `module_push.py`. |
| 2026-05-22 | **Fix P40 (waypoint loop death)**: After `env.step()` auto-resets a terminated env, subsequent waypoint IK targets commanded the robot back to table position → teleport → infinite force → object explodes to 3000m. Terminated envs now hold `cur_joints` instead of executing new waypoint targets. `train_push.py:564-567`. |
| 2026-05-22 | **Fix P41 (exploded state saved into PPO buffer)**: `needs_reset = done & ~terminated` skipped explicit reset for terminated envs, leaving `obs` with post-explosion 3000m values. These observations were then captured by `obs_pre_push = obs.clone()` for the next push. Now `needs_reset = done` — all done envs get explicit `env.env.reset()` for clean observations. `train_push.py:663`. |
| 2026-05-22 | **Fix P42 (zero penalty for terminated envs)**: `reward[terminated] = 0.0` gave zero signal for off-table/exploded pushes — agent never learned to avoid them. Changed to −10.0 penalty. `train_push.py:582`. |
| 2026-05-22 | **Fix P43 (dynamic minibatches)**: `nminibatches` now derived from `num_envs` via `max(1, num_envs // 16)` with a while loop to ensure `num_envs % nminibatches == 0` (avoids wasted samples from `drop_last=True`). Keeps mini-batch size ~240 transitions independent of env count — manages GPU memory at scale without touching `push_nsteps` (LSTM temporal window stays fixed at 15). `train_push.py:149-153`. |
| 2026-05-22 | **Fix P44 (elbow-IK no-terminate)**: Elbow-negative IK solutions (`wrist_1_joint < 0`) caused immediate episode termination via `terminated=True` → −10 penalty. The IK fallback `ik_ok[elbow_bad] = False` already safely holds `prev_joint_cmd`, so the IK issue is recoverable. Removed `terminated[elbow_bad] = True` — unreachable (Xs, Ys) pairs now produce zero improvement (static penalty only) instead of death, giving PPO a continuous gradient away from bad workspace regions without destroying episodes. `train_push.py:567-572`. |

---

## 3. ASP + GoalEncoder Implementation <a name="asp-goalencoder"></a>

> Tracks all changes on `asp_goal_encoder` relative to `master`, how each component maps to
> hardware-in-the-loop validation tests, and what remains.

### 3.1 Framework Overview

| Paper | Role |
|---|---|
| Plappert et al. 2021 — *Asymmetric Self-Play* | Automatic curriculum via Alice↔Bob adversarial game |
| Sukhbaatar et al. 2018 — *Goal Embeddings via Self-Play* | GoalEncoder φ-MLP compresses goal into fixed-size latent |

#### Adversarial Loop

```
Alice (PPO, 35D obs, 100 steps)
  → manipulates objects freely
  → leaves workspace in non-trivial state
  → that state becomes the goal

Bob (PPOABC + GoalEncoder, 51D obs, 100 steps)
  → must reproduce Alice's configuration from scratch
  → goal enters via GoalEncoder latent (8D) injected into actor trunk
  → reward: +1 per object at goal, −1 if object leaves goal, +5 completion
  → phase-end progress: r = clamp(0.6·Δpos/init + 0.4·Δrot/init, −1, +1)

When Bob fails → Alice's trajectory → ABC buffer → imitation loss β=0.5
```

#### Observation Spaces

| Agent | Dimension | Contents |
|---|---|---|
| Alice | 35D | `robot_state(7)` + `obj1_state(14)` + `obj2_state(14)` |
| Bob | 51D | `robot_state(7)` + `[obj_state(14)+goal(6)+dist(2)] × 2` |
| Goal state | 12D | `[pos(3)+euler(3)] × 2` objects, local frame |

#### Kinematic Pipeline (cuRobo variant)

```
Policy output (6D MultiCategorical, 11 bins)
  → decode: XYZ delta + Rx/Ry delta + sticky gripper
  → accumulate: ee_target_local (position integrator)
              + ee_target_quat_w (orientation quaternion)
  → TCP offset correction (finger midpoint vs wrist_3_link)
  → cuRobo solve_batch(N envs) → joint positions
  → JointPositionActionCfg → Isaac Lab physics
```

---

### 3.2 What Has Been Implemented

#### Core Training Loop (`train_curobo.py`)

- **cuRobo IK pipeline** replacing RMPFlow / DiffIK:
  - `_bins_to_xyz_rxy_gripper`: decodes 6D bins → XYZ delta + Rx/Ry delta + sticky gripper
  - Per-env EE position accumulator `ee_target_local` (reset on phase transition / done)
  - Per-env EE orientation accumulator `ee_target_quat_w` via quaternion composition
  - TCP offset correction from finger-midpoint vs wrist_3_link each step
  - `ik_solver.solve_batch(N)` seeded from `_prev_joint_cmd` for smooth trajectories
  - IK failure recovery: reverts EE accumulator, holds current joint positions.  Alice IK failures trigger immediate episode termination with -1 penalty (arm locked in place); Bob IK failures are non-terminal.
  - Phase sync: re-anchors accumulators to physics TCP state after phase transition or episode done
  - Workspace clamp: X ∈ [−0.50, 0.50], Y ∈ [0.25, 0.70], Z ∈ [0.00, 0.55] metres (env-local)
  - EE home offset applied after every sync (reset / phase boundary): X += 0.02 m, Y = 0.50 m, Z = 0.05 m — arm resets to its default joint configuration then IK drives it to the preferred low-hover resting pose directly above the T-block spawn position
  - cuRobo CUDA graph warm-up before training loop (~3 ms → ~0.5 ms per step)
  - IK fail rate logged to TensorBoard (`Metrics/IKFailRate`) each iteration

- **Two-phase ASP structure**:
  - Alice phase (100 steps): explores, builds goal state
  - Bob phase (200 steps): reproduces goal, receives sparse reward
  - Phase stagger at startup: random offset across envs prevents simultaneous resets
  - LSTM hidden states zeroed on phase transitions and episode done events
  - LSTM hidden-state propagation across PPO updates (Fix P19): pre-action hidden states captured, stored in RolloutStorage, yielded during mini-batch updates, passed to `evaluate()` — PPO ratio π_new/π_old reflects genuine weight changes, not LSTM amnesia

- **Alice Behavioral Cloning (ABC)**:
  - Alice trajectory buffer: `(alice_traj_obs, alice_traj_act, alice_traj_len)` per env per rollout
  - Gate: `bob_done & ~bob_success & goal_valid & traj_len >= max(10, alice_timesteps//2)`
  - `GPUDemonstrationBuffer`: sliding window of 500 trajectories, evicts oldest on overflow
  - Buffer persisted across checkpoints (`abc_buffer.pt`), loaded on resume

- **Historical policy pool**:
  - 20% of active envs per phase use a past policy snapshot (`HIST_FRAC = 0.2`)
  - Pool holds last 5 snapshots; saved every 50 iterations

- **Fixed controllers** (SR-coupled controllers removed — Fixes 1 & 2):
  - Alice entropy coef: `ent_coef = 0.005` (fixed from YAML; reduced from 0.05 per Fix P23)
  - Alice LR: cosine decay `lr_max=3e-4 → lr_min=5e-5` over `max_iterations`
  - Bob `abc_coef`: fixed at 0.5 (paper Table 2)

- **Diagnostic flags**: `--test_reward_pipeline`, `--alice_sandbox`, `--dummy_alice`, `--profile`, `--test_hparams`

- **Checkpoint system**: periodic, best-model, SIGTERM emergency; resume via `--chkpt_alice/bob`

- **TensorBoard logging**: Loss, Reward, Metrics, GoalEncoder, ABC, IK overhead — Alice rotation change (per-axis roll/pitch/yaw), Bob PosSR/RotSR (position-only and rotation-only success rates independent of combined success), Bob PosError/RotError (now actually populated, were dead)

#### GoalEncoder (`algorithms/goal_encoder.py`)

- φ-MLP: `Linear(6→64) → Tanh → Linear(64→8)` per object
- **Difference variant** (default): `g_i = φ(goal_i) − φ(current_i)`
- **Max-pool** across objects → 8D pooled embedding `g_pooled` (Fix 7; was sum-pool)
- ZYX Euler angles (matches observation space; quaternion→axis-angle path removed)
- Auxiliary distance-prediction head: trains encoder to predict `(pos_dist, rot_dist)` — geometric inductive bias (Fix 13: intentionally kept)

#### PPOABC (`algorithms/rl/ppo/ppo_abc.py`)

- Total Bob loss per mini-batch: `L = L_PPO + β·L_ABC (last batch) + aux_coef × L_aux`
- `epoch_bc_loss` computed once per epoch before mini-batch loop; added to last mini-batch loss → single `optimizer.step()` (Fix 5)
- `detach_goal_encoder=False` during ABC — GoalEncoder receives ABC gradients (Fix 14)
- `aux_coef = 0.1`; `abc_warmup_threshold = 0.0` (ABC active from iteration 1, Fix 9)

#### Actor Network — Bob (`algorithms/rl/ppo/module.py`)

- PermInvEncoder (PI): `Linear(14→512) → LN → ReLU → Linear(512→512) → LN → ReLU`, max-pool, post-pool LayerNorm
- GoalEncoder output injected additively into first trunk layer: `h1 = ReLU(LN(W·enc + Wg·g_pooled))`
- `_goal_proj = Linear(8→512, bias=False)` scaled ×0.1 at init (prevents ReLU saturation)
- LSTMCell(128→256) → MultiCategorical(6 dims × 11 bins)
- `num_cat_dims` default: 6 (Fix 12; was 4)

#### Environment & Tasks

- `AsyncDualPlayEnvWrapper` (`tasks/utils/wrapper.py`): phase management, goal validation, reward, ABC buffer
- `AsyncDualPlayCuRoboEnvCfg` (`tasks/async_dual_play_curobo.py`): scene with `JointPositionActionCfg`
- Alice per-step rewards: **0.0 by default** (Fixes 3 & 11).  **Fix P31**: `--diag_alice_shaping` adds per-step EE→object proximity reward (`0.005 × clamp(0.3 − ‖ee−obj‖, 0, 0.3)`) to give Alice deliberate approach actions for ABC bootstrapping.  Max cumulative contribution ≤0.14/phase via GAE vs 5.0+ ASP outcome rewards — the adversarial curriculum remains dominant.
- T-block (`t_shape.usda`) as sole task object, scale (2.0, 2.0, 1.5), spawn position (0.0, 0.5, 0.05)
- Goal ghost matches T-block shape (no random spawn function)
- Goal validation: z_max=0.05 rejects airborne goals; out_of_zone goals fully invalid; shallow goals (displacement 0.05–0.10m) valid for Bob but Alice gets −1 penalty instead of +1 (Fix P23)
- Bob early termination: Alice paid +5, ghost hidden, objects random-safe reset
- Bob reward: sparse `{+1/−1/+5}` + phase-end progress reward `r = clamp(0.6·Δpos/init_pos + 0.4·Δrot/init_rot, −1, +1)` paid once at Bob termination. Init errors captured on first Bob step. Gives 100% episode coverage — every Bob trial produces a grade. `bob_timesteps` reduced 200→100 to halve credit-assignment horizon for single-object push task. (Dense per-step delta was tested v2–v5 and reverted per Fix P27.)
- 7 cm XY displacement filter removed from Alice's goal validity check (Fix 10)

#### Diagnostic Test Suite (`diagnostics/`)

| File | Test | Trigger |
|---|---|---|
| `test_reward_pipeline.py` | Test 1: teleport objects to goal, verify SR > 0 | `--test_reward_pipeline` |
| `test_alice_sandbox.py` | Test 2: ValidGoals trend, GoalValidityRate, EntropyCoef=0.05 | Offline TB |
| `test_ppo_abc_balance.py` | Test 3a: abc_coef constant, ABC loss > 0, surrogate finite | Offline TB |
| `test_checkpoint_chain.py` | Test 3b: ABC buffer save/load round-trip | Offline |
| `test_abc_goal_encoder.py` | Test 4a: forward pass + gradient flow | Offline, checkpoint |
| `test_goal_encoder_latent.py` | Test 4b: t-SNE silhouette > 0.15, noise invariance < 0.20 | Offline, checkpoint |

**Run all tests locally:**
```bash
bash asyncDualPlayPPO/diagnostics/run_diagnostics.sh
```

#### HPC Slurm Scripts (`hpc/`)

| Script | Purpose |
|---|---|
| `train_curobo.slurm` | Production cuRobo training run (baseline sparse) |
| `train_curobo_shaping.slurm` | Production cuRobo + `--diag_alice_shaping` (Fix P31 — EE→obj proximity) |
| `train_curobo_large.slurm` | Large-scale (512+ envs) |
| `train_curobo_profile.slurm` | 3-iteration profiler run |
| `diagnostic_tests.slurm` | Full 4-test suite on HPC |
| `test1_ppo_reward.slurm` | Test 1 only |
| `test2_alice_exploration.slurm` | Test 2 only (200 iters, random Bob) |
| `test3_asp_tug_of_war.slurm` | Test 3 only (50 iters, full pipeline) |

#### Supporting Utilities

- `utils/historical_pool.py`: ring buffer of past 5 snapshots, `sample_env_subset()`
- `utils/episode_manager.py`: phase tracking, goal storage, checkpoint support
- `utils/profiler.py`: `TrainingProfiler` with `section()` context manager, `get_section_frac()`
- `utils/goal_validator.py`: `validate_goal()` — minimum displacement threshold check
- `optuna_sweep.py`: Optuna hyperparameter sweep wrapper

---

### 3.3 Architecture ↔ Hardware-in-the-Loop Mapping

#### Test 1 → Reward Pipeline / Perception Layer

**What it checks:** `_compute_bob_sparse_rewards` fires at correct thresholds when objects are
teleported to exact goal coordinates.

**HIL relevance:** Physical tracking system must resolve object poses within:
- Position: L2 distance ≤ 0.05 m
- Rotation: max |ZYX Euler diff| with `[0,π]` wraparound ≤ 0.2 rad

#### Test 2 → Curriculum Emergence Layer

**What it checks:** `Metrics/Alice/ValidGoals` trends upward over 200 iterations with random Bob.

**HIL relevance:** `MeanDisp3D` tracks average object displacement — should increase as Alice
learns more complex manipulations. `alice_sandbox` mode isolates Alice's curriculum from Bob.

#### Test 3 → PPO + ABC Optimization Layer

**What it checks:** ABC loss nonzero once buffer is populated; `test_checkpoint_chain.py`
verifies `(obs, acts, old_lp)` tensors survive serialization.

**HIL relevance:** ABC buffer persistence across training interruptions is critical — cold
restart wastes physical robot time.

#### Test 4 → GoalEncoder / Representation Layer

**What it checks:** Forward pass integrity (4a); t-SNE cluster separation, noise invariance (4b).

**HIL relevance:** Noise invariance test (σ=2cm perturbation → relative embedding change < 0.20)
directly simulates camera measurement noise on the physical tracking system.

---

### 3.4 Open Issues & Fix Summary

| # | Issue | Severity | Status | Files Changed |
|---|---|---|---|---|
| 4.1 | Entropy coef mismatch in test (0.01 vs 0.05) | Infra | ✅ Fixed | `diagnostics/test_alice_sandbox.py` |
| 4.2 | Shell scripts wrong entry point | Infra | ✅ Fixed | `diagnostics/run_diagnostics.sh`, `run_diagnostic_tests.sh` |
| 4.3 | GoalEncoder stale axis-angle dead code | Infra | ✅ Fixed | `algorithms/goal_encoder.py` |
| Fix 1 | SR-coupled `abc_coef` | Critical | ✅ Fixed | `train_curobo.py:1247`, `ppo_abc.py:57,62` |
| Fix 2 | SR-coupled Alice entropy | Critical | ✅ Fixed | `train_curobo.py:1234` |
| 4.6 | Test 2 OOZ penalty not verified | Infra | ✅ Fixed | `train_curobo.py`, `diagnostics/test_alice_sandbox.py` |
| 4.7 | Test 4b no-op in CI | Infra | ✅ Fixed | `diagnostics/run_diagnostics.sh` |
| 4.8 | cuRobo <10% overhead not checked | Infra | ✅ Fixed | `utils/profiler.py`, `train_curobo.py` |
| Fix 3 | Dense potential shaping for Alice | Critical | ✅ Fixed | `tasks/utils/wrapper.py:963-966` |
| Fix 4 | Dense potential shaping for Bob | Critical | ✅ Fixed | `tasks/utils/wrapper.py:1043,1113` |
| Fix 5 | ABC as separate backward pass | Critical | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:127-132,276-281` |
| Fix 6 | GoalEncoder architecture ≠ paper | High | Intentional | — |
| Fix 7 | Sum-pool → max-pool for goal embedding | High | ✅ Fixed | `algorithms/rl/ppo/module.py:429` |
| Fix 8 | EMA joint smoothing (`_JC_ALPHA` 0.2→1.0) | Medium | ✅ Fixed | `train_curobo.py:715` |
| Fix 9 | `abc_warmup_threshold` gate | Medium | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:62` |
| Fix 10 | 7cm XY displacement filter removed | Medium | ✅ Fixed | `tasks/utils/wrapper.py:576` |
| Fix 11 | Alice physics penalties (covered by Fix 3) | Medium | ✅ Fixed | `tasks/utils/wrapper.py:963-966` |
| Fix 12 | `num_cat_dims` default 4→6 | Medium | ✅ Fixed | `algorithms/rl/ppo/module.py:186` |
| Fix 13 | Aux loss head on GoalEncoder | Low | Intentional | — |
| Fix 14 | GoalEncoder `detach=True→False` during ABC | Low | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:99` |
| Fix 15 | `ppo.py log()` crash on MultiCategorical | Low | ✅ Fixed | `algorithms/rl/ppo/ppo.py:232-235` |
| Fix 16 | KL adaptive LR dead code in MC mode | Low | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:176` |
| Fix 17 | EE home offset after every sync | Low | ✅ Fixed | `train_curobo.py:75-80,730-731,1048-1049` |
| 4.9 | Charlie hierarchical controller | — | Future research | — |
| 4.10 | Physical sim-to-real interface | — | Future hardware | — |
| Fix 18 | out_of_zone goals accepted as valid | Medium | ✅ Fixed | `utils/goal_validator.py:140` |
| Fix 19 | No z_max check — airborne T-block goals accepted | High | ✅ Fixed | `utils/goal_validator.py:85-88`, `tasks/utils/wrapper.py:158-163` |
| Fix 20 | Bob off-table termination: ghost not hidden, Alice unpaid, objects not random-reset | High | ✅ Fixed | `tasks/utils/wrapper.py:393-406` |
| Fix 21 | ABC debug per-step prints spamming logs | Low | ✅ Fixed | `train_curobo.py:1210-1238` |
| Fix 22 | T-block task space: single T-block object, EE home Y=0.50 | Medium | ✅ Fixed | `tasks/async_dual_play_diffik.py:165-207`, `train_curobo.py:79,730-731,1048-1049` |
| Fix 23 | Alice IK fail: no penalty or termination — arm gets stuck in fail-loop | High | ✅ Fixed | `train_curobo.py:1015-1043` |
| Fix 24 | Trivially-easy goals: Bob starts within success threshold → instant win | High | ✅ Fixed | `tasks/utils/wrapper.py:652-678` |
| Fix P1 | Push-PPO: trunk Linear layers inited with gain=0.01 — activations ~100× too small, policy gradient dead | Critical (Push) | ✅ Fixed | `algorithms/rl/ppo/module_push.py:79-86` |
| Fix P2 | Push-PPO: rotation success threshold 0.035 rad (2°) with uniform [0,2π] goal yaw — 1.1% success window, completion bonus unreachable | High (Push) | ✅ Fixed | `tasks/utils/wrapper_push.py:22` |
| Fix P3 | Push-PPO: LSTM hidden zeroed before every push — LSTM stateless within episode, cannot learn multi-push sequences | Medium (Push) | ✅ Fixed | `train_push.py:383-389` |
| Fix P4 | Push-PPO: mid-trajectory physics termination auto-resets env; compute_push_reward then sees post-reset obs → garbage reward enters GAE | High (Push) | ✅ Fixed | `train_push.py:460-462` |
| Fix P5 | Push-PPO: sr_buf maxlen=200 vs 1024 pushes/iter — only last ~6 push_steps sampled, SR metric systematically undercounts | Low (Push) | ✅ Fixed | `train_push.py:340` |
| Fix P6 | Push-PPO: 11-bin action space → 0.06m minimum push_delta, coarser than 0.05m success threshold — precision literally impossible | Critical (Push) | ✅ Fixed | `train_push.py:143,283`, `action_push.py:decode_push_action` (num_bins=21) |
| Fix P7 | Push-PPO: dim 4 (yaw) decoded to [-π,π] but silently dropped by waypoint generator — 1/6 of policy capacity wasted; no direct rotation control | Critical (Push) | ✅ Fixed | `action_push.py:41,125-143` (Phase 4 yaw quat interp), `train_push.py:409` (pass yaw) |
| Fix P8 | Push-PPO: max_pushes_per_episode=3 → no room to recover from bad push or commit to precision approach; "make progress" the only rational strategy | Medium (Push) | ✅ Fixed | `train_push.py:130` |
| Fix P9 | Push-PPO: max_yaw=π in decode_push_action → 0.314 rad/bin (coarser than 0.2 rad threshold) AND IK forced into elbow-forward branch >~0.8 rad; reduced to 1.0 rad for 0.1 rad/bin precision and elbow-up branch stability | High (Push) | ✅ Fixed | `action_push.py:175,211` |
| Fix P10 | Push-PPO: decode_push_action default num_bins=11 mismatched train_push.py override (21); PushConfig had duplicate `num_bins` line | Low (Push) | ✅ Fixed | `action_push.py:172,222-223` |
| Fix P11 | Push-PPO: no checkpoint resume — iteration always restarted at 0; optimizer state (momentum) lost on load | Medium (Push) | ✅ Fixed | `train_push.py:84-87,341`, `ppo.py:109-123` |
| Fix P12 | Push-PPO: ent_coef=0.01 too high vs reward scale — entropy bonus (−0.18) dominates surrogate loss (~±0.02), no gradient signal to drive policy toward rewards. γ=0.998 gives GAE horizon ~19 steps → all 32 rollout pushes look equally important, no credit assignment. | Critical (Push) | ✅ Fixed | `train_push.py:139-140` (ent_coef=0.002, γ=0.95), `train_push.py:345,483,575,594,604` (+RotationSR metric) |
| Fix P13 | Push-PPO: LSTM amnesia in `evaluate()` — zero-init hidden state for pushes 2–5 creates spurious `π_new/π_old` ratio driven by memory state mismatch, not weight change. PPO clip fires on 80% of transitions regardless of update quality → gradient collapse. | Critical (Push) | ✅ Fixed | `module_push.py:128,139-153` (return h_in, accept hidden_state), `storage.py:73-75,92-101,120-126` (store/yield hidden), `ppo.py:352-357` (slice+pass), `train_push.py:395,476` (capture+store) |
| Fix P14 | Push-PPO: position-only completion bonus teaches policy rotation doesn't matter — agent gets +5 for matching position and ignores rotation entirely. | Critical (Push) | ✅ Fixed | `wrapper_push.py:24,224-232` (+2 rotation sub-bonus gated on pos AND rot), `wrapper_push.py:109,131,292,305` (_gave_rot_bonus buffer) |
| Fix P15 | Push-PPO: `max(|roll|,|pitch|,|yaw|)` for rotation reward tracks wobble instead of yaw during translation — tipped block's roll/pitch contaminates the dense improvement signal. | High (Push) | ✅ Fixed | `wrapper_push.py:62-75` (_yaw_distance_rad), `wrapper_push.py:197-198,212` (y_prev/y_now used for rot_imp) |
| Fix P16 | Push-PPO: tipped blocks are unrecoverable but no termination — subsequent pushes waste episode budget, polluted transitions enter PPO buffer. | High (Push) | ✅ Fixed | `wrapper_push.py:30` (TIP_OVER_THRESHOLD), `wrapper_push.py:241-243` (−5 penalty), `wrapper_push.py:286-288` (check_done tip-over) |
| Fix P17 | Push-PPO: positional penalty exists (−0.5 × d_now) but no rotational urgency — agent has no continuous pressure to fix yaw, can loiter after achieving position. | Medium (Push) | ✅ Fixed | `wrapper_push.py:28,214` (PUSH_DENSE_ROT_BETA=0.25, −β_rot × yaw_err) |
| Fix P18 | Push-PPO: reward coefficients too small → GAE advantage signal near zero → PosErr frozen at 0.25m for 500 iterations despite rotation learning (RotSR 12%→36%). Scaling α (10→12, 1.2×) and γ (2→5, 2.5×) widens reward gap 2.4× with penalties held constant. Fresh-run-safe: initial value loss ~12 vs 52+ at α=30. | Critical (Push) | ✅ Fixed | `wrapper_push.py:25-26` (PUSH_DENSE_ALPHA=12, PUSH_DENSE_ROT_ALPHA=5) |
| Fix P19 | ASP: `evaluate()` missing `hidden_state` → `TypeError` crash — `ppo.py` was updated to pass it (Fix P13 pattern) but `module.py` never accepted the kwarg. Same LSTM amnesia bug as Push-PPO Fix P13 applied to ASP. Pre-action hidden states now captured, zeroed for hist/non-active envs, stored in RolloutStorage, yielded during PPO mini-batches, passed to `evaluate()`. `ppo_abc.py.update()` also updated to retrieve hidden states. | Critical (ASP) | ✅ Fixed | `module.py:602-623`, `ppo_abc.py:164-176`, `train_curobo.py:876-879,924-932,1216,1254,1277` |
| Fix P20 | ASP: `bob_pos_err_buf` / `bob_rot_err_buf` defined but never populated → PosError/RotError TensorBoard metrics always 0. Now filled from `ep_info["bob_pos_err"]`/`ep_info["bob_rot_err"]` for finished-Bob envs. New `bob_pos_sr_buf` / `bob_rot_sr_buf` track position-only and rotation-only SR independently (matches push baseline). | Medium (ASP) | ✅ Fixed | `train_curobo.py:1207-1216` |
| Fix P21 | ASP: No Alice orientation-change tracking — impossible to know if curriculum shifts from pure translation to rotation manipulation. Per-axis `[AliceRot] roll/pitch/yaw` tracking added, computed as `_euler_diff_per_axis(start_ori, goal_ori)` on each Alice phase end. Logged to TensorBoard and iteration summary. | Medium (ASP) | ✅ Fixed | `train_curobo.py:482-490,1308-1316` |
| Fix P22 | Log analyzer missing new ASP metrics — `analyze_training.py` now parses `[AliceRot]` and `[BobSR]` lines, writes to CSV, and plots in both combined-overview and separate-plot modes. | Low | ✅ Fixed | `logs/analyze_training.py` |
| Fix P23 | ASP: `ent_coef=0.05` entropy bonus (~0.7) dominated Alice's surrogate loss (~0.02), keeping her random and unable to break out of 70–87% not-moved rate. Reduced to 0.005 — entropy contribution now ~0.07 (3.5× surrogate instead of 35×). Both Alice and Bob benefit since they share `ppo_continuous.yaml`. | Critical (ASP) | ✅ Fixed | `cfg/ppo/ppo_continuous.yaml:25` |
| Fix P24 | ASP: Alice rewarded +1 for valid goals with minimal displacement (0.06m) and +5 when Bob fails — could farm +6 from micro-nudges. Added minimum-displacement penalty: goals with max displacement 0.05–0.10m get −1 "shallow" penalty (still valid for Bob's practice). Alice must move objects >0.10m to earn +1. Reward ladder: off-table −3 / not-moved 0 / shallow −1 / out-of-zone −3 / valid +1. | Critical (ASP) | ✅ Fixed | `utils/goal_validator.py:125-175`, `tasks/utils/wrapper.py:598` |
| Fix P25 | ASP: Bob received only sparse {+1/−1/+5} rewards with zero per-step feedback — impossible to learn at 35 env scale. Added per-step potential-based delta reward `R = Φ(s') − Φ(s)` with `Φ(s) = −(pos_err + 3.0·yaw_err)`, scaled by 5.0. Strict delta-only — no constant per-step penalty. If Bob doesn't move, reward = 0. Iterated through v2 (value explosion), v3 (scaled down), v4 (penalty drain), v5 (too small). Final v5 form: meaningful gradient (~±0.06–0.28 per step), no explosion, no stationary drain. | Critical (ASP) | ❌ **REVERTED 2026-05-19** — see Fix P27 | |
| Fix P26 | ASP: Bob couldn't control object rotation because EE tilt range was limited to ±0.05 rad/step (2.9°). `max_delta_rot` 0.05→0.10 rad/step (5.7°) and Rx/Ry clamp 0.05→0.10 rad. Bob now has 2× the per-step torque authority to rotate objects through contact. `BOB_DENSE_ROT_WEIGHT` set to 3.0 (vs position 5.0) so rotation carries meaningful weight in Φ(s). | High (ASP) | ✅ Fixed | `train_curobo.py:279,301-303`, `tasks/utils/wrapper.py:45` |
| **Fix P27** | **Bob dense delta reward reverted** — the per-step `Φ(s') − Φ(s)` signal was zero-mean noise at 35-env scale, diluting GAE advantages and killing gradient flow for both agents. Sparse-only `{+1/−1/+5}` restored. | **Critical (ASP)** | ✅ Fixed | `tasks/utils/wrapper.py` (removed ~150 lines: `BOB_DENSE_POS_SCALE`, `BOB_DENSE_ROT_WEIGHT`, `_compute_bob_dense_reward()`, state tracking, step logging) |
| **Fix P28** | **Phase-end progress reward for Bob** — mirrors Alice's episodic feedback: `r_progress = clamp(w_pos·(init−final)/init + w_rot·(init−final)/init, −1, +1)`, paid once at Bob termination. Init errors captured on Bob's first step via `_compute_bob_sparse_rewards`. Progress computed in `_handle_bob_completion` and early-success path. `bob_timesteps` 200→100 halves credit-assignment horizon for single-object T-block task. | **Critical (ASP)** | ✅ Fixed | `tasks/utils/wrapper.py` (+40 lines: `bob_init_pos_err`, `bob_init_rot_err`, `_bob_progress_captured`, init capture, progress computation, reward injection); `cfg/task/AsyncDualPlay.yaml:15` |
| Fix P29 | Push-PPO: critic output layer `gain=1.0` → initial V≈±5–10, GAE chain-reacts at 512 envs → Val loss 27k–357k. Reduced to `gain=0.01` — initial V≈0.057, GAE stable. | Critical (Push) | ✅ Fixed | `module_push.py:83` (_critic_out gain 1.0→0.01) |
| Fix P30 | Push-PPO: PhysX glitches launch object to Z=1863m → single-step reward spikes of −3400/−81600 → critic permanently destroyed. Reward components clamped: `pos_imp∈[−5,5]`, `rot_imp∈[−4,4]`, `penalty∈[−2,0]`, `rot_penalty∈[−1,0]`. `check_done` kills env if `d_now > 0.5`m (out-of-bounds). | Critical (Push) | ✅ Fixed | `wrapper_push.py:219-223,272-274` |
| **Fix P31** | **ABC deadlock diagnosed** — Alice learns to move objects (not-moved 73%→21%, avg disp 0.09→0.16m) but her **actions** remain high-entropy random walks because the entropy bonus (`0.005 × H ≈ 0.06`) dominates her surrogate loss (`~|0.005|`). ABC computes `bc_ratio = exp(lp − old_lp) ≈ 1.0` for all 1081 iterations because Alice's random action sequences provide no consistent gradient direction for Bob to clone. Bob's PPO gradient is zero (sparse rewards, SR 1–3%). Bob's value function converges to predict ~0 (val loss 0.02–0.06). Net result: Bob's policy stays at random initialization forever. **Fix**: `--diag_alice_shaping` (EE→object proximity reward: `0.005 × clamp(0.3 − ‖ee−obj‖, 0, 0.3)` per step) gives Alice deliberate approach actions, providing structured demonstrations for ABC to bootstrap Bob. The shaping is ≤3% of ASP outcome rewards (max 0.14/phase via GAE vs 5.0+ from Bob-fail bonus), so the adversarial curriculum remains dominant. New HPC script: `hpc/train_curobo_shaping.slurm`. | **Critical (ASP)** | ✅ Fixed | `train_curobo.py:1130-1135` (shaping already wired), `hpc/train_curobo_shaping.slurm` (new) |
| **Fix P32** | **Push-PPO rollouts too slow** — 115 substeps per push (3,680 sequential physx+IK steps per iteration at 32 pushes). Rollouts dominated wall-clock, making training impractical at scale. Substeps scaled 115→76 (~1.5× faster). CUDA-synced wall-clock profiler added to `train_push.py` to identify remaining bottlenecks: `agent`, `decode`, `ik`, `physics`, `reward`, `store`, `ppo`. | **High (Push)** | ✅ Fixed | `action_push.py:16-24`, `train_push.py:41-42,372-393,435+` |
| **Fix P33** | **cuRobo IK dominates push training** — profiler showed `solve_batch` at 65ms/call, 69% of iteration wall-clock. Default solver config had `n_iters=100, inner_iters=25` per env. LBFGS reduced to `n_iters=30, inner_iters=10` — IK dropped 65→18ms/call (3.6×), total iteration 232→116s (2×). `n_problems=1` (sequential env loop) could not be changed without breaking CUDA graph shapes. | **High (Push)** | ✅ Fixed | `train_push.py:211-214` |
| **Fix P34** | **4D action space** — redesigned from 6D (offset_x/y, push_dx/dy, yaw, push_dz) to 4D macro-params (Xs, Ys, length, theta). Xs/Ys = absolute push start in world coords; push endpoint Xf=Xs+len·cosθ, Yf=Ys+len·sinθ. Gripper always closed — engage/release phases removed. Waypoints: approach→descend→push→retract→return (72 substeps, down from 76). Actor head 126→84 dims. All test scripts updated. | **Critical (Push)** | ✅ Fixed | `action_push.py`, `train_push.py`, `module_push.py`, `test_push_primitive.py`, `test_spin.py`, `validate_push.py` |
| **Fix P35** | **Push debug markers** — green sphere at (Xs,Ys), red sphere at (Xf,Yf), blue cylinder arrow connecting them. Three independent `VisualizationMarkers`, updated every push. `_update_push_markers()` wrapped in try/except for safety. | **Low** | ✅ Fixed | `train_push.py:269-334` |
| **Fix P36** | **Per-env debug logging** — when `num_envs ≤ 5`, each push logs per-env bin indices and decoded params: bins=(10, 12, 8, 5) Xs=+0.00 Ys=+0.57 len=0.06 θ=45° → Xf=+0.04 Yf=+0.61. | **Low** | ✅ Fixed | `train_push.py:238,549-556` |
| **Fix P37** | **Length limit** — push length clamped to [0, 0.20] m (was [0, 0.30]). Tightens action space, preventing over-aggressive pushes. | **Medium (Push)** | ✅ Fixed | `action_push.py:130,152,158` |
| **Fix P38** | **Profiler removed** — inline CUDA-synced profiler removed from `train_push.py`. Replaced by per-push markers (Fix P35) and per-env debug logging (Fix P36) for visibility. `import time` also removed. | **Low** | ✅ Fixed | `train_push.py` |
| **Fix P39** | **Gripper removed from push observation** — gripper is always closed in push primitive, carries no useful signal. `_OBS_ROBOT_DIM` 7→6, total obs 29D→28D. Removed `gripper_pos` ObsTerm from `PushPolicyCfg`. | **High (Push)** | ✅ Fixed | `wrapper_push.py:37-43`, `push_task_curobo.py:39-44,50-55`, `module_push.py:8`, `net.md`, `README.md` |
| **Fix P40** | **Waypoint loop ignores death** — after `env.step()` auto-resets terminated env, subsequent waypoint IK targets teleport robot back to table, causing object to explode to 3000m. Terminated envs now hold `cur_joints` instead of executing new waypoint targets. | **Critical (Push)** | ✅ Fixed | `train_push.py:564-567` |
| **Fix P41** | **Exploded state saved into PPO buffer** — `needs_reset = done & ~terminated` skipped explicit reset for terminated envs; `obs` held post-explosion 3000m values captured by `obs_pre_push = obs.clone()` for next push. Now `needs_reset = done`. | **Critical (Push)** | ✅ Fixed | `train_push.py:663` |
| **Fix P42** | **Zero penalty for terminated envs** — `reward[terminated] = 0.0` gave no signal for off-table/exploded pushes. Changed to −10.0 so agent learns to avoid them. | **Critical (Push)** | ✅ Fixed | `train_push.py:582` |
| **Fix P43** | **Dynamic minibatches** — `nminibatches` now derived from `num_envs` via `max(1, num_envs // 16)`, with a divisibility fallback loop. Keeps mini-batch size ~240 transitions regardless of env count so GPU memory scales predictably. `push_nsteps` stays fixed at 15 (LSTM temporal depth); only breadth scales. | **Medium (Push)** | ✅ Fixed | `train_push.py:149-153` |
| **Fix P44** | **Elbow-IK no longer terminates episode** — elbow-negative IK solutions set `terminated=True`, causing a −10 penalty for trivial IK-unreachable (Xs,Ys) pairs. The existing fallback `ik_ok[elbow_bad]=False` already holds `prev_joint_cmd` safely. Removed `terminated[elbow_bad]=True` — bad pushes get static penalty, not death. Gives PPO a smooth gradient away from unreachable workspace. | **Critical (Push)** | ✅ Fixed | `train_push.py:567-572` |

**Proposed additional tests (not yet implemented):**

| Test | Validates | Priority |
|---|---|---|
| A — Alice proposer collapse + MeanDisp3D floor | Fixes 2, 3, 10, 11 | Medium |
| B — PPO & ABC gradient norm ratio | Fixes 5, 1 | Medium |
| C — Unfrozen GoalEncoder aux-loss spike detection | Fixes 14, 13 | Low |
| D — Hardware jitter / joint acceleration (UR5e gate) | Fix 8 | **Critical before UR5e deployment** |
| E — Max-pool object saturation with distractors | Fix 7 | Medium |

---

### 3.5 Key Hyperparameter Reference (`cfg/ppo/ppo_continuous.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `optim_stepsize` | 3e-4 | LR for both Alice and Bob |
| `alice_lr_min` | 5e-5 | Cosine decay floor for Alice |
| `cliprange` | 0.2 | PPO ε-clip (paper Table 2) |
| `noptepochs` | 3 | PPO mini-epochs |
| `nminibatches` | 4 | PPO minibatches per epoch |
| `ent_coef` | **0.005** | Alice/Bob entropy coef (fixed; was 0.05 — Fix P23) |
| `gamma` | 0.998 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `abc_coef` | **0.5** | Bob BC loss weight (fixed, Fix 1) |
| `abc_traj_maxlen` | 500 | ABC trajectory store capacity |
| `abc_n_trajs` | 16 | Trajectories sampled per Bob update |
| `aux_coef` | 0.1 | GoalEncoder auxiliary distance loss |
| `goal_embed_dim` | 8 | GoalEncoder latent K |
| `num_bins` | 11 | Bins per MultiCategorical dimension (ASP; Push baseline uses 21 — Fix P6) |
| `num_cat_dims` | 6 | Action dims: X, Y, Z, Rx, Ry, Gripper |
| `lstm_hidden_size` | 256 | LSTM hidden state size |
| `alice_timesteps` | 100 | Steps per Alice phase |
| `bob_timesteps` | **100** | Steps per Bob phase (was 200; halved per Fix P28 — single-object push doesn't need multi-stage stacking budget) |
| `_EE_HOME_X_OFFSET` | 0.02 m | X offset added to IK target after every sync (home pose) |
| `_EE_HOME_Y` | 0.50 m | Fixed Y of IK target after every sync — directly over T-block spawn |
| `_EE_HOME_Z` | 0.05 m | Fixed Z of IK target after every sync (5 cm above table) |
| `BOB_PROGRESS_W_POS` | 0.6 | Phase-end progress weight for position (Fix P28) |
| `BOB_PROGRESS_W_ROT` | 0.4 | Phase-end progress weight for rotation |

---

## 4. cuRobo IK Integration <a name="curobo-ik"></a>

### 4.1 How the Three Controllers Compare

| | RMPFlow (`train.py`) | DiffIK (`train_diffik.py`) | cuRobo IK (`train_curobo.py`) |
|---|---|---|---|
| **Policy output** | EE Cartesian delta (MultiCategorical, 6×11 bins) | EE Cartesian delta → DiffIK → Δθ | EE Cartesian delta → cuRobo → θ* |
| **Joint control** | Handled internally by RMPFlow | JointPositionActionCfg | JointPositionActionCfg |
| **IK quality** | RMPFlow (reactive, may drift) | First-order Jacobian pseudo-inverse (poor near singularities) | Batch MPPI/CuTorch solver, singularity-aware, seed-conditioned |
| **Speed per step** | Fast (native Isaac Lab) | Fast + Jacobian solve | Adds GPU IK solve (~1–3 ms/batch) |

### 4.2 cuRobo Batch IK Pattern

```python
goal_pose = Pose(position=ee_targets_local,   # (N, 3)
                 quaternion=fixed_quat.expand(N, 4))
result = ik_solver.solve_batch(goal_pose,
                               seed_config=cur_joints.unsqueeze(1),  # (N,1,6)
                               retract_config=cur_joints)
joint_cmd[:, :6] = torch.where(
    result.success.unsqueeze(-1),
    result.solution.view(N, 6),
    cur_joints          # hold last-good if IK fails
)
```

### 4.3 Pros of cuRobo IK

- **Singularity handling**: manipulability-aware solver avoids degenerate configurations
- **Seed conditioning**: smooth, continuous joint trajectories
- **Guaranteed reachability check**: `result.success` per-env; enables clean fallback
- **Consistent EE-to-joint mapping**: no reactive planning noise from RMPFlow
- **GPU-native batching**: `solve_batch(N)` runs as a single CUDA kernel; overhead <5 ms at 512 envs
- **CUDA graph warm-up**: traces once before training loop → ~0.5 ms per call

### 4.4 Cons and Drawbacks

- **Must be imported before AppLauncher** — hard constraint; handled via the cuRobo import block at the top of `train_curobo.py`
- **IK failure during early training** — Alice IK failures result in immediate episode termination with -1 penalty (arm locked in place). Bob IK failures are non-terminal (hold position). This provides a hard workspace-bounds signal that the policy learns to avoid.
- **Orientation fixed to "tool pointing down"** — Option B chosen; reduces IK failures for tabletop manipulation
- **CUDA graph warm-up adds ~30s to startup** — use warm-up call before training loop:
  ```python
  _warmup_pose = Pose(position=torch.zeros(num_envs, 3, device=device),
                      quaternion=fixed_quat.expand(num_envs, 4))
  ik_solver.solve_batch(_warmup_pose,
                        seed_config=torch.zeros(num_envs, 1, 6, device=device),
                        retract_config=torch.zeros(num_envs, 6, device=device))
  ```
- **Memory overhead**: ~400–800 MB VRAM at N=512 (fits on RTX Pro 6000)
- **HPC requires Apptainer overlay** — see Section 5

### 4.5 Relation to the ASP + ABC Training Loop

The Alice/Bob phase structure, ABC buffer, historical pool, GoalEncoder, and PPOABC loss are
**controller-agnostic** — they operate on observations and rewards, not joint angles. Only these
files needed modification for the cuRobo switch:
1. `train_curobo.py` — new entry point
2. `tasks/async_dual_play_curobo.py` — `JointPositionActionCfg`
3. `hpc/train_curobo.slurm` — SLURM script with overlay

### 4.6 Human vs Model Evaluation Script

`tests/test_human_vs_bob.py` — side-by-side evaluation with `num_envs=2`.

- Same goal injected into both arenas
- Human drives arena 0 via gamepad + cuRobo IK; loaded Bob drives arena 1
- `--chkpt_alice` optional: Alice proposes the goal for `--alice_steps` steps (default 100)

```bash
python tests/test_human_vs_bob.py \
    --chkpt_bob  runs/my_run/bob/model_500.pt \
    --chkpt_alice runs/my_run/alice/model_500.pt \
    --num_objects 2 --max_vel 1.0
```

### 4.7 Profiling

`train_profile.slurm` runs 3 iterations at 2048 envs. Expected profiler output:

```
Section          |  calls |   total(s) |   mean(ms) |    max(ms)
curobo_ik        |      3 |      0.042 |      14.00 |      16.2
env_step         |      3 |      1.821 |     607.00 |     621.0
alice_act        |      3 |      0.003 |       1.00 |       1.1
bob_act          |      3 |      0.003 |       0.98 |       1.0
```

`curobo_ik` should remain well under 10% of `env_step` time.

---

## 5. HPC Setup & Run Guide <a name="hpc-setup"></a>

### 5.1 Installing cuRobo Locally

```bash
# 1. Activate the Isaac Lab environment
source /home/vlad/env_isaaclab/bin/activate

# 2. Verify torch/CUDA first
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.7.0+cu128  12.8

# 3. Clone and pin to v0.7.5
git clone https://github.com/NVlabs/curobo.git /tmp/curobo
cd /tmp/curobo
git checkout v0.7.5

# 4. Install (no-build-isolation keeps the existing CUDA 12.8 PyTorch)
pip install -e ".[no_dev]" --no-build-isolation

# 5. Verify
python -c "import curobo; print(curobo.__version__)"
# Expected: 0.7.5
```

### 5.2 One-Time HPC Setup

#### Step 1: Pull the Isaac Lab container

```bash
cd /home/<you>/master_isaac/asyncDualPlayPPO
apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0
```

Takes ~20 minutes; produces `isaac-lab.sif` (~30 GB). Slurm scripts expect it in the project root.

#### Step 2: Build a cuRobo-patched overlay image

```bash
# 2a — Create the overlay (8 GB writable ext3 image)
apptainer overlay create --size 8192 curobo_overlay.img

# 2b — Install cuRobo inside the overlay
apptainer exec --nv --overlay curobo_overlay.img:rw isaac-lab.sif bash
```

Inside the shell:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.7.0+cu128  12.8

git clone https://github.com/NVlabs/curobo.git /tmp/curobo
cd /tmp/curobo && git checkout v0.7.5
pip install -e ".[no_dev]" --no-build-isolation

python -c "import curobo; print(curobo.__version__)"
# Expected: 0.7.5
exit
```

> **Why `--no-build-isolation`?** Build isolation pulls a second PyTorch, mismatching CUDA 12.8 and breaking imports.

#### Step 3: Verify the overlay

```bash
apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    python -c "import curobo; import isaaclab; print('OK')"
```

#### Step 4: Update slurm scripts to use the overlay

In `hpc/train_curobo.slurm` and `hpc/train_curobo_profile.slurm`, change:

```bash
apptainer exec --nv \
```
to:
```bash
apptainer exec --nv --overlay "$PROJECT_ROOT/curobo_overlay.img":ro \
```

> Use `:ro` (read-only) at job runtime.

#### Step 5: Create cache directories

```bash
mkdir -p .cache \
         .isaac_cache/kit/data \
         .isaac_cache/kit/cache \
         .isaac_cache/kit/logs
```

#### Step 6: Verify registration

```bash
apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train_curobo.py \
    --num_envs 16 --max_iterations 3 --headless
```

---

### 5.3 Local Training Commands

```bash
source /home/vlad/env_isaaclab/bin/activate
cd /home3/s3426394/master_isaac

# Minimal smoke test
python -m asyncDualPlayPPO.train_curobo \
    --num_envs 16 --max_iterations 500 --exp_name curobo_test --headless

# Full local run
python -m asyncDualPlayPPO.train_curobo \
    --num_envs 512 --nsteps 300 --max_iterations 100000 \
    --save_interval 100 --exp_name curobo_local --headless

# Resume from checkpoint
python -m asyncDualPlayPPO.train_curobo \
    --num_envs 512 --max_iterations 100000 --exp_name curobo_local \
    --resume_path runs/curobo_local/bob/model_1000.pt \
    --resume_path_alice runs/curobo_local/alice/model_1000.pt \
    --resume_iteration 1000 --headless
```

---

### 5.4 HPC Training

#### Quick smoke test (interactive node)

```bash
srun --partition=gpu --gpus-per-node=rtx_pro_6000:1 --time=00:15:00 --pty bash

apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    /workspace/isaaclab/isaaclab.sh -p train_curobo.py \
    --num_envs 64 --max_iterations 10 --headless --exp_name smoke_test
```

Watch for:
- `[cuRobo] IK solver created.` — solver initialised
- `IK fail %` per iteration — should drop below 30% after ~50 iterations
- No `CUDA error` or `out of memory` messages

#### Full production run

```bash
sbatch hpc/train_curobo.slurm
```

Defaults: 4096 envs, 100 000 iterations. Checkpoints every 10 iters to `runs/hpc_curobo_4096env_1obj/`.
On SIGUSR1 (2 min before wall-time), job syncs to NFS and resubmits itself.

```bash
tail -f slurm-<JOBID>-curobo.out
# Key lines per iteration:
# [Iter N] SR=0.12 | IK fail%=18.3 | Alice valid=47/64 | avg XY=0.142m
```

#### Resume from checkpoint

```bash
sbatch hpc/train_curobo.slurm   # auto-detects latest checkpoint

# Or manually:
apptainer exec --nv --overlay curobo_overlay.img:ro isaac-lab.sif \
    /workspace/isaaclab/isaaclab.sh -p train_curobo.py \
    --num_envs 4096 \
    --chkpt_alice runs/hpc_curobo_4096env_1obj/alice/model_500.pt \
    --chkpt_bob   runs/hpc_curobo_4096env_1obj/bob/model_500.pt \
    --resume_iteration 500 --headless
```

---

### 5.5 Troubleshooting

**`ImportError: No module named 'curobo'`**  
→ Overlay not passed to `apptainer exec`. Add `--overlay curobo_overlay.img:ro`.

**`CUDA error: device-side assert triggered`**  
→ Shape mismatch in IK batch. Ensure `IKSolverConfig(..., batch_size=num_envs)` matches `--num_envs`.

**IK fail rate stuck above 50%**  
→ EE target outside reachable workspace. Tighten bounds:
```python
_WS_XY  = 0.5   # reduce if fail rate is high
_WS_Z   = (0.00, 0.60)
```

**`isaac-lab.sif` not found**  
→ Must be in the directory where `sbatch` is called (project root), or set `SIF_IMAGE` to an absolute path.

**Container rebuild (if overlay unavailable)**  
```singularity
Bootstrap: docker
From: nvcr.io/nvidia/isaac-lab:2.3.0

%post
    git clone https://github.com/NVlabs/curobo.git /tmp/curobo
    cd /tmp/curobo && git checkout v0.7.5
    pip install -e ".[no_dev]" --no-build-isolation
    rm -rf /tmp/curobo
```
```bash
apptainer build isaac-lab-curobo.sif isaac-lab-curobo.def
```
Then update `SIF_IMAGE="isaac-lab-curobo.sif"` in both slurm scripts.

---

## 6. Push-PPO Baseline <a name="push-ppo"></a>

**Date**: 2026-05-08

### 6.1 Overview

A **baseline PPO approach** for tabletop pushing: single PPO agent with a **push primitive**
macro-action (no Alice/Bob, no ABC, no goal encoder). The agent predicts push parameters; the
environment executes a multi-step push trajectory using cuRobo IK.

### 6.2 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orientation | Tool-down `[0,1,0,0]` for approach/descend; yaw-rotated during push phase (Fix P7) | Approach is tool-down for clean contact; push phase applies Z-rotation for direct torque control |
| Gripper | Always closed during push | Simplest possible baseline |
| Action space | Relative offsets from object | Generalizes across object positions |
| Action frequency | Macro-action | Agent predicts push params once; env executes multi-step trajectory |
| Pushes per episode | 5 (was 3, Fix P8) | Room to recover from bad pushes and commit to precision finish |

### 6.3 Push Primitive Architecture

```
Phase 1: Approach   (12 steps, was 18)  EE → above object, tool-down, gripper open
Phase 2: Engage     ( 3 steps, was 5)   Close gripper at approach height
Phase 3: Descend    (16 steps, was 24)  EE down to contact height (table + 0.110 m), tool-down
Phase 4: Push       (20 steps, was 30)  EE moves: contact_xy → contact_xy + (push_dx, push_dy), quat interpolates tool-down → tool-down ⊗ RotZ(yaw)  (Fix P7)
Phase 5: Retract    (16 steps, was 24)  EE up to approach height, gripper closed, keeps final yaw quat
Phase 6: Release    ( 1 step, was 2)    Open gripper at approach height
Phase 7: Return     ( 8 steps, was 12)  EE back to current TCP position at approach height, tool-down

Total: 76 substeps per push macro-action (~1.5 s at 50 Hz, was 115 substeps / 2.3 s — Fix P32)
```

### 6.4 Action Space

MultiCategorical: **6D × 21 bins**  (Fix P6: was 11 bins → 0.03m/bin resolution, below 0.05m success threshold)

| Dim | Parameter | Range |
|-----|-----------|-------|
| 0 | `approach_offset_x` | [-0.15, 0.15] m |
| 1 | `approach_offset_y` | [-0.15, 0.15] m |
| 2 | `push_dx` | [-0.30, 0.30] m |
| 3 | `push_dy` | [-0.30, 0.30] m |
| 4 | `yaw` | [-1.0, 1.0] rad — **EE Z-rotation during push phase** (Fix P7: was dead dim; Fix P9: was ±π, reduced to ±1.0 rad for 0.1 rad/bin precision and elbow-up IK branch) |
| 5 | `push_dz` | [-0.03, 0.03] m |

### 6.5 Observation Space (29D)

```
[ee_pos(3) | ee_euler(3) | gripper(1) | obj_pos(3) | obj_euler(3) | obj_linvel(3) |
 obj_angvel(3) | ee_obj_dist(1) | obj_contact(1) | goal_pos(3) | goal_euler(3) |
 pos_dist(1) | rot_dist(1)]
```

### 6.6 Reward Structure

Dense shaping computed **after each push macro-action**:

```
R = α·(d_prev−d_now)                                                            position improvement
  + γ·(y_prev−y_now)                                                            yaw-only rotation improvement (Fix P15)
  − β·d_now                                                                     distance penalty
  − β_rot·y_now                                                                 continuous yaw penalty (Fix P17)
  + completion_bonus           +5  for pos < 0.05                               position gate (keeps SR floor)
  + rotation_sub_bonus          +2  for pos < 0.05 AND rot < 0.2                rotation polish (Fix P14)
  − tip_penalty                 −5  if abs(roll) > 0.3 or abs(pitch) > 0.3      tip-over gate (Fix P16)
```

where:
- `d_prev` / `d_now` = L2 position error before / after the push (metres)
- `y_prev` / `y_now` = yaw-only Euler difference before / after (radians, wraparound-aware) — isolates Z-axis rotation from roll/pitch wobble (Fix P15)
- `α = 12.0` — position improvement gain (symmetric: rewards getting closer, penalizes moving away) — 1.2× scaled (Fix P18)
- `γ = 5.0` — rotation improvement gain — 2.5× scaled (Fix P18)
- `β = 0.5` — distance penalty per step
- `β_rot = 0.25` — continuous yaw penalty per step (Fix P17)
- `completion_bonus = +5.0` when object enters goal zone (pos < 0.05 m) — position-only gate preserves 5.7% SR floor
- `rotation_sub_bonus = +2.0` when position AND rotation both match (pos < 0.05 m AND rot < 0.2 rad) — priority-driven curriculum: primary spatial → secondary rotation (Fix P14)
- `tip_penalty = −5.0` when object is tipped (|roll| > 0.3 or |pitch| > 0.3 rad) — episode also terminates early (Fix P16)

**Design rationale**: off-center pushes induce torque via the `(offset_x, offset_y, yaw)` parameters.
The symmetric improvement terms prevent reward hacking. The position gate keeps the bonus accessible
at 5–6% event rate so GAE can propagate it. The rotation sub‑bonus creates a curriculum: once the
agent reliably reaches the zone, the +2 teaches it to also match orientation. The tip‑over penalty
prunes unrecoverable states from the PPO buffer, preventing batch pollution.

### 6.7 Network Architecture

```
obs (29D)
  │
  ├─ Linear(29 → 512) → ReLU     ← orthogonal init gain=sqrt(2) (Fix P1)
  ├─ Linear(512 → 256) → ReLU    ← orthogonal init gain=sqrt(2) (Fix P1)
  ├─ Linear(256 → 128) → ReLU    ← orthogonal init gain=sqrt(2) (Fix P1)
  ├─ LSTM(128 → 256)             ← hidden propagates across pushes within episode (Fix P3)
  │
  ├─ Actor head:  Linear(256 → 126) → (6, 21) → MultiCategorical  ← gain=0.01 (Fix P6)
  └─ Critic head: Linear(29 → 512) → ReLU → Linear(512 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 1)
```

**Weight init**: trunk layers use `orthogonal_(gain=sqrt(2))` for ReLU activations; actor head uses `gain=0.01` only (Fix P1). Previously gain=0.01 was applied to all layers, making activations ~100× too small and killing gradient signal.

**LSTM sequencing**: hidden state propagates push-to-push within an episode; zeroed only at episode done boundaries. Hidden states are stored at rollout time and yielded during PPO mini‑batch updates so `evaluate()` recomputes action log‑probs with the correct temporal context — the ratio `π_new/π_old` reflects genuine weight changes, not LSTM amnesia (Fix P13).

### 6.8 Files

| File | Purpose |
|------|---------|
| `tasks/push_task_curobo.py` | Environment config (single agent, single object) |
| `tasks/utils/wrapper_push.py` | Push env wrapper: macro-action execution, reward, reset |
| `tasks/utils/action_push.py` | Push primitive: trajectory generation + cuRobo IK |
| `algorithms/rl/ppo/module_push.py` | Simplified ActorCritic (flat MLP + LSTM) |
| `train_push.py` | Training script |
| `tests/validate_push.py` | Validation script |
| `tests/test_push_primitive.py` | Interactive scenario-loop test |
| `tests/test_spin.py` | Yaw-rotation spin test |

### 6.11 Yaw Rotation Quaternion (Fix P7)

The push phase quaternion interpolates from tool-down to a Z-rotated variant.
Tool-down quaternion `q_td = (0, 1, 0, 0)` (wxyz) composed with Z-rotation by angle `a`
produces `q = (0, cos(a/2), sin(a/2), 0)`. This is computed directly per waypoint
in `action_push.py:125-137` without explicit quaternion multiplication:
```python
half = (alpha * yaw) * 0.5
quat = (0, cos(half), sin(half), 0)  # tool-down ⊗ RotZ(alpha × yaw)
```
Phase 5 (retract) keeps the final yaw quat to avoid snap-back while near the object.
Approach, descend, release, and return phases stay tool-down throughout.

### 6.12 Profiler (Fix P32 + Fix P33)

Per-iteration CUDA-synced wall-clock profiler built into `train_push.py`.
Tracks 7 sections and prints a compact table each iteration.

**Measured at 64 envs, 76 substeps/push, RTX 3060 Ti (after Fix P33):**

```
[Profiler]     name    tot(s)   calls   ms/call   %iter
[Profiler]  physics    71.548    2432     29.42   61.9%
[Profiler]       ik    43.781    2432     18.00   37.9%
[Profiler]   decode     0.165      32      5.16    0.1%
[Profiler]      ppo     0.072       1     71.98    0.1%
[Profiler]   reward     0.032      32      1.01    0.0%
[Profiler]    agent     0.030      32      0.93    0.0%
[Profiler]    store     0.006      32      0.17    0.0%
[Profiler]    TOTAL   115.633
```

**Before tuning (Fix P32, no IK tuning):**
```
[Profiler]       ik   166.879    2432     68.62   69.1%
[Profiler]  physics    74.025    2432     30.44   30.7%
[Profiler]    TOTAL   241.354
```

IK dropped 65→18ms via LBFGS `n_iters=30, inner_iters=10` (was 100/25).
Physics remains the dominant bottleneck at 62%. Further improvements require
cutting substep count or reducing PhysX complexity. PPO update is negligible
at 0.06% of iteration time.

### 6.9 Running

```bash
# Non-headless (with viewer)
python -m asyncDualPlayPPO.train_push \
    --num_envs 64 --max_iterations 500 --exp_name push_baseline

# Headless
python -m asyncDualPlayPPO.train_push \
    --num_envs 64 --max_iterations 500 --exp_name push_baseline --headless

# Resume from checkpoint (Fix P11)
python -m asyncDualPlayPPO.train_push \
    --num_envs 32 --max_iterations 1000 --exp_name push_baseline_v2 \
    --chkpt runs/push_baseline/agent/model_250.pt \
    --resume_iteration 250 --headless
```

### 6.10 Known Issues & Fixes

#### RTX SceneDB Segfault — IOMMU + NVIDIA 595 + Kernel 6.8.0-111 — 2026-05-11

**Symptom**: Isaac Sim crashes at app startup (~20s) with a segfault in
`librtx.scenedb.plugin.so!carbOnPluginStartup`. The crash occurs during RTX scene database
shader enumeration exactly when the first viewport frame renders (just after `app ready`).

```
001: librtx.scenedb.plugin.so!_M_realloc_insert<std::tuple<char const*, float, float, unsigned int, unsigned int, unsigned int>>
004: librtx.scenedb.plugin.so!carbOnPluginStartup+0x3b4de
008: libcarb.scenerenderer-rtx.plugin.so!carbOnPluginShutdown+0xe4b
```

The crash is a C++ `std::vector` `_M_realloc_insert` corruption — a stale/mangled pointer read
from a GPU-mapped DMA buffer during shader parameter enumeration.

**Root cause — conflicting NVIDIA driver versions**: The `nvidia-driver-595-open 595.58.03` and
`nvidia-driver-580-open 580.142` packages are **both installed simultaneously**. The kernel
module loaded is 595.58.03 (`/proc/driver/nvidia/version` confirms `NVRM version: 595.58.03`),
but all userspace libraries (`libnvidia-compute-*`, `libnvidia-gl-*`, `nvidia-utils-*`) are
at version **580.142**. This kernel/userspace ABI mismatch is confirmed by:

```
$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.142
```

The mismatch causes the NVIDIA Resource Manager (RM) to serve incompatible DMA buffer
mappings to the RTX rendering pipeline — Vulkan memory objects allocated by the 595 kernel
module are interpreted using 580 userspace library semantics, producing stale/corrupted
pointers during `std::vector` reallocation in `librtx.scenedb.plugin.so`.

**Python-level workarounds tested and failed**:

| Flag | Effect |
|---|---|
| `--/app/asyncRendering=false` (both variants) | Forces synchronous RTX rendering — prevents race condition but crash still occurs at first frame shader compilation |
| `--/persistent/exts/omni.kit.viewport.menubar.lighting/autoLightRig/enabled=false` | Prevents `SetLightingMenuModeCommand` from triggering stage traversal — crash still occurs during viewport init |

None of these prevent the crash because the Vulkan shader compilation pipeline itself triggers
the RTX scene DB enumeration unconditionally on the first render frame.

**Verified workaround — headless mode**:
```bash
python tests/test_push_primitive.py --headless
```
Headless mode skips RTX viewport rendering entirely — no scene DB enumeration → no crash.

**System-level fix** (requires sudo — ask your sysadmin):

```bash
# Confirm the conflict:
dpkg -l | grep nvidia-driver-  # shows both 580-open and 595-open installed

# Fix A — purge 595, keep proven 580 (what was working):
sudo apt purge nvidia-driver-595-open nvidia-firmware-595-*
sudo apt install --reinstall nvidia-driver-580-open
sudo reboot

# Fix B — fully upgrade to 595 (remove all 580 packages):
sudo apt purge '.*nvidia.*580.*' && sudo apt autoremove
sudo apt install nvidia-driver-595-open
sudo reboot
```

After either fix, verify:
```bash
nvidia-smi  # should show driver version matching across kernel + userspace
```

**Which tests are affected**:

| Test | Status | Workaround |
|---|---|---|
| `test_push_primitive.py` | ❌ segfaults without `--headless` | Add `--headless` or apply system fix |
| `test_curobo_follow_target.py` | ❌ same crash pattern | Add `--headless` or apply system fix |
| `train_push.py` | ⚠️ affected if run without `--headless` | `--headless` already default for training |
| `train_curobo.py` (HPC) | ✅ unaffected (HPC runs headless in Apptainer) | — |

**Task-local code mitigation**: Both `test_push_primitive.py` and `test_curobo_follow_target.py`
include the async rendering flags + cuRobo-before-AppLauncher import guard. These are necessary
for older driver/kernel combos but insufficient against the 595 + 6.8.0-111 + IOMMU breakage.

**References**:
- Kernel module version: `/proc/driver/nvidia/version` → `NVRM version: 595.58.03`
- Userspace library version: `nvidia-smi` → `NVML library version: 580.142`
- Conflicting packages: `dpkg -l | grep nvidia-driver-` shows both `580-open` and `595-open`
- Diagnostics from `nvidia-bug-report.sh` or equivalent show `RM version mismatch` in NVML init
- The feature flag comments in `tests/test_curobo_follow_target.py:65-70` document the IOMMU + async rendering race condition on RTX 3060 Ti, which is a separate pre-existing issue made worse (not caused) by the driver version conflict

---

#### `--headless` ArgParser conflict — 2026-05-10

`AppLauncher.add_app_launcher_args()` adds `--headless` itself. The original `train_push.py`
also added it manually, causing:

```
ValueError: The passed ArgParser object already has the field 'headless'.
```

**Fix:** removed the manual `parser.add_argument("--headless", ...)` line. `AppLauncher`
supplies it with `default=False`.

---

#### `num_envs` property setter conflict — 2026-05-10

`PushEnvWrapper.__init__` assigned `self.num_envs = env.num_envs` while the class defines a
`@property num_envs` (no setter):

```
AttributeError: property 'num_envs' of 'PushEnvWrapper' object has no setter
```

**Fix (`wrapper_push.py`):**
- Removed the `self.num_envs = env.num_envs` assignment; the property already delegates to `self.env.num_envs`.
- Swapped ordering in `reset_done_envs`: snapshot `push_count`/`at_goal` into episode logs *before* resetting them to zero.

---

#### Env reset correctness — 2026-05-10

Two places called `env.env.reset()` (no `env_ids`), resetting **all** parallel environments
instead of only the finished ones.

`ManagerBasedRLEnv.step()` auto-resets terminated envs and recomputes `obs_buf` after the
reset, so `obs` returned from `PushEnvWrapper.step()` already contains post-reset observations.

**Fix (`train_push.py`):**
1. Initialize `terminated = zeros(bool)` before the waypoint loop; accumulate `terminated |= step_terminated` each substep.
2. Removed mid-trajectory `if terminated.any(): env.env.reset()` block entirely.
3. In the post-push done block, replaced `env.env.reset()` with `env.env.reset(env_ids=reset_ids)` for only `done & ~terminated` envs.

---

#### Goal ghost placed at world origin instead of on table — 2026-05-10

**Fix (`wrapper_push.py`):**
- `_sample_goals(env_ids)` takes explicit env ids, writes into `self.goal_pos_euler[env_ids]` in-place.
- `_move_goal_ghost(env_ids)`: converts goal pos → world frame, calls `write_root_pose_to_sim()`.
- `_update_goal_in_extras()` now writes `env.extras["goal_state"]` (singular).
- `reset()` and `reset_done_envs(dones)` call the full chain.

---

#### EE cannot reach contact height — 2026-05-11

The UR5e with tool-down `[0,1,0,0]` orientation has an effective minimum TCP Z of ~0.115 m
due to kinematic workspace limits. cuRobo reports IK success (position error < 5 mm for the
ee_link) but converges to the closest feasible point when the target is below the reachable
workspace.

**Fix (`test_push_primitive.py`):**
- **Fixed TCP→wrist3 offset calibration**: instead of using the live `_tcp_offset()` (which
  drifts during approach/orient because the arm isn't yet at tool-down), a frozen offset is
  measured once at startup. The calibration solves IK for a tool-down pose at Z=0.25 m
  seeded from the current joint configuration, steps the PD controller 30 times to settle,
  then freezes the measured offset. `ik_target = wp_pos - _FIXED_TCP_OFFSET` is used for
  all subsequent waypoints.
- **Workspace clamp**: `_WS_Z = (0.00, 0.55)` clamps the wrist3-target Z to prevent cuRobo
  from receiving infeasible targets. The minimum can be raised if needed.

---

## 7. Push Primitive Test <a name="push-primitive-test"></a>

**Date**: 2026-05-11

### 7.1 Overview

`tests/test_push_primitive.py` — interactive scenario-loop test that cycles through
pre-defined push scenarios to visually validate the push primitive. Runs in the viewer;
press Ctrl+C or close the viewport to exit.

### 7.2 Architecture

```
┌─────────────────────────────────────────────────────┐
│  SCENARIOS[6] — each is a 3-push sequence          │
│                                                     │
│  Each push: {offset_x, offset_y, push_dx, push_dy}  │
│                                                     │
│  ① Get object position from observation             │
│  ② compute_push_waypoints() → 76 waypoints  (was 115, Fix P32)     │
│  ③ Per waypoint:                                    │
│     ik_target = wp_pos − _FIXED_TCP_OFFSET           │
│     cuRobo solve_batch → joint positions            │
│     env.step() → physics                            │
│  ④ Print displacement / contact / velocity          │
│  ⑤ Pause 60 steps between pushes                    │
│  ⑥ Reset environment after each scenario            │
└─────────────────────────────────────────────────────┘
```

### 7.3 Key Configuration

| Setting | Value |
|---------|-------|
| cuRobo config | `ur5e.yml` (ee_link: tool0) |
| Orientation | Fixed tool-down `[0,1,0,0]` |
| Gripper | Always closed during push |
| Steps per push | 76: 12+3+16+20+16+1+8 (was 115, scaled ~1.5× per Fix P32) |
| Approach height | 0.40 m above table |
| Contact height | 0.110 m (cmd) → ~0.095 m actual TCP |
| TCP offset | Calibrated fixed offset at startup (30-step PD settle) |
| Workspace | X=[-0.5,0.5], Y=[0.25,0.70], Z=[0.232,0.55] (tool0 frame) |
| Pause between pushes | 60 steps (~1.2 s) |
| Object | T-block only (cube/cylinder/rect/triangle removed from scene) |

### 7.4 Scenarios

| S# | Push 1 | Push 2 | Push 3 |
|----|--------|--------|--------|
| 0 | Fwd 0.10 | Left 0.10 | Fwd 0.20 |
| 1 | Fwd 0.10 | Right 0.10 | (no-op) |
| 2 | Fwd 0.10 | Bwd 0.10 | (no-op) |
| 3 | Fwd 0.10 | Fwd 0.10 | (no-op) |
| 4 | Fwd 0.10 | LeftFwd 0.07 | (no-op) |
| 5 | Fwd 0.10 | RightFwd 0.07 | (no-op) |

All scenarios use `offset_x=0.05, offset_y=0.05` (5 cm safety margin from object center).

### 7.5 Running

```bash
python -m asyncDualPlayPPO.tests.test_push_primitive
python -m asyncDualPlayPPO.tests.test_push_primitive --step-delay 0.05
```

### 7.6 T-Block Object & Inertia Fix — 2026-05-13

#### Problem

Off-center pushes induced no object rotation (e.g. T-block pushed from the back-right
corner translated without spinning). Root cause: every object had `physics:diagonalInertia = (1, 1, 1)`
baked into its USD/URDF — a placeholder ~10 000× too high for small tabletop blocks (0.04–0.1 kg).

#### Root Cause in USD Physics

USD Physics precedence for rigid body properties:
1. If `physics:diagonalInertia` is explicitly authored → PhysX uses it as-is (ignores density/geometry)
2. If not authored → PhysX computes inertia from collision geometry × mass/density

Because all objects had `(1,1,1)` authored, density/geometry were ignored and objects behaved like
flywheels — enormous torque needed for any angular acceleration.

#### Fix

**a) T-block asset (`t_shape.usda`):**
- Removed `physics:diagonalInertia = (1, 1, 1)` — lets PhysX compute from collision geometry
- Kept explicit `physics:mass = 0.1` — light, responsive; density override removed from config
- Scale `(2.0, 2.0, 1.5)` applied at spawn for better EE contact surface
- Fixed file reference `t_shape.usd` → `t_shape.usda` (binary `.usd` removed in prior commit)

**b) URDF objects (`cube/cylinder/rect/triangle/concave.urdf`):**
- Removed `<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>` from all URDF files
- Existing binary USD files retain old inertia until regenerated; URDF fixes are for future conversions
- URDF objects keep their explicit mass (0.04–0.10 kg) via `<mass>`, and `MassPropertiesCfg(density=300.0)` for density

**c) Generator scripts (`gen_t_shape.py`, `generate_t_shape_block.py`):**
- Removed hardcoded `physics:diagonalInertia` and `physics:mass` attribute creation
- Comment added: mass/inertia left unset — density-based computation handles it

**d) Test flow (`test_push_primitive.py`):**
- Changed `target_object` comment from "long green cuboid" to "T-block"
- Loop exits after `len(SCENARIOS)` scenarios (was infinite `while simulation_app.is_running()`)

**e) Isaac Sim cache cleanup:**
- Cleared `/isaac-sim-5.1.0/kit/cache/`, `/kit/logs/`, shader caches (`extscache/omni.*.shadercache.*`)
- Cleared project `.isaac_cache/` and `logs/`

#### Result

Objects now rotate smoothly when pushed off-center:
- T-block (S1 Push 1): 0° → +37° with `offset_x=-0.08, offset_y=0.08`
- T-block (S6 Push 2): 0° → −33° with `offset_x=-0.10, offset_y=0.00`
- No flyaway or drift (final velocities ~0 m/s after each push)
- Object movement is smooth (explicit `mass=0.1` prevents sluggish density-based mass)

#### Files Changed

| File | Change |
|------|--------|
| `tasks/push_task_curobo.py:164-166` | Fixed USD reference `.usd`→`.usda`, scale `(2.0,2.0,1.5)`, removed `mass_props` |
| `assets/blocks/t_shape.usda` | Removed `diagonalInertia` and `mass` lines; readded `mass=0.1` only |
| `assets/blocks/gen_t_shape.py` | Removed mass and inertia attribute creation |
| `assets/blocks/generate_t_shape_block.py` | Removed mass and inertia attribute creation |
| `assets/blocks/{cube,cylinder,rect,triangle,concave}.urdf` | Removed `<inertia>` blocks |
| `tests/test_push_primitive.py` | Updated comment; loop exits after all 6 scenarios |

---

### 7.7 Robot Lag Fix — 2026-05-13

#### Problem

After the T-block / inertia fix, the robot arm moved sluggishly (visible lag in joint tracking) while the rest of the scene was unaffected. `nvtop` showed clean GPU utilization.

#### Root Cause — Free-falling physics objects

`push_task_curobo.py` was extended to pre-load all 5 block shapes into the scene simultaneously (`cube`, `cylinder`, `rect`, `triangle`, `target_object`), with the 4 inactive ones placed at `Z=-2.0` and `disable_gravity=False`. Because there is no collision surface below `Z=0` in the scene, the 4 hidden objects were in **permanent free-fall** — accelerating indefinitely, never reaching a resting state, so PhysX could never put them to sleep. Every physics step had to integrate 4 continuously moving rigid bodies, adding articulation-solver overhead that manifested as robot joint lag.

This is distinct from the earlier approach-height issue (which was a waypoint-spacing problem). The lag persisted even after correcting the step counts because the physics overhead remained.

#### Fix

Removed cube, cylinder, rect, and triangle from the `PushTaskSceneCfg` entirely (`= None`). All 6 scenarios now use `target_object` (T-block). The `_swap_object` helper and the `SCENARIO_OBJECTS` / `_ALL_OBJECT_NAMES` lists were removed from `test_push_primitive.py`.

| File | Change |
|------|--------|
| `tasks/push_task_curobo.py` | `cube = cylinder = rect = triangle = None` (removed 4 free-falling rigid bodies) |
| `tests/test_push_primitive.py` | Removed `_swap_object`, `SCENARIO_OBJECTS`, `_ALL_OBJECT_NAMES`; `active_obj_name` hardcoded to `"target_object"` |
| `tasks/utils/action_push.py` | `PUSH_APPROACH_HEIGHT = 0.40 m`; substeps 115→76 per push (Fix P32) |