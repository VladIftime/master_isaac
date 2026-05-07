# ASP + GoalEncoder — Implementation Status (`asp_goal_encoder` branch)

> This document tracks all changes made on the `asp_goal_encoder` branch relative to
> `master`, explains how each component maps to the system architecture and hardware-in-the-loop
> validation tests, and lists what remains to be implemented.

---

## 1. Framework Overview

The framework combines two papers applied to two physical UR5e arms in Isaac Lab / Isaac Sim:

| Paper | Role |
|---|---|
| Plappert et al. 2021 — *Asymmetric Self-Play* | Automatic curriculum via Alice↔Bob adversarial game |
| Sukhbaatar et al. 2018 — *Goal Embeddings via Self-Play* | GoalEncoder φ-MLP compresses goal into fixed-size latent |

### Adversarial Loop

```
Alice (PPO, 35D obs, 100 steps)
  → manipulates objects freely
  → leaves workspace in non-trivial state
  → that state becomes the goal

Bob (PPOABC + GoalEncoder, 51D obs, 200 steps)
  → must reproduce Alice's configuration from scratch
  → goal enters via GoalEncoder latent (8D) injected into actor trunk
  → reward: +1 per object at goal, −1 if object leaves goal, +5 completion

When Bob fails → Alice's trajectory → ABC buffer → imitation loss β=0.5
```

### Observation Spaces

| Agent | Dimension | Contents |
|---|---|---|
| Alice | 35D | `robot_state(7)` + `obj1_state(14)` + `obj2_state(14)` |
| Bob | 51D | `robot_state(7)` + `[obj_state(14)+goal(6)+dist(2)] × 2` |
| Goal state | 12D | `[pos(3)+euler(3)] × 2` objects, local frame |

### Kinematic Pipeline (cuRobo variant)

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

## 2. What Has Been Implemented

### 2.1 Core Training Loop (`train_curobo.py`)

- **cuRobo IK pipeline** replacing RMPFlow / DiffIK:
  - `_bins_to_xyz_rxy_gripper`: decodes 6D bins → XYZ delta + Rx/Ry delta + sticky gripper
  - Per-env EE position accumulator `ee_target_local` (reset on phase transition / done)
  - Per-env EE orientation accumulator `ee_target_quat_w` via quaternion composition (avoids gimbal lock)
  - TCP offset correction computed live from finger-midpoint vs wrist_3_link each step
  - `ik_solver.solve_batch(N)` seeded from `_prev_joint_cmd` for smooth trajectories
  - IK failure recovery: reverts EE accumulator to pre-step snapshot, holds current joint positions
  - Phase sync: re-anchors both accumulators to physics TCP state after every phase transition or episode done
  - Workspace clamp: X ∈ [−0.50, 0.50], Y ∈ [0.25, 0.70], Z ∈ [0.00, 0.55] metres (env-local)
  - cuRobo CUDA graph warm-up before training loop (reduces per-step latency ~3 ms → ~0.5 ms)
  - IK fail rate logged to TensorBoard (`Metrics/IKFailRate`) each iteration

- **Two-phase ASP structure**:
  - Alice phase (100 steps): explores, builds goal state
  - Bob phase (200 steps): reproduces goal, receives sparse reward
  - Phase stagger at startup: random offset across envs prevents simultaneous resets
  - LSTM hidden states zeroed on phase transitions and episode done events

- **Alice Behavioral Cloning (ABC)**:
  - Alice trajectory buffer: `(alice_traj_obs, alice_traj_act, alice_traj_len)` per env per rollout
  - Gate: `bob_done & ~bob_success & goal_valid & traj_len >= max(10, alice_timesteps//2)`
  - BC observation construction: Alice obs repackaged with goal appended per timestep
  - `old_log_prob` batch-computed under current Bob policy before storage
  - `GPUDemonstrationBuffer`: sliding window of 500 trajectories, evicts oldest on overflow
  - Buffer size logged (`Metrics/ABC/BufferSize`)
  - Buffer persisted across checkpoints (`abc_buffer.pt`), loaded on resume

- **Historical policy pool**:
  - 20% of active envs per phase use a past policy snapshot (HIST_FRAC = 0.2)
  - Pool holds last 5 snapshots; saved every 50 iterations
  - `sample_env_subset`: splits env indices into current vs historical fractions

- **Fixed controllers** (SR-coupled controllers removed per Fixes 1 & 2):
  - Alice entropy coef: `ent_coef = 0.05` (fixed from YAML, logged as constant)
  - Alice LR: cosine decay `lr_max=3e-4 → lr_min=5e-5` over `max_iterations`
  - Bob `abc_coef`: fixed at 0.5 (paper Table 2 value)

- **Diagnostic flags**:
  - `--test_reward_pipeline`: runs `run_test1()` from `diagnostics/test_reward_pipeline.py`, exits
  - `--alice_sandbox`: suppresses Bob PPO updates; Bob acts randomly (Test 2)
  - `--dummy_alice`, `--dummy_goal_distance`, `--test_movement`: diagnostic wrappers
  - `--diag_alice_exploration`: uses `DiagnosticAliceWrapper`
  - `--test_hparams`: audits YAML hyperparameters against paper Table 2 values
  - `--profile`: enables per-section profiler (`curobo_ik`, `alice_act`, `bob_act`, etc.)

- **Checkpoint system**:
  - Periodic saves every `save_interval` iterations: model weights, ABC buffer, episode manager, train state
  - SIGTERM emergency checkpoint handler: saves immediately on signal, then exits cleanly
  - Best-model checkpoint: overwritten whenever `bob_success_rate > best_bob_success_rate`
  - Resume: `--chkpt_alice`, `--chkpt_bob`, `--resume_iteration`

- **TensorBoard logging**:
  - `Loss/Alice/{Value,Surrogate}`, `Loss/Bob/{Value,Surrogate,ABC}`
  - `Reward/{Alice,Bob}`, `Metrics/Bob/{SuccessRate,PosError,RotError}`
  - `Metrics/Alice/{ValidGoals,InvalidGoals,GoalValidityRate,MeanDisp3D}`
  - `GoalEncoder/{embedding_norm,embedding_std,aux_pos_loss,aux_rot_loss}`
  - `Metrics/ABC/{BufferSize,IsWarm}`, `Metrics/IKFailRate`, `Metrics/IKOverheadFrac`
  - `Alice/{EntropyCoef,LearningRate}`, `Bob/ABCCoef`

### 2.2 GoalEncoder (`algorithms/goal_encoder.py`)

- φ-MLP: `Linear(6→64) → Tanh → Linear(64→8)` per object
- **Difference variant** (default): `g_i = φ(goal_i) − φ(current_i)`
- **Absolute variant**: `g_i = φ(goal_i)`
- **Max-pool** across objects → 8D pooled embedding `g_pooled` (Fix 7; was sum-pool)
- Internally uses ZYX Euler angles (matches observation space; quaternion→axis-angle path removed)
- Auxiliary distance-prediction head: `aux_loss` trains encoder to predict `(pos_dist, rot_dist)` — adds geometric inductive bias without a separate training phase (Fix 13: intentionally kept)

### 2.3 PPOABC (`algorithms/rl/ppo/ppo_abc.py`)

- Total Bob loss per mini-batch: `L = L_PPO + β·L_ABC (last batch) + aux_coef × L_aux`
- **Fix 5**: `epoch_bc_loss` computed once per epoch (before mini-batch loop) via `_compute_abc_loss_sequential()`; added to the **last** mini-batch loss → single `optimizer.step()` with combined gradients (paper: L = L_PPO + β·L_ABC per epoch)
- ABC loss: PPO-clipped NLL over complete trajectories, evaluated sequentially through LSTM to maintain correct hidden state
- **Fix 14**: `detach_goal_encoder=False` during ABC evaluation — GoalEncoder receives gradients from ABC loss
- `aux_coef = 0.1` geometric regulariser; `GoalEncoder/aux_pos_loss` and `GoalEncoder/aux_rot_loss` logged per iteration via `self.writer`
- **Fix 9**: `abc_warmup_threshold = 0.0` unconditionally — ABC active from iteration 1
- **Fix 16**: KL adaptive LR block guarded with `not self.actor_critic.use_multicategorical` (dead code in MC mode)

### 2.4 Actor Network — Bob (`algorithms/rl/ppo/module.py`)

- PermInvEncoder (PI): shared `Linear(14→512) → LN → ReLU → Linear(512→512) → LN → ReLU`, max-pool, post-pool LayerNorm
- GoalEncoder output injected additively into first trunk layer:
  `h1 = ReLU(LN(W·enc + Wg·g_pooled))`
- `_goal_proj = Linear(8→512, bias=False)` scaled ×0.1 at init to prevent ReLU saturation
- LSTMCell(128→256) → MultiCategorical(6 dims × 11 bins)
- **Fix 12**: `num_cat_dims` default changed from 4 → 6 (paper Appendix B.2: XYZ + Rx/Ry + gripper)
- Critic: direct MLP on full raw obs (no bottleneck)
- `act_with_hidden()`: returns `(acts, logprob, val, mu, sigma, new_hidden)` for per-phase rollout

### 2.5 Environment & Tasks

- `AsyncDualPlayEnvWrapper` (`tasks/utils/wrapper.py`): phase management, goal validation, reward computation, ABC buffer writes
- `AsyncDualPlayCuRoboEnvCfg` (`tasks/async_dual_play_curobo.py`): scene with `JointPositionActionCfg`
- Object pool at startup: randomly selects from `[concave, cube, cylinder, rect, triangle]` USD assets
- Objects scaled (1.5, 1.5, 1.5), colored green; table turns red (Alice) / blue (Bob) in non-headless mode
- No IK reset on IK failure — robot holds last valid pose, episode continues
- **Fix 3 + 11**: Alice per-step rewards set to 0.0 unconditionally; base_rewards passthrough and potential shaping removed
- **Fix 4**: Bob potential-shaping block removed; `rewards = step_rewards + completion_bonus` only
- **Fix 10**: 7 cm min-XY displacement filter removed from Alice's goal validity check

### 2.6 Diagnostic Test Suite (`diagnostics/`)

| File | Test | Trigger |
|---|---|---|
| `test_reward_pipeline.py` | Test 1: teleport objects to exact goal coords, assert SR > 0 and +5 bonus fires | `--test_reward_pipeline` flag |
| `test_alice_sandbox.py` | Test 2: offline TB analysis — ValidGoals trend, GoalValidityRate, EntropyCoef=0.05, InvalidGoals > 0, no NaN | Offline, reads TB event files |
| `test_ppo_abc_balance.py` | Test 3a: abc_coef constant 0.5, ABC loss > 0 after buffer fill, surrogate finite, ABC/PPO loss ratio < 5× | Offline, reads TB event files |
| `test_checkpoint_chain.py` | Test 3b: ABC buffer save/load round-trip, shapes and keys intact | Offline, reads `abc_buffer.pt` |
| `test_abc_goal_encoder.py` | Test 4a: `_encode_obs()` forward pass, g_pooled shape=(B,K), gradient flows into GoalEncoder | Offline, loads checkpoint |
| `test_goal_encoder_latent.py` | Test 4b: t-SNE silhouette > 0.15, noise invariance < 0.20, TB embedding_norm health | Offline, loads checkpoint |

#### Proposed additional tests (not yet implemented)

| Test | Validates | Gap vs current suite | Priority |
|---|---|---|---|
| **A** — Alice proposer collapse + MeanDisp3D floor | Fixes 2, 3, 10, 11 | `test_alice_sandbox.py` checks ValidGoals trend but never asserts `MeanDisp3D.mean() > 0.04` — micro-movement exploitation has no guard | Medium |
| **B** — PPO & ABC gradient norm ratio | Fixes 5, 1 | `test_ppo_abc_balance.py` checks loss magnitude ratio (< 5×) but not gradient norms; `ppo_abc.py` does not log `Loss/Bob/GradNorm_{PPO,ABC}` | Medium |
| **C** — Unfrozen GoalEncoder aux-loss spike detection | Fixes 14, 13 | `GoalEncoder/aux_pos_loss` and `aux_rot_loss` ARE logged; `test_goal_encoder_latent.py` checks overall latent health but has no consecutive-spike check on aux loss | Low |
| **D** — Hardware jitter / joint acceleration (UR5e gate) | Fix 8 | Not covered anywhere — no test records step-to-step `Δq` or asserts `max(Δq) < threshold` | **Critical before UR5e deployment** |
| **E** — Max-pool object saturation with distractors | Fix 7 | Not covered anywhere — no test varies object count to verify embedding norm does not grow with N | Medium |

### 2.7 HPC Slurm Scripts (`hpc/`)

| Script | Purpose |
|---|---|
| `train_curobo.slurm` | Production cuRobo training run |
| `train_curobo_large.slurm` | Large-scale (512+ envs) |
| `train_curobo_profile.slurm` | 3-iteration profiler run |
| `diagnostic_tests.slurm` | Runs full 4-test suite on HPC |
| `test1_ppo_reward.slurm` | Test 1 only |
| `test2_alice_exploration.slurm` | Test 2 only (200 iters, random Bob) |
| `test3_asp_tug_of_war.slurm` | Test 3 only (50 iters, full pipeline) |

### 2.8 Supporting Utilities

- `utils/historical_pool.py`: ring buffer of past 5 policy snapshots, `sample_env_subset()` for HIST_FRAC split
- `utils/episode_manager.py`: phase tracking, goal storage (`goal_states`), `state_dict()` / `load_state_dict()` for checkpointing
- `utils/profiler.py`: `TrainingProfiler` with `section()` context manager, `mark_start/stop`, `get_section_frac()`, per-iteration summaries
- `utils/goal_validator.py`: `validate_goal()` checks minimum object displacement threshold
- `optuna_sweep.py`: Optuna hyperparameter sweep wrapper

---

## 3. Architecture ↔ Hardware-in-the-Loop Mapping

The four diagnostic tests directly validate the four hardware integration layers:

### Test 1 → Reward Pipeline / Perception Layer

**What it checks:** `_compute_bob_sparse_rewards` fires at the correct thresholds when objects are teleported to exact goal coordinates.

**HIL relevance:** Before training on physical UR5e arms, the state estimation system (cameras / object trackers) must resolve object poses to within the reward thresholds:
- Position: L2 distance ≤ 0.05 m
- Rotation: max |ZYX Euler diff| with `[0,π]` wraparound ≤ 0.2 rad

If the physical tracking system has noise exceeding these thresholds, Bob will see false negatives on every step — the learning gradient is destroyed. Physical validation: place objects manually to the goal pose with precision tools, verify SR > 0.

The `rot_distance_euler()` function in `test_reward_pipeline.py` implements `[0,π]` wraparound identically to `observations.py` — the unit tests guard against boundary spikes at ±180°.

### Test 2 → Curriculum Emergence Layer

**What it checks:** `Metrics/Alice/ValidGoals` trends upward over 200 iterations with random Bob, confirming Alice is learning to produce valid goals without curriculum collapse.

**HIL relevance:** On physical hardware, Alice's joint velocity commands are processed through cuRobo before reaching the UR5e arms. Alice entropy must remain high enough for exploration while staying within the kinematic reach of both arms. The `MeanDisp3D` metric tracks average object displacement — this should increase as Alice learns more complex manipulations.

The `alice_sandbox` mode (Bob acts randomly) isolates Alice's curriculum from Bob's learning — exactly the condition needed to verify that the curriculum emerges from Alice's own reward signal, not from Bob's feedback.

### Test 3 → PPO + ABC Optimization Layer

**What it checks:** ABC loss is nonzero once the buffer is populated, surrogate loss is finite, `abc_coef` is stable.

**HIL relevance:** On physical arms, the ABC buffer persistence across training interruptions (`test_checkpoint_chain.py`) is critical. A cold-restart loss of the ABC buffer means Bob must re-explore from scratch, wasting hours of physical robot time. The checkpoint round-trip test verifies `(obs, acts, old_lp)` tensors survive serialization with correct shapes and dtypes.

Physical validation: verify that actions sampled from the ABC buffer translate to smooth UR5e motions via cuRobo without high-frequency jitter (indicates imitation learning gradients are yielding physically plausible motor primitives).

### Test 4 → GoalEncoder / Representation Layer

**What it checks:** Forward pass integrity (Test 4a), t-SNE cluster separation by task type, noise invariance (Test 4b).

**HIL relevance:** On physical hardware, the GoalEncoder must abstract away task-irrelevant variation (arm resting pose, lighting, background) and map the same physical object configuration to the same latent vector. The noise invariance test (σ=2cm perturbation → relative embedding change < 0.20) directly simulates camera measurement noise on the physical tracking system.

Physical validation: extract specific cluster centroids from the t-SNE visualization, feed as fixed goal commands to the physical Bob policy, verify consistent motor primitives across repeated trials.

---

## 4. Open Issues and Status

### 4.1 ✅ FIXED — Entropy Coef Mismatch in Test

**Was:** `test_alice_sandbox.py` asserted entropy = 0.01; `ppo_continuous.yaml` has `ent_coef: 0.05`.

**Fix applied:** `diagnostics/test_alice_sandbox.py` now checks against `EXPECTED_ENT_COEF = 0.05`
and prints the expected value in failure messages.

---

### 4.2 ✅ FIXED — Shell Script Entry Point Mismatch

**Was:** `diagnostics/run_diagnostics.sh` and `run_diagnostic_tests.sh` called
`python -m asyncDualPlayPPO.train`. The diagnostic flags (`--test_reward_pipeline`,
`--alice_sandbox`) exist only in `train_curobo.py`.

**Fix applied:** All three invocation lines in both scripts updated to
`python -m asyncDualPlayPPO.train_curobo`.

---

### 4.3 ✅ FIXED — GoalEncoder Stale Axis-Angle References

**Was:** Module docstring, class docstring, `_preprocess_pose`, and `forward()` all contained
stale references to a quaternion→axis-angle conversion that was already removed from the
implementation.

**Fix applied:**
- Removed the `quat_to_axis_angle()` function (dead code — no callers)
- Updated module docstring: now correctly states ZYX Euler input with rationale
- Fixed class `Args` docstring for `pose_dim` ("pos(3) + euler(3)")
- Removed stale `# Convert quat → axis-angle` comment from `forward()`
- Collapsed `_preprocess_pose` to a one-line pass-through with concise docstring

---

### 4.4 ✅ FIXED — SR-Coupled abc_coef (Fix 1, Critical)

**Paper:** β=0.5 constant (Table 2)

**Was:** `_abc_coef_start * (1 − bob_sr)` with EMA 0.95 after anneal period — injected parasitic second-order feedback loop, destroyed trust region.

**Fix applied:** `train_curobo.py:1247` — `bob_ppo.abc_coef = ppo_cfg["params"]["learn"].get("abc_coef", 0.5)` (fixed β per Table 2). `ppo_abc.py:57,62` — `abc_warmup_threshold = 0.0` unconditionally.

**When to re-enable dynamic controller:** Once training runs demonstrate stable SR > 0.3, restore:
```python
target_abc = abc_coef_start * (1.0 - bob_sr)
bob_ppo.abc_coef = 0.95 * bob_ppo.abc_coef + 0.05 * target_abc
```
Also update `test_ppo_abc_balance.py` — its `std < 1e-3` assertion assumes constant β.

---

### 4.5 ✅ FIXED — SR-Coupled Alice Entropy (Fix 2, Critical)

**Paper:** `entropy_coef=0.01` fixed (Table 2)

**Was:** Phase 2 (iter≥250) used PI controller coupled to SR_B — vicious feedback loop causing premature mode collapse.

**Fix applied:** `train_curobo.py:1234` — entropy_coef read-only from config (default 0.05 per `ppo_continuous.yaml`). PI controller and decay schedule removed entirely. Logging kept.

**When to re-enable dynamic controller:** Set `abc_anneal_iters` back to a nonzero value and restore the per-iteration PI controller block. Also update `test_alice_sandbox.py` — the fixed-value check will need to become a trend/range check.

---

### 4.6 ✅ FIXED — Test 2 Missing Out-of-Zone Penalty Verification

**Was:** `test_alice_sandbox.py` had no check that invalid/OOZ goals were ever produced.

**Fix applied (two parts):**

1. `train_curobo.py`: added `writer.add_scalar("Metrics/Alice/InvalidGoals", ...)` so the
   count is now logged to TensorBoard every iteration.

2. `diagnostics/test_alice_sandbox.py`: added check 4 — asserts `InvalidGoals.sum() > 0`
   across the run, confirming OOZ penalty logic fired at least once.

---

### 4.7 ✅ FIXED — Test 4b No-Op in CI

**Was:** `run_diagnostics.sh` always fed the 50-iteration Test 3 checkpoint to Test 4b.
The test exits via `sys.exit(0)` (SKIP) at 50 iters — effectively never validated.

**Fix applied:** `diagnostics/run_diagnostics.sh` Test 4b block now uses environment
variables `LONG_RUN_CKPT` and `LONG_RUN_LOG`. For latent-space validation:

```bash
LONG_RUN_CKPT=runs/production/bob/model_500.pt \
LONG_RUN_LOG=runs/production/summary \
    bash diagnostics/run_diagnostics.sh --test 4
```

---

### 4.8 ✅ FIXED — cuRobo <10% Overhead Check Missing

**Was:** `Metrics/IKFailRate` was logged but there was no check that IK time stayed
within the <10% budget of total rollout step time.

**Fix applied:**

1. `utils/profiler.py`: added `get_section_frac(section, relative_to)` method.

2. `train_curobo.py`: computes `_ik_overhead = profiler.get_section_frac("curobo_ik", "env_step")`,
   logs it as `Metrics/IKOverheadFrac`, prints WARNING if it exceeds 10%.

---

### 4.9 [FUTURE] Charlie Hierarchical Controller — Not Yet Implemented

The second paper (Sukhbaatar et al. 2018) describes a **Charlie controller**: a high-level
policy generating goal embeddings `g_t ∈ R^8` to command a frozen pre-trained Bob.

No `charlie.py` exists. This is a research extension requiring:
- A separate Charlie PPO policy trained on top of a frozen Bob checkpoint
- An evaluation mode that wires Charlie's output directly to Bob's `_goal_proj`
- Hierarchical episode structure: Charlie proposes sub-goals, Bob executes

---

### 4.10 [FUTURE] Physical Sim-to-Real Interface — Not Yet Implemented

Deployment to physical UR5e arms requires:
- **UR5e driver**: ROS2 / UR RTDE interface for joint position commands from cuRobo
- **State estimation bridge**: camera → local frame object poses at <0.05 m / 0.2 rad
- **Inference script**: checkpoint → `actor_critic.act()` → cuRobo → physical arm
- **Physical reward detection**: `rot_distance_euler()` on tracked physical poses

---

### 4.11 ✅ FIXED — Dense Potential Shaping for Alice (Fix 3, Critical)

**Paper:** Alice gets zero per-step reward — only outcome {+5/+1/−3/0} at phase end.

**Was:** `wrapper.py` applied Φ(s) = 3.0·(1−exp(−5·dist)) every Alice step, rewarding object displacement. Alice optimised for moving objects far rather than finding Bob's blind spots.

**Fix applied:** `wrapper.py:963-966` — Alice block unconditionally sets `rewards[is_alice] = 0.0`. Base_rewards passthrough (physics penalties) removed simultaneously (covers Fix 11).

---

### 4.12 ✅ FIXED — Dense Potential Shaping for Bob (Fix 4, Critical)

**Paper:** Sparse {+1/−1/+5} only.

**Was:** F = γ·Φ(s') − Φ(s), Φ(s) = −Σ pos_dist, every Bob step. Bob received a dense gradient toward the goal regardless of goal difficulty, undermining the autocurriculum.

**Fix applied:** `wrapper.py:1043,1113` — shaping block removed; `rewards = step_rewards + completion_bonus` only.

---

### 4.13 ✅ FIXED — ABC as Separate Backward Pass (Fix 5, Critical)

**Paper:** L = L_PPO + β·L_ABC as a single combined loss per mini-batch, every epoch.

**Was:** All `num_epochs × num_batches` PPO steps completed first, then one separate ABC backward pass. PPO and ABC gradients never co-mingled; policy shifted before ABC could correct it.

**Fix applied:** `ppo_abc.py:127-132,276-281` — `epoch_bc_loss` computed once per epoch before the mini-batch loop; added to the last mini-batch's PPO loss → single `optimizer.step()` with L = L_PPO + β·L_ABC per epoch.

---

### 4.14 [INTENTIONAL] GoalEncoder Architecture — Charlie Paper Extension (Fix 6)

**Paper (Figure 12):** Goal state processed by the same PI encoder as current object state, then summed.

**Implementation:** Separate GoalEncoder φ-MLP (goal_pose, current_pose) → K=8 latent; injected additively after actor layer 1. This is the "Charlie" paper architecture.

**Status:** Not fixed — deliberate design choice. Fixing would require removing GoalEncoder entirely and adding a second PermInvEncoder pass for goal states; deferred.

---

### 4.15 ✅ FIXED — Sum-Pool Instead of Max-Pool for Goal Embedding (Fix 7, High)

**Paper:** Max-pool over objects throughout.

**Was:** `module.py` used `g_pooled = g_per_obj.sum(dim=1)` — inconsistent with both the paper's PI encoder and the GoalEncoder's own internal max-pool. Sum-pool saturates on large object counts.

**Fix applied:** `module.py:429` — `g_pooled = g_per_obj.max(dim=1)[0]`.

---

### 4.16 ✅ FIXED — EMA Joint Smoothing (Fix 8, Medium)

**Paper:** Direct TCP servoing, no filtering.

**Was:** `_JC_ALPHA=0.2` → smoothed = 0.2·raw_IK + 0.8·prev_cmd. Arm position dominated by IK solutions from steps t−2 through t−5; policy output and actual EE motion decorrelated in time.

**Fix applied:** `train_curobo.py:715` — `_JC_ALPHA = 1.0` (no smoothing; raw IK solution used directly).

> **UR5e deployment gate:** No test currently validates that cuRobo alone provides sufficient smoothing. Run proposed Test D (joint Δq check) before physical deployment.

---

### 4.17 ✅ FIXED — abc_warmup_threshold Gate (Fix 9, Medium)

**Paper:** β=0.5 from iteration 1.

**Was:** ABC held at 0 until `alice_mean_rew ≥ abc_warmup_threshold`, delaying engagement.

**Fix applied:** `ppo_abc.py:62` — `self.abc_warmup_threshold = 0.0` unconditionally.

---

### 4.18 ✅ FIXED — Min XY Displacement Filter (Fix 10, Medium)

**Paper:** Requires only 3D displacement > Bob's success threshold (~4cm); no XY-specific filter.

**Was:** Alice's goal rejected if no object moves >7cm in XY — filtered rotation-only and short-range displacement goals that the paper explicitly includes.

**Fix applied:** `wrapper.py:576` — entire XY displacement filter block removed; goal validity determined solely by `validate_goal()` (position + rotation thresholds in 3D).

---

### 4.19 ✅ FIXED — Alice Receives Physics Penalties (Fix 11)

**Paper:** Alice gets zero per-step reward during her phase.

**Was:** `rewards[is_alice] = base_rewards[is_alice]` — collision/OOB penalties from RewardManager passed through.

**Fix applied:** Covered by Fix 3 — `wrapper.py:963-966` sets `rewards[is_alice] = 0.0` unconditionally.

---

### 4.20 ✅ FIXED — num_cat_dims Default is 4, Not 6 (Fix 12)

**Paper:** 6D × 11 bins (XYZ + Rx/Ry + gripper) — Appendix B.2.

**Was:** `model_cfg.get("num_cat_dims", 4)` — default 4 dims; `bins_to_delta` assumes 6 (slices :3, 3:5, 5:6). If config omitted `num_cat_dims`, silently produced wrong-shaped outputs.

**Fix applied:** `module.py:186` — `model_cfg.get("num_cat_dims", 6)`.

---

### 4.21 [INTENTIONAL] Aux Loss Head on GoalEncoder (Fix 13, Low)

**Paper:** No auxiliary loss; object states fed directly.

**Implementation:** GoalEncoder predicts (pos_dist, rot_dist) as supervised auxiliary signal. Logged per iteration as `GoalEncoder/aux_pos_loss` and `GoalEncoder/aux_rot_loss`.

**Status:** Not fixed — provides useful geometric supervision outside the PPO/ABC loop; kept as-is.

---

### 4.22 ✅ FIXED — GoalEncoder Frozen During ABC (Fix 14, Low)

**Paper:** No separate goal encoder to freeze.

**Was:** `detach_goal_encoder=True` during ABC forward — GoalEncoder received no gradient from ABC loss.

**Fix applied:** `ppo_abc.py:99` — `detach_goal_encoder=False`. GoalEncoder now receives ABC gradients.

> **Watch:** If `GoalEncoder/aux_pos_loss` or `aux_rot_loss` spikes after unfreezing, it signals that ABC trajectories are forcing the encoder to memorise Alice's sub-optimal poses. Re-freeze by reverting to `detach_goal_encoder=True` at that point.

---

### 4.23 ✅ FIXED — ppo.py log() Crashes with MultiCategorical (Fix 15, Low)

**Was:** `self.actor_critic.log_std.exp().mean()` — `log_std` not created when `use_multicategorical=True` → AttributeError at logging time. Silent in production because `ppo_abc.py` doesn't call `log()`, but crashes if `ppo.py` is used directly.

**Fix applied:** `ppo.py:232-235` — guarded with `if hasattr(self.actor_critic, "log_std"):`.

---

### 4.24 ✅ FIXED — KL Adaptive LR is Dead Code in MC Mode (Fix 16, Low)

**Paper:** Fixed lr=3×10⁻⁴ (Table 2).

**Was:** KL formula uses `sigma_batch`; in MC mode `sigma=zeros` always → KL=0 → adaptive LR never fires. `desired_kl=None` by default anyway — doubly dead.

**Fix applied:** `ppo_abc.py:176` — added `and not self.actor_critic.use_multicategorical` guard to the KL block condition. Dead code path now explicitly skipped.

---

## 5. Fix Summary

| # | Issue | Severity | Status | Files Changed |
|---|---|---|---|---|
| 4.1 | Entropy coef mismatch in test (0.01 vs 0.05) | Infra | ✅ Fixed | `diagnostics/test_alice_sandbox.py` |
| 4.2 | Shell scripts wrong entry point | Infra | ✅ Fixed | `diagnostics/run_diagnostics.sh`, `run_diagnostic_tests.sh` |
| 4.3 | GoalEncoder stale axis-angle dead code | Infra | ✅ Fixed | `algorithms/goal_encoder.py` |
| 4.4 / Fix 1 | SR-coupled abc_coef | Critical | ✅ Fixed | `train_curobo.py:1247`, `ppo_abc.py:57,62` |
| 4.5 / Fix 2 | SR-coupled Alice entropy | Critical | ✅ Fixed | `train_curobo.py:1234` |
| 4.6 | Test 2 OOZ penalty not verified | Infra | ✅ Fixed | `train_curobo.py`, `diagnostics/test_alice_sandbox.py` |
| 4.7 | Test 4b no-op in CI | Infra | ✅ Fixed | `diagnostics/run_diagnostics.sh` |
| 4.8 | cuRobo <10% overhead not checked | Infra | ✅ Fixed | `utils/profiler.py`, `train_curobo.py` |
| 4.9 | Charlie controller | — | Future research | — |
| 4.10 | Physical deployment stack | — | Future hardware | — |
| Fix 3 | Dense potential shaping for Alice | Critical | ✅ Fixed | `tasks/utils/wrapper.py:963-966` |
| Fix 4 | Dense potential shaping for Bob | Critical | ✅ Fixed | `tasks/utils/wrapper.py:1043,1113` |
| Fix 5 | ABC as separate backward pass | Critical | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:127-132,276-281` |
| Fix 6 | GoalEncoder architecture ≠ paper | High | Intentional | — |
| Fix 7 | Sum-pool → max-pool for goal embedding | High | ✅ Fixed | `algorithms/rl/ppo/module.py:429` |
| Fix 8 | EMA joint smoothing (_JC_ALPHA 0.2→1.0) | Medium | ✅ Fixed | `train_curobo.py:715` |
| Fix 9 | abc_warmup_threshold gate | Medium | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:62` |
| Fix 10 | 7cm XY displacement filter removed | Medium | ✅ Fixed | `tasks/utils/wrapper.py:576` |
| Fix 11 | Alice physics penalties (covered by Fix 3) | Medium | ✅ Fixed | `tasks/utils/wrapper.py:963-966` |
| Fix 12 | num_cat_dims default 4→6 | Medium | ✅ Fixed | `algorithms/rl/ppo/module.py:186` |
| Fix 13 | Aux loss head on GoalEncoder | Low | Intentional | — |
| Fix 14 | GoalEncoder detach=True→False during ABC | Low | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:99` |
| Fix 15 | ppo.py log() crash on MultiCategorical | Low | ✅ Fixed | `algorithms/rl/ppo/ppo.py:232-235` |
| Fix 16 | KL adaptive LR dead code in MC mode | Low | ✅ Fixed | `algorithms/rl/ppo/ppo_abc.py:176` |

---

## 6. Key Hyperparameter Reference (from `cfg/ppo/ppo_continuous.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `optim_stepsize` | 3e-4 | LR for both Alice and Bob |
| `alice_lr_min` | 5e-5 | Cosine decay floor for Alice |
| `cliprange` | 0.2 | PPO ε-clip (paper Table 2) |
| `noptepochs` | 3 | PPO mini-epochs |
| `nminibatches` | 4 | PPO minibatches per epoch |
| `ent_coef` | **0.05** | Alice entropy coef (fixed, Fix 2) |
| `gamma` | 0.998 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `abc_coef` | **0.5** | Bob BC loss weight (fixed, Fix 1) |
| `abc_traj_maxlen` | 500 | ABC trajectory store capacity |
| `abc_n_trajs` | 16 | Trajectories sampled per Bob update |
| `aux_coef` | 0.1 | GoalEncoder auxiliary distance loss |
| `goal_embed_dim` | 8 | GoalEncoder latent K |
| `num_bins` | 11 | Bins per MultiCategorical dimension |
| `num_cat_dims` | 6 | Action dims: X, Y, Z, Rx, Ry, Gripper |
| `lstm_hidden_size` | 256 | LSTM hidden state size |
| `alice_timesteps` | 100 | Steps per Alice phase |
| `bob_timesteps` | 200 | Steps per Bob phase |
