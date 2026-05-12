# Implementation Record — ASP + GoalEncoder + Push-PPO Baseline

**Branch**: `asp_goal_encoder`  
**Last updated**: 2026-05-11 (Push primitive refactor: no yaw/grip/spin, fixed offset calibration)

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

Bob (PPOABC + GoalEncoder, 51D obs, 200 steps)
  → must reproduce Alice's configuration from scratch
  → goal enters via GoalEncoder latent (8D) injected into actor trunk
  → reward: +1 per object at goal, −1 if object leaves goal, +5 completion

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
  - IK failure recovery: reverts EE accumulator, holds current joint positions
  - Phase sync: re-anchors accumulators to physics TCP state after phase transition or episode done
  - Workspace clamp: X ∈ [−0.50, 0.50], Y ∈ [0.25, 0.70], Z ∈ [0.00, 0.55] metres (env-local)
  - EE home offset applied after every sync (reset / phase boundary): X += 0.02 m, Z = 0.05 m — arm resets to its default joint configuration then IK drives it to the preferred low-hover resting pose in the first few steps
  - cuRobo CUDA graph warm-up before training loop (~3 ms → ~0.5 ms per step)
  - IK fail rate logged to TensorBoard (`Metrics/IKFailRate`) each iteration

- **Two-phase ASP structure**:
  - Alice phase (100 steps): explores, builds goal state
  - Bob phase (200 steps): reproduces goal, receives sparse reward
  - Phase stagger at startup: random offset across envs prevents simultaneous resets
  - LSTM hidden states zeroed on phase transitions and episode done events

- **Alice Behavioral Cloning (ABC)**:
  - Alice trajectory buffer: `(alice_traj_obs, alice_traj_act, alice_traj_len)` per env per rollout
  - Gate: `bob_done & ~bob_success & goal_valid & traj_len >= max(10, alice_timesteps//2)`
  - `GPUDemonstrationBuffer`: sliding window of 500 trajectories, evicts oldest on overflow
  - Buffer persisted across checkpoints (`abc_buffer.pt`), loaded on resume

- **Historical policy pool**:
  - 20% of active envs per phase use a past policy snapshot (`HIST_FRAC = 0.2`)
  - Pool holds last 5 snapshots; saved every 50 iterations

- **Fixed controllers** (SR-coupled controllers removed — Fixes 1 & 2):
  - Alice entropy coef: `ent_coef = 0.05` (fixed from YAML)
  - Alice LR: cosine decay `lr_max=3e-4 → lr_min=5e-5` over `max_iterations`
  - Bob `abc_coef`: fixed at 0.5 (paper Table 2)

- **Diagnostic flags**: `--test_reward_pipeline`, `--alice_sandbox`, `--dummy_alice`, `--profile`, `--test_hparams`

- **Checkpoint system**: periodic, best-model, SIGTERM emergency; resume via `--chkpt_alice/bob`

- **TensorBoard logging**: Loss, Reward, Metrics, GoalEncoder, ABC, IK overhead — full set

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
- Alice per-step rewards unconditionally 0.0 (Fixes 3 & 11)
- Bob reward: sparse `{+1/−1/+5}` only, no potential shaping (Fix 4)
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
| `train_curobo.slurm` | Production cuRobo training run |
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
| `_EE_HOME_X_OFFSET` | 0.02 m | X offset added to IK target after every sync (home pose) |
| `_EE_HOME_Z` | 0.05 m | Fixed Z of IK target after every sync (5 cm above table) |

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
- **IK failure during early training** — mitigation: episode reset on IK failure, dense EE-to-object reward teaches workspace awareness
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
| Orientation | Fixed tool-down `[0,1,0,0]` | No yaw rotation — simplifies IK, all pushes face forward |
| Gripper | Always closed during push | Simplest possible baseline |
| Action space | Relative offsets from object | Generalizes across object positions |
| Action frequency | Macro-action | Agent predicts push params once; env executes multi-step trajectory |

### 6.3 Push Primitive Architecture

```
Phase 1: Approach   (18 steps)  EE → above object, tool-down, gripper open
Phase 2: Engage     ( 5 steps)  Close gripper at approach height
Phase 3: Descend    (20 steps)  EE down to contact height (table + 0.015 m)
Phase 4: Push       (30 steps)  EE moves: contact_xy → contact_xy + (push_dx, push_dy)
Phase 5: Retract    (10 steps)  EE up to approach height, gripper closed
Phase 6: Release    ( 5 steps)  Open gripper at approach height
Phase 7: Return     (12 steps)  EE back to current TCP position at approach height

Total: 100 substeps per push macro-action (2.0 s at 50 Hz)
```

### 6.4 Action Space

MultiCategorical: **6D × 11 bins**

| Dim | Parameter | Range |
|-----|-----------|-------|
| 0 | `approach_offset_x` | [-0.15, 0.15] m |
| 1 | `approach_offset_y` | [-0.15, 0.15] m |
| 2 | `push_dx` | [-0.30, 0.30] m |
| 3 | `push_dy` | [-0.30, 0.30] m |
| 4 | `yaw` | [-π, π] rad (predicted but NOT used by waypoint generator) |
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
R = α·(d_prev−d_now)  +  γ·(r_prev−r_now)  −  β·d_now  +  completion_bonus
```

where:
- `d_prev` / `d_now` = L2 position error before / after the push (metres)
- `r_prev` / `r_now` = max absolute Euler-angle difference before / after (radians, wraparound-aware)
- `α = 10.0` — position improvement gain (symmetric: rewards getting closer, penalizes moving away)
- `γ = 2.0` — rotation improvement gain (lower weight because 1 rad ≈ 57° is harder to change than 1 m via planar pushing)
- `β = 0.5` — distance penalty per step (keeps episodes short, prevents passive exploration)
- `completion_bonus = +5.0` when object enters goal zone (pos < 0.05 m, rot < 0.035 rad ≈ 2°)

**Design rationale** (Akella & Mason 1998, "Posing Polygonal Objects in the Plane by Pushing", IJRR):
off-center pushes induce torque — the `(offset_x, offset_y)` parameters create a moment arm
relative to the object's center of mass. The agent can learn to chain pushes (e.g. push right
side to spin CCW, then centered push to translate) to achieve any target pose. The symmetric
improvement terms (no `max(0,·)` clipping) prevent reward hacking by penalizing regression
equally. The `−β·d_now` penalty provides continuous pressure to finish episodes efficiently.

### 6.7 Network Architecture

```
obs (29D)
  │
  ├─ Linear(29 → 512) → ReLU
  ├─ Linear(512 → 256) → ReLU
  ├─ LSTM(256 → 256)
  │
  ├─ Actor head:  Linear(256 → 66) → (6, 11) → MultiCategorical
  └─ Critic head: Linear(256 → 128) → ReLU → Linear(128 → 1)
```

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

### 6.9 Running

```bash
# Non-headless (with viewer)
python -m asyncDualPlayPPO.train_push \
    --num_envs 64 --max_iterations 500 --exp_name push_baseline

# Headless
python -m asyncDualPlayPPO.train_push \
    --num_envs 64 --max_iterations 500 --exp_name push_baseline --headless
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
│  ② compute_push_waypoints() → 100 waypoints         │
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
| Steps per push | 100: 18+5+20+30+10+5+12 |
| TCP offset | Calibrated fixed offset at startup (30-step PD settle) |
| Workspace | X=[-0.5,0.5], Y=[0.25,0.70], Z=[0.00,0.55] |
| Pause between pushes | 60 steps (~1.2 s) |

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