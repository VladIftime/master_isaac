# Network Architecture: Paper vs. Current Implementation

---

## Original OpenAI ASP Network
<!-- add reference to the paper -->


### Input Vectors

```
robot_joint_position = [joint1, joint2, joint3, joint4, joint5, joint6]
                         6 arm joint angles (radians)

gripper_position     = [tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, finger_position]
                         EE Cartesian pose (metres + radians) + finger opening

object_state         = [pos_x, pos_y, pos_z,
                         rot_roll, rot_pitch, rot_yaw,          ← Euler angles (3D)
                         vel_x, vel_y, vel_z,
                         rotvel_x, rotvel_y, rotvel_z,
                         gripper_distance, gripper_contact]     = 14D per object

goal_state (Bob only) = [desired_pos_x, desired_pos_y, desired_pos_z,
                          desired_rot_roll, desired_rot_pitch, desired_rot_yaw,  ← Euler (3D)
                          relative_distance]                    = 7D per object
```

### Forward Pass

```
robot_joint_position (6D)   →  Embedding Linear(6→256)   → LayerNorm(256)  ─┐
gripper_position     (7D)   →  Embedding Linear(7→256)   → LayerNorm(256)  ─┤
                                                                              Sum*
object_state (14D × N)      →  PI Embedding:                                 │
                                 shared Linear(14→512) → ReLU                │
                                 shared Linear(512→512) → ReLU               │
                                 sum-pool over N objects                      │
                                 LayerNorm(512)                             ──┤
                                                                              Sum*
goal_state (7D × N) [Bob]  →   PI Embedding (same structure as object)     ──┘

Sum* (256+256+512[+512 Bob]) → ReLU → MLP → LSTM → Actor head / Value head
```

---

## Current Implementation

### Input Vectors

```
ee_pose         = [ee_x, ee_y, ee_z, roll, pitch, yaw]
                    EE position (metres, env-local) + ZYX Euler angles
                    ─── Euler matches paper

gripper_state   = [finger_joint_angle]
                    Raw finger joint position (radians)

     robot_state (7D) = concat(ee_pose(6), gripper_state(1))
     ─── No joint angles: RMPFlow handles IK internally ───

object_state    = [pos_x, pos_y, pos_z,
                    roll, pitch, yaw,                          ← ZYX Euler (3D, not quat)
                    vel_x, vel_y, vel_z,
                    angvel_x, angvel_y, angvel_z,
                    gripper_distance, gripper_contact]         = 14D per object

goal_state (Bob only)
                = [desired_pos_x, desired_pos_y, desired_pos_z,
                    desired_roll, desired_pitch, desired_yaw]  = 6D per object (Euler, not quat)

goal_distance (Bob only)
                = [pos_dist,   ← L2(current_pos, goal_pos) in metres
                    rot_dist]  ← max |Euler diff| with wraparound, range [0, π]
                                                                             = 2D per object
```

**Assembled observation vectors:**

```
Alice obs = [robot_state(7) | obj1_state(14) | obj2_state(14)]          = 35D
Bob obs   = [robot_state(7) | obj1_state(14) | obj1_goal(6) | obj1_dist(2)
                            | obj2_state(14) | obj2_goal(6) | obj2_dist(2)]  = 51D
```
*(Interleaved per-object layout: each 22D chunk = state+goal+dist for one object.
 This matches the `view(batch, num_objects, 22)` reshape in `_encode_obs`.)*

### Forward Pass (Alice)

```
robot_state (7D) ───────────────────────────────────────────────────────────┐
                                                                             │
object_state (14D × 2 objects) →  PI Embedding (PermInvEncoder):           │ concat
                                     shared Linear(14→512) → LayerNorm → ReLU
                                     shared Linear(512→512) → LayerNorm → ReLU
                                     max-pool over 2 objects               │
                                     LayerNorm(512)  ← post-pool norm      │
                                                                            ─┘
                          concat [robot(7) | PI_pooled(512)] = 519D
                                     ↓
              actor_trunk: Linear(519→512) → ReLU → Linear(512→256) → ReLU → Linear(256→128)
                                     ↓
                          LSTMCell(128→256)
                                     ↓
              ┌────────────────────────────────┐
              ▼                                ▼
       Actor head                       Value head
       Linear(256→6×11=66)              Linear(35→512) → ReLU
       MultiCategorical                 → Linear(512→256) → ReLU
       (6 action dims × 11 bins)        → Linear(256→128) → ReLU → Linear(128→1)
```

**Action bins:**
```
dims 0-2: XYZ Cartesian delta   → (bin − 5) / 5 × max_delta_m        [max_delta_m = 0.04 m]
dims 3-4: Rx, Ry rotation delta → (bin − 5) / 5 × max_delta_rot       [max_delta_rot = 0.05 rad, clamped ±0.1 rad]
dim  5:   Gripper               → sticky: bins 0-2 → close (−1), bins 8-10 → open (+1), center → hold
```

### Forward Pass (Bob)

```
robot_state (7D) ───────────────────────────────────────────────────────────┐
                                                                             │
GoalEncoder (φ MLP, shared across objects):                                 │
  input per object: current_pose(6D) + goal_pose(6D)                       │
  φ: Linear(6→64) → Tanh → Linear(64→K=8)   ← no final activation        │
  g_i = φ(goal_i) − φ(current_i)  [difference variant]                    │
  g_pooled = sum-pool(g_0, g_1)              → 8D  (additive injection)    │
                                                                             │
PI Embedding (PermInvEncoder):                                              │
  input: ONLY obj_states (14D each) — goal enters via additive injection   │ concat
  shared Linear(14→512) → LayerNorm → ReLU                                 │
  shared Linear(512→512) → LayerNorm → ReLU                                │
  max-pool over 2 objects                                                   │
  LayerNorm(512)  ← post-pool norm                                         │
                                                                            ─┘
                    concat [robot(7) | PI_pooled(512)] = 519D
                                     ↓
  h₁ = Linear(519→512)(enc) + Linear(8→512, no bias)(g_pooled)  ← additive goal injection
  h₁ = ReLU(LayerNorm(h₁))
                                     ↓
       actor_trunk_rest: Linear(512→256) → ReLU → Linear(256→128)
                                     ↓
                          LSTMCell(128→256)
                                     ↓
              ┌────────────────────────────────┐
              ▼                                ▼
       Actor head                       Value head
       Linear(256→6×11=66)              Linear(51→512) → ReLU
       MultiCategorical                 → Linear(512→256) → ReLU
                                        → Linear(256→128) → ReLU → Linear(128→1)
                                        (full raw obs, no goal encoder bottleneck)
```

---

## Differences: Paper vs. Current

| | Paper | Current |
|--|-------|---------|
| **Robot arm state** | Joint angles (6D) | ❌ Removed — RMPFlow handles IK |
| **Gripper / EE state** | EE Cartesian pose + finger (7D) | EE Euler pose + finger (7D) |
| **Object rotation** | Euler angles (3D) | **Euler angles (3D)** — matches paper |
| **Object state dims** | 14D per object | **14D** per object |
| **Goal rotation** | Euler angles (3D) | **Euler angles (3D)** — matches paper |
| **Goal state dims** | 7D (3D pos + 3D euler + 1 scalar dist) | **6D** goal pose + **2D** dist (separate term) |
| **Goal encoding** | Raw PI embedding on goal states | **GoalEncoder → K=8 latent** per object |
| **GoalEncoder φ activation** | — | **Tanh** (paper §2.4) |
| **GoalEncoder input** | — | **6D Euler pose** (pos3 + euler3) |
| **GoalEncoder pooling** | — | **Sum-pool** (g = Σ g_i; "AND" semantics — all objects contribute) |
| **Additive goal injection** | ❌ | ✅ `h = ReLU(LN(W·enc + Wg·g))` |
| **PI encoder per-obj input** | 14D obj state | **14D obj state only** (goal separated out) |
| **Pooling (PI encoder)** | Sum-pool | **Max-pool** (more robust, standard DeepSets) |
| **Post-pool norm** | LayerNorm ✅ | LayerNorm ✅ |
| **Alice obs dim** | — | **35D** |
| **Bob obs dim** | — | **51D** |
| **Actor trunk** | MLP → LSTM | **Linear(519→512)→ReLU→(256)→(128) → LSTMCell(128→256)** |
| **Action space** | Continuous Gaussian | **MultiCategorical: 6 dims × 11 bins** |

---

## Rotation Representation Note

The current implementation uses ZYX Euler angles (roll, pitch, yaw) at observation time,
matching the paper's Appendix A.2 ("three Euler angles on three dimensions").
Quaternions are produced by IsaacSim but converted in `observations.py` before the
policy ever sees them.

The GoalEncoder's φ MLP therefore receives 6D inputs (pos3 + euler3) and computes
difference embeddings `φ(goal) − φ(current)` that are meaningful under linear arithmetic

# cuRobo Training Pipeline (`train_curobo.py`)

This section documents the full training pipeline for the cuRobo-IK variant of ASP, covering the DRL loop, the IK action pipeline, and all adaptive controllers.

---

## 1. Environment and Solver Setup

### Import ordering constraint

cuRobo must be imported **before** `AppLauncher`. Isaac Sim's `AppLauncher` prepends its own pip bundle to `sys.path`; importing cuRobo after would pick up Isaac Sim's incompatible torch. Additionally, `torch._dynamo`, `torch._C`, and `torch.optim` must be cached in `sys.modules` before `AppLauncher` runs to prevent dynamo conflicts.

```python
# CORRECT order:
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
import torch, torch._dynamo, torch._C, torch.optim  # lock correct versions
from isaaclab.app import AppLauncher
```

### Action space replacement (DiffIK → JointPositionActionCfg)

In `train_diffik.py` / `train.py`, the environment action is a 6D Cartesian delta processed by Isaac Lab's internal DiffIK controller. In the cuRobo variant, cuRobo computes joint positions externally, so the environment action is replaced with direct joint positions:

```python
env_cfg.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=_ARM_JOINT_NAMES,   # 6 UR5e joints
    scale=1.0,
    use_default_offset=False,       # action IS joint position in radians
)
```

The 7D `env_full` action vector contains `[q0..q5 (arm joints), gripper]`.

### cuRobo IK solver initialization

```python
_ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
_robot_cfg  = RobotConfig.from_dict(_ur5e_yaml["robot_cfg"], _tensor_args)
_ik_config  = IKSolverConfig.load_from_robot_config(_robot_cfg, world_model=None, tensor_args=_tensor_args)
ik_solver   = IKSolver(_ik_config)
```

**CUDA graph warm-up**: cuRobo traces a CUDA graph on the first `solve_batch()` call for a given batch size. A dummy warm-up call with `num_envs` is made at startup so all subsequent rollout calls reuse the graph (latency: ~3 ms cold → ~0.5 ms warm).

### Accumulated EE targets

Two per-env tensors accumulate the arm's target pose across steps:

| Tensor | Shape | Init | Meaning |
|--------|-------|------|---------|
| `ee_target_local` | `(N, 3)` | wrist_3_link pos − env_origin | EE position in env-local frame |
| `ee_target_quat_w` | `(N, 4)` | `[0, 1, 0, 0]` wxyz (tool-down) | EE orientation in world frame |

These are **integrals** — each step's decoded delta is added onto the running target. cuRobo then solves IK to that absolute target, not to a delta.

---

## 2. Action Decoding: Bins → IK Targets

The policy outputs `(N, 6)` integer bin indices from the MultiCategorical head. The decoder `_bins_to_xyz_rxy_gripper` maps them to continuous deltas:

```
Dim layout (11 bins, bin 5 = zero):
    0-2: XYZ    → (bin − 5) / 5 × max_delta_m        [max_delta_m = 0.04 m]
    3-4: Rx, Ry → (bin − 5) / 5 × max_delta_rot       [max_delta_rot = 0.05 rad, clamped to ±0.1 rad]
    5:   Gripper → sticky: bins 0-2 → close (−1), bins 8-10 → open (+1), center → hold
```

The gripper state is **sticky**: it holds its last value unless a decisive bin is selected. Gripper state is reset to open (1.0) on phase boundary or episode done.

---

## 3. Per-Step cuRobo IK Pipeline

Executed every rollout step for all `N` environments simultaneously.

**Step 1 — Decode actions**
```python
xyz, rxry, gripper_state = _bins_to_xyz_rxy_gripper(bin_indices, gripper_state)
```

**Step 2 — Accumulate EE position target**
```python
ee_target_local[envs] += xyz           # integrate delta onto running target
ee_target_local.clamp_(WS_X, WS_Y, WS_Z)  # clamp to workspace
```
Workspace limits (env-local frame): X ∈ [−0.50, 0.50], Y ∈ [0.25, 0.70], Z ∈ [0.00, 0.55].

**Step 3 — Accumulate EE orientation target**

Convert (Rx, Ry) deltas to a delta quaternion and compose with the running orientation:
```python
q_delta = quat_mul(q_ry, q_rx)        # Ry applied first, then Rx
ee_target_quat_w = quat_mul(ee_target_quat_w, q_delta)
ee_target_quat_w = normalize(ee_target_quat_w)
```
Quaternion composition avoids gimbal lock and makes phase sync trivial (just reset to `QUAT_TOOL_DOWN = [0, 1, 0, 0]` wxyz).

**Step 4 — TCP offset correction**

cuRobo targets the `wrist_3_link` frame; the gripper fingertip midpoint is ~5 cm ahead. The offset is computed live from current physics body positions so it tracks any orientation change:
```python
tcp_offset = (lf_w + rf_w) / 2 − w3_w    # finger midpoint minus wrist_3 (world frame)
ik_target  = ee_target_local − tcp_offset  # wrist_3 target needed to place TCP at goal
```
Note: `ee_target_local` and `tcp_offset` are both relative to the same origin, so `env_origins` cancel — this subtraction is valid.

**Step 5 — Batch IK solve**
```python
result = ik_solver.solve_batch(
    CuroboPose(position=ik_target, quaternion=ee_target_quat_w),
    seed_config=prev_joint_cmd.unsqueeze(1),   # (N, 1, 6) — warm-start from last command
    retract_config=prev_joint_cmd,              # (N, 6)
)
```
Seeding from `_prev_joint_cmd` (not physics joint positions) keeps cuRobo finding solutions along the already-smoothed trajectory.

**Step 6 — IK failure handling**

If `result.success[i]` is False for env `i`:
- Revert `ee_target_local[i]` and `ee_target_quat_w[i]` to their pre-step snapshot (prevents drift compounding).
- Fall back to `raw_cmd[i] = cur_joints[i]` (hold current physics state).

IK fail rate is logged to TensorBoard each iteration as `Metrics/IKFailRate`.

**Step 7 — EMA joint smoothing + assemble 7D command**
```python
smoothed = 0.2 * raw_cmd + 0.8 * prev_joint_cmd   # α = 0.2
env_full[:, :6] = smoothed                         # arm joints
env_full[:, 6]  = gripper_state                    # finger width
```
The EMA blends 20% of the new IK solution each step, giving inertia similar to RMPFlow's integrated velocity.

---

## 4. Phase Sync (After `env.step()`)

After each `env.step()`, Isaac Lab's PhysX state is fresh. Any environment where the phase transitioned (Alice→Bob or Bob→Alice) or an episode ended must have its IK accumulators snapped to the current physics state to prevent target drift:

```python
needs_sync = phase_changed | dones
if needs_sync.any():
    tcp_sync = (lf_w[sync_ids] + rf_w[sync_ids]) / 2
    ee_target_local[sync_ids]  = tcp_sync − env_origins[sync_ids]  # re-anchor position
    prev_joint_cmd[sync_ids]   = physics_joint_pos[sync_ids]        # re-anchor joints
    ee_target_quat_w[sync_ids] = QUAT_TOOL_DOWN                     # reset orientation
```

LSTM hidden states are zeroed on the same events:
- Alice hidden → zeroed when `dones[alice_ids]` or `_alice_just_ended[alice_ids]`
- Bob hidden → zeroed when `dones[bob_ids]` or `_bob_just_ended[bob_ids]`

---

## 5. Training Loop Structure

```
while bob_updates < max_iterations:

    ┌─ ROLLOUT (rollout_length = alice_timesteps + bob_timesteps steps) ─────────────┐
    │  for t in range(300):                                                           │
    │    ① Alice acts  (80% current policy + 20% historical snapshot)               │
    │    ② Bob acts    (80% current policy + 20% historical snapshot)               │
    │    ③ Record Alice trajectory for ABC (obs, act, traj_len per env)             │
    │    ④ cuRobo IK pipeline → 7D joint command                                    │
    │    ⑤ env.step(env_full) → obs, rewards, dones, extras                        │
    │    ⑥ Phase sync (on transition or done)                                       │
    │    ⑦ Add to Alice storage / Bob storage                                       │
    │    ⑧ On bob_done & ~bob_success & goal_valid → add to ABC buffer             │
    └────────────────────────────────────────────────────────────────────────────────┘

     ┌─ CONTROLLERS ──────────────────────────────────────────────────────────────────┐
     │  Alice entropy coef: fixed 0.05 (paper Table 2)                                │
     │  Alice LR: cosine decay lr_max=3e-4 → lr_min=5e-5                              │
     │  Bob abc_coef: fixed 0.5 (paper Table 2)                                       │
     └────────────────────────────────────────────────────────────────────────────────┘

    ┌─ PPO UPDATES ──────────────────────────────────────────────────────────────────┐
    │  Alice: standard PPO (3 epochs, 4 minibatches)                                 │
    │  Bob:   PPOABC = PPO + clipped ABC loss + aux GoalEncoder loss                 │
    └────────────────────────────────────────────────────────────────────────────────┘

    ┌─ CHECKPOINT (every save_interval iters) ───────────────────────────────────────┐
    │  model_{iter}.pt, abc_buffer.pt, episode_manager_{iter}.pt, train_state.pt     │
    └────────────────────────────────────────────────────────────────────────────────┘
```

### Rollout timing
- `alice_timesteps = 100` steps per Alice phase (goal-setting)
- `bob_timesteps = 200` steps per Bob phase (goal-replication)
- `rollout_length = 300` total steps per iteration
- Episodes are staggered at startup across envs so not all envs reset simultaneously.

### Historical policy pool
- `HIST_FRAC = 0.2` → for each policy's active env subset, 20% use a past snapshot.
- Snapshot saved every `HIST_SAVE_INTERVAL` iterations to `HistoricalPolicyPool`.
- `sample_policy()` reuses a persistent `_hist_clone` (load_state_dict, not deepcopy).

---

## 6. ABC Buffer Population

Gate condition (checked every step):
```python
just_failed_bob = bob_done_this_step & (~bob_success) & goal_valid
```

For each gated env:
1. Retrieve Alice's recorded trajectory `(obs[0:T], act[0:T])` for this episode.
2. Reject if `T < max(10, alice_timesteps // 2)` (too short to be a valid demo).
3. Construct Bob's BC observation: `bc_obs = construct_bob_observation(alice_obs, goal_states)` — repackages Alice's obs with the goal appended to each timestep.
4. Compute `old_log_prob` for the trajectory under current Bob policy (batch evaluate).
5. Store `(bc_obs, traj_acts, old_lp)` in `GPUDemonstrationBuffer` (sliding window, `traj_maxlen=500`).

Old trajectories are evicted as new failures arrive, keeping BC signal relevant to Alice's current task distribution.

---

## 7. Fixed Controllers

All controllers use fixed values per paper Table 2 (Fix 1, Fix 2).  They are set once
per iteration, after the rollout and before the PPO updates.

### Alice entropy coef

Fixed at the YAML value (`ent_coef: 0.05` per `ppo_continuous.yaml`).  The two-phase
SR-coupled controller (exponential decay + PI controller) has been removed to prevent
a vicious feedback loop causing premature mode collapse.

### Alice learning rate

Cosine decay over the full training run:
```
alice_lr = alice_lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × iter/max_iter))
```
Range: `lr_max = 3e-4` → `lr_min = 5e-5`.

### Bob abc_coef

Fixed at the paper Table 2 value (β = 0.5).  The inverse-SR controller
(`target_abc = abc_coef_start × (1 − bob_sr)`) has been removed — it injected a
parasitic second-order feedback loop that destroyed the trust region.

---

## 8. PPO Updates

### Alice update (standard PPO)
- `noptepochs = 3`, `nminibatches = 4`, `cliprange = 0.2`
- Entropy bonus: `ent_coef × H(π)` (fixed, see controller above)
- No ABC or aux loss.

### Bob update (PPOABC)

**PPO loss** (clipped surrogate + value):
```
L_PPO = −min(r·A, clip(r, 1−ε, 1+ε)·A) + vf_coef × (V − R)²
```

**ABC loss** (clipped behavioral cloning, sequential LSTM evaluation):

Trajectories sampled from `GPUDemonstrationBuffer` are evaluated sequentially through the LSTM (not batched) to maintain correct hidden state dependencies:
```python
for t in range(max_T):
    logits, (h, c) = actor_forward(obs[t], (h, c), detach_goal_encoder=True)
    lp = MultiCategorical(logits).log_prob(acts[t])
bc_ratio = exp(lp − old_lp)
L_BC = −mean(min(bc_ratio, clip(bc_ratio, 1−ε, 1+ε)))   # PPO-clipped NLL
```
`detach_goal_encoder=True` prevents ABC from distorting the latent goal representation.

**Auxiliary distance loss** (GoalEncoder regularizer):
```
L_aux = aux_coef × GoalEncoder.aux_loss(goal_poses, current_poses)
```
Trains the encoder to also predict position and rotation distances — adds a geometric inductive bias without a separate training phase.

**Total Bob loss**:
```
L_total = L_PPO + abc_coef × L_BC + aux_coef × L_aux
```

---

## 9. Observation Layouts

### Alice obs (flat, `alice_obs_dim`)
```
[robot_state(7) | obj1_state(14) | obj2_state(14)]
= 35D (2 objects)
```
No goal info. `robot_state` = `[ee_pose(6), gripper(1)]` (ZYX Euler, env-local frame).

### Bob obs (flat, `bob_obs_dim`)
```
[robot_state(7) | obj1_state(14) + goal1_pose(6) + dist1(2) | obj2_state(14) + goal2_pose(6) + dist2(2)]
= 7 + 2×22 = 51D (2 objects)
```
`obj_state(14)` = `[pos(3), euler(3), linvel(3), angvel(3), dist(1), contact(1)]`.
`goal_pose(6)` = `[pos(3), euler(3)]`.
`dist(2)` = `[pos_dist(1), rot_dist(1)]`.

---

## 10. Network Forward Pass (cuRobo training, no architectural changes)

The network architecture is identical to the DiffIK variant — cuRobo only replaces the IK controller outside the policy.

### Alice forward pass
```
alice_obs (35D)
    │
    ├─ robot_state (7D) ─────────────────────────────────────────────┐
    │                                                                  │
    └─ obj_features (28D → 2×14D) → PermInvEncoder                   │
           shared MLP: 14→512→512 (LN+ReLU each)                     │
           max-pool across 2 objects                                   │
           pool_norm (LayerNorm)                                       │
           concat robot_state → (7+512 = 519D)                        │
               │                                                        │
               └──────────────────────────────────────────────────────┘
                                    │ (519D)
                               actor_trunk
                               Linear(519→512)→ReLU
                               Linear(512→256)→ReLU
                               (no goal injection — Alice has no goal)
                                    │ (256D)
                               LSTMCell(256→256)
                                    │ (256D)
                               actor_head: Linear(256→66)   [6 dims × 11 bins]
                               reshape → (batch, 6, 11) → MultiCategorical
```

### Bob forward pass (with GoalEncoder + additive injection)
```
bob_obs (51D)
    │
    ├─ robot_state (7D)
    │
    ├─ per-object chunks: 2 × [obj_state(14) + goal_pose(6) + dist(2)]
    │       │                           │
    │       │ obj_state (14D)           │ goal_pose (6D) + current_pose (6D from obj_state)
    │       │     │                     │
    │       │     │              GoalEncoder.encode_per_object()
    │       │     │              MLP variant (default: "difference")
    │       │     │              per-object embedding: (batch, 2, K=8)
    │       │     │              sum-pool across objects → g_pooled (batch, 8)
    │       │     │
    │       └─ obj_state (14D×2) → PermInvEncoder (same as Alice)
    │              shared MLP: 14→512→512
    │              max-pool + pool_norm
    │              concat robot_state → (519D)
    │
    │ (519D from PI encoder)
    actor_trunk_layer1: Linear(519→512)       ← first layer split out
    h1 = ReLU(h1)
    h1 = LayerNorm(h1 + goal_proj(g_pooled))  ← additive goal injection
    actor_trunk_rest: Linear(512→256)→ReLU   ← remaining trunk layers
    │ (256D)
    LSTMCell(256→256)
    │ (256D)
    actor_head: Linear(256→66) → (batch, 6, 11) → MultiCategorical

critic (no bottleneck, always full raw obs):
    obs (51D) → MLP(51→512→256→128→1)
```

Goal projection: `_goal_proj = Linear(8→512, bias=False)` scaled down by ×0.1 at init to avoid ReLU saturation before training begins.

---

## 11. Checkpoint Files

Each checkpoint saves four artifacts:

| File | Contents |
|------|----------|
| `bob/model_{iter}.pt` | Bob ActorCritic weights + optimizer state |
| `alice/model_{iter}.pt` | Alice ActorCritic weights + optimizer state |
| `bob/abc_buffer.pt` | GPUDemonstrationBuffer trajectory store |
| `bob/episode_manager_{iter}.pt` | Phase step counters, goal states |
| `bob/train_state_{iter}.pt` | `entropy_coef`, `abc_coef`, `bob_success_buf` (deque) |

On `SIGTERM`, the same set is written immediately before exit (emergency checkpoint).

Resume: `--resume_path bob/model_{iter}.pt` loads both Alice and Bob weights, optimizer state, and train_state from the matching iteration.
