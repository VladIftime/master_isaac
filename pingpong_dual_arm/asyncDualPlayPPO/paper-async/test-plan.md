# Pre-Pipeline Diagnostic Test Implementation Plan

**Execution order:** Test 1 → Test 2 → Test 3 → Test 4. Each test is a hard gate; do not proceed if it fails.

---

## Infrastructure notes (already in place)

The following metrics are already logged to tensorboard in `train_curobo.py` — the tests read from these rather than adding new logging:

| Scalar key | Source |
|---|---|
| `Metrics/Alice/ValidGoals` | `wrapper._iter_stats["valid_goals"]` |
| `Loss/Bob/ABC` | `ppo_abc.update()` return `mean_bc_loss` |
| `Loss/Bob/Surrogate` | `ppo_abc.update()` return `mean_surrogate_loss` |
| `Bob/ABCCoef` | `bob_ppo.abc_coef` |
| `Metrics/ABC/BufferSize` | `bob_ppo.abc_buffer.size` |
| `GoalEncoder/embedding_norm` | sampled from `actor_forward` |
| `Metrics/Bob/SuccessRate` | `_compute_bob_sparse_rewards` SR |

`_euler_xyz_to_quat` and `_quat_to_euler_xyz` already exist in `asyncDualPlayPPO/tasks/utils/observations.py`.

---

## Test 1 — Reward Pipeline Integrity via Teleportation

### Goal

Verify `_compute_bob_sparse_rewards` fires `+1` / `−1` / `+5` at the right thresholds when objects are placed exactly at goal coordinates, with no robot involvement.

### New file

`asyncDualPlayPPO/diagnostics/test_reward_pipeline.py`

### Approach

The teleportation hook intercepts the first step of Bob's phase. Instead of using the actor's output, it reads `episode_manager.goal_states` (shape `(N, num_objects × 6)`, format `[pos(3) + euler_xyz(3)]` per object), converts to `(pos, quat)`, and calls `write_root_pose_to_sim` + `write_root_velocity_to_sim(zeros)` before the env processes physics. The env then steps normally and the reward path fires.

#### Implementation steps

**Step 1 — Create `TeleportBobStep` callable**

```python
# diagnostics/test_reward_pipeline.py
from asyncDualPlayPPO.tasks.utils.observations import _euler_xyz_to_quat

def teleport_to_goals(env, episode_manager):
    """Teleport all objects to their current goal poses."""
    goal = episode_manager.goal_states          # (N, num_obj * 6)
    if goal is None:
        return
    origins = env.env.scene.env_origins        # (N, 3)
    zero_vel = torch.zeros(env.num_envs, 6, device=env.device)

    for obj_idx, obj_name in enumerate(["target_object", "cube"][:episode_manager.num_objects]):
        g = goal[:, obj_idx * 6 : obj_idx * 6 + 6]   # (N, 6)
        pos_local = g[:, :3]
        quat      = _euler_xyz_to_quat(g[:, 3:6])     # (N, 4)  w,x,y,z
        pos_world = pos_local + origins
        pose      = torch.cat([pos_world, quat], dim=1)
        obj = env.env.scene[obj_name]
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(zero_vel)
```

**Step 2 — Run diagnostic loop**

```python
def run_test1(env, episode_manager, n_iterations=50):
    sr_history = []
    for it in range(n_iterations):
        # Wait until at least one env enters Bob phase
        obs, _, dones, info = env.step(torch.zeros(env.num_envs, env.action_space.shape[0]))

        if episode_manager.is_bob_phase().any():
            # Teleport objects to goals for Bob-phase envs
            teleport_to_goals(env, episode_manager)
            # Step once — reward pipeline fires here
            obs, rew, dones, info = env.step(torch.zeros_like(...))
            sr = info.get("bob_success_rate", 0.0)
            sr_history.append(sr)
            print(f"[T1] iter {it:3d}  SR={sr:.3f}  max_rew={rew.max():.2f}")

    assert max(sr_history) > 0.0, "FAIL: SR never exceeded 0 — reward tensor is broken"
    assert any(r >= 1.0 for r in ...), "FAIL: +1 sub-reward never triggered"
    print("[T1] PASS")
```

**Step 3 — Entry point**

Add flag `--test_reward_pipeline` to `train_curobo.py` just before the main training while-loop:

```python
if args.test_reward_pipeline:
    from asyncDualPlayPPO.diagnostics.test_reward_pipeline import run_test1
    run_test1(env, env.episode_manager)
    exit(0)
```

Or run standalone (requires sim init):

```
python -m asyncDualPlayPPO.train --test_reward_pipeline --num_envs 16 --headless
```

### Specific reward events to assert

| Event | Trigger | Expected value |
|---|---|---|
| Object enters goal zone | `pos_dist < 0.05` AND `rot_dist < 0.2` | `+1.0` in `step_rewards` |
| Object leaves goal zone | transitions from success → failure | `−1.0` |
| All objects at goal simultaneously | `all_success & ~completion_given` | `+5.0` in `completion_bonus` |
| Teleport to exact goal | zero pos/rot error by construction | SR = 1.0 within first Bob step |

### Distance metric contract to verify

- **pos_dist**: Euclidean `‖pos_goal − pos_curr‖₂` in metres — found in `observations.py:goal_distance`
- **rot_dist**: max-abs-diff of ZYX Euler angles with `[0, π]` wraparound — must verify that `abs_diff = min(|Δ|, π − |Δ|)` is used, not raw `|Δ|`; grep for this pattern in `observations.py` / `wrapper.py` and add an explicit unit test:

```python
# Unit test for wraparound
assert rot_distance(euler=[0, 0, 0],   goal=[0, 0,  3.13]) < 0.02   # near-zero, not near-2π
assert rot_distance(euler=[0, 0, 3.0], goal=[0, 0, -3.0]) < 0.30   # crosses ±π boundary
```

### Pass / Fail

- **PASS**: SR > 0 after iteration 1; `+5` reward observed; no exceptions thrown
- **FAIL**: SR = 0.0 throughout; reward tensor returns zeros after exact teleport

---

## Test 2 — Alice Exploration Sandbox

### Goal

Verify Alice generates an increasing number of valid goals over 200 iterations, out-of-zone penalties trigger correctly, and the curriculum expands (goal displacement distribution widens over time).

### New file

`asyncDualPlayPPO/diagnostics/test_alice_sandbox.py`

### What is already tracked

`wrapper._iter_stats["valid_goals"]` is incremented in `_validate_and_store_goal` (wrapper.py:590) and logged as `Metrics/Alice/ValidGoals` every iteration. No new instrumentation is needed for the counter itself.

The out-of-zone penalty (`−3`) is applied at wrapper.py:390 and flows into `alice_total_reward`.

### Approach

Run a normal training invocation with Bob's PPO update disabled so that Alice's exploration is not influenced by Bob's improving policy. This isolates the curriculum emergence signal.

**Step 1 — Add `--alice_sandbox` flag to `train_curobo.py`**

In the `perform_bob_update` closure (train_curobo.py:614), guard the update:

```python
def perform_bob_update(current_bob_obs):
    if args.alice_sandbox:
        return  # skip all Bob PPO updates; Bob takes random actions
    ...
```

When `--alice_sandbox` is set, replace Bob's actor with a random policy:

```python
if args.alice_sandbox:
    bob_actions = torch.randint(0, 11, (env.num_envs, 6), device=device).float()
```

**Step 2 — Capture per-iteration stats**

After each iteration, read `env.get_iter_stats()` which already contains:

```python
stats = env.get_iter_stats()
valid_goals   = stats["valid_goals"]       # already logged to TB
invalid_goals = stats["invalid_goals"]
alice_total   = stats["alice_total"]
disp_3d_sum   = stats["alice_disp_3d_sum"] # sum of 3D displacements for valid goals
```

Compute per-iteration metrics:
- `goal_validity_rate = valid_goals / max(1, valid_goals + invalid_goals)`
- `mean_disp = disp_3d_sum / max(1, valid_goals)` — tracks if goals are getting harder

**Step 3 — Run and assert**

```
python -m asyncDualPlayPPO.train --alice_sandbox --num_envs 32 --max_iterations 200 --headless
```

After the run, read the tensorboard event file and check:

```python
from torch.utils.tensorboard import SummaryWriter  # or use tbparse
# Valid goals must trend upward: compare first 20 iters vs last 20 iters
early_avg = mean(valid_goals[0:20])
late_avg  = mean(valid_goals[180:200])
assert late_avg > early_avg, "FAIL: Valid Goals not climbing"
```

**Step 4 — Log additional sandbox metrics**

Add to `train_curobo.py` inside the existing metrics block (around line 1269):

```python
writer.add_scalar("Metrics/Alice/GoalValidityRate", _valid_goals / max(1, _valid_goals + _invalid_goals), bob_updates)
writer.add_scalar("Metrics/Alice/MeanDisp3D", _stats.get("alice_disp_3d_sum", 0) / max(1, _valid_goals), bob_updates)
```

These are the two most diagnostic signals for curriculum health.

### What to watch

| Metric | Expected trajectory | Red flag |
|---|---|---|
| `Metrics/Alice/ValidGoals` | Climbing over 200 iters | Flat at 0 → Alice not moving objects |
| `Metrics/Alice/GoalValidityRate` | Starts low (~20–40%), climbs to >60% | Stays <10% → all goals OOB |
| `Reward/Alice` | Initially near 0, grows positively | Always 0 → reward gating broken |
| `Alice/EntropyCoef` | Fixed at 0.01 (confirmed by Fix 2) | Any deviation → entropy controller not removed |
| `Metrics/Alice/MeanDisp3D` | Gradually increases as Alice gets bolder | Stays constant → curriculum not expanding |

### Pass / Fail

- **PASS**: `late_avg > early_avg` for Valid Goals; out-of-zone −3 penalties appear in logs; no NaN rewards
- **FAIL**: Valid Goals flat or zero; `alice_total_reward` always 0

---

## Test 3 — PPO and ABC Optimization Balance

### Goal

Verify that after Fix 1 (constant β) and Fix 5 (combined loss), the ABC coefficient is fixed at β=0.5, ABC loss is non-zero from iteration 1, the trajectory buffer grows and contains only failed episodes, and the combined loss produces stable learning.

> **Note on divergence from test description:** The test description in the brief describes the old SR-coupled abc_coef controller (Phase 1 linear decay + Phase 2 EMA). That controller was removed in Fix 1 (divergence from paper Table 2). This test instead validates the corrected behavior: β=0.5 constant, ABC active from the first non-empty buffer fill.

### New files

- `asyncDualPlayPPO/diagnostics/test_ppo_abc_balance.py` — reads TB logs and asserts on metrics
- `asyncDualPlayPPO/diagnostics/test_checkpoint_chain.py` — buffer persistence smoke test

### Approach

**Step 1 — Run 50 iterations of full pipeline**

```
python -m asyncDualPlayPPO.train --num_envs 32 --max_iterations 50 --headless
```

This generates a tensorboard run and checkpoint files.

**Step 2 — `test_ppo_abc_balance.py`: read TB event file and assert**

```python
import glob
from tbparse import SummaryReader  # or manual protobuf parsing

run_dir = "<log_dir>"
reader  = SummaryReader(run_dir)
df      = reader.scalars

# 1. ABCCoef must be constant 0.5
abc_coef = df[df.tag == "Bob/ABCCoef"]["value"]
assert abc_coef.std() < 1e-4,  "FAIL: abc_coef is not constant — SR-coupling still active"
assert abs(abc_coef.mean() - 0.5) < 0.01, "FAIL: abc_coef != 0.5"

# 2. ABC loss must be >0 once buffer has ≥1 trajectory
buf_size = df[df.tag == "Metrics/ABC/BufferSize"]["value"]
first_warm_iter = buf_size[buf_size > 0].index[0]
abc_loss_after_warm = df[(df.tag == "Loss/Bob/ABC") & (df.step >= first_warm_iter)]["value"]
assert (abc_loss_after_warm > 0).all(), "FAIL: ABC loss zero despite non-empty buffer"

# 3. Surrogate loss must be finite and non-zero
surr = df[df.tag == "Loss/Bob/Surrogate"]["value"]
assert surr.isfinite().all(),   "FAIL: non-finite surrogate loss detected"
assert (surr.abs() > 1e-6).any(), "FAIL: surrogate loss is identically zero"

# 4. Loss ratio: ABC should not dominate PPO
ratio = abc_loss_after_warm.values / (surr.values[-len(abc_loss_after_warm):] + 1e-8)
assert ratio.mean() < 5.0, "WARN: ABC loss is >5× surrogate — may overwrite PPO signal"
```

**Step 3 — Verify trajectory buffer filter**

Add a one-time assertion inside `train_curobo.py` (debug mode only):

```python
# After abc_buffer.add_trajectory():
# Check that the just-added trajectory corresponds to a failed episode
assert (just_failed_bob & ~bob_success & goal_valid).any(), \
    "BUG: trajectory added to ABC buffer from a successful episode"
```

Also verify buffer caps at `traj_maxlen` by sampling `bob_ppo.abc_buffer.size` after 50 iterations.

**Step 4 — `test_checkpoint_chain.py`: buffer persistence**

```python
# Run 10 iterations, save checkpoint
train(..., max_iterations=10, save_interval=10)

# Load checkpoint
bob_ppo_restored = PPOABC(...)
bob_ppo_restored.load("<ckpt_path>")
buf_path = "<log_dir>/abc_buffer.pt"
bob_ppo_restored.abc_buffer.load(buf_path)

# Verify buffer size matches original
assert bob_ppo_restored.abc_buffer.size == original_size, \
    "FAIL: buffer size changed after checkpoint round-trip"

# Verify trajectory shapes are intact
for traj in bob_ppo_restored.abc_buffer._traj_store:
    assert "obs" in traj and "acts" in traj and "log_probs" in traj
    assert traj["obs"].ndim == 2   # (T, obs_dim)
    assert traj["acts"].ndim == 2  # (T, 6)
```

### Pass / Fail

| Check | PASS | FAIL |
|---|---|---|
| `Bob/ABCCoef` | Constant 0.5 ± 0.001 | Any variation across iterations |
| `Loss/Bob/ABC` | > 0 after first non-empty buffer | Zero despite buffer size > 0 |
| Buffer filter | Only failed+invalid trajectories in store | Any successful episode in store |
| Checkpoint round-trip | Buffer size and shapes preserved | Size changes or `KeyError` on load |

---

## Test 4 — Goal Encoder Latent Space

### Goal

Verify the GoalEncoder produces semantically interpretable embeddings: t-SNE shows geometric cluster separation, noise perturbation does not break cluster membership, and integration with the actor forward pass is end-to-end differentiable.

### New files

- `asyncDualPlayPPO/diagnostics/test_goal_encoder_latent.py` — embedding extraction + t-SNE + silhouette
- `asyncDualPlayPPO/diagnostics/test_abc_goal_encoder.py` — forward pass integration test (no training needed)

### Prerequisite

A checkpoint from at least 500 training iterations (so the GoalEncoder has seen enough goal diversity). Tests 1–3 together provide ~300 iterations; run an additional 200 before this test.

### `test_abc_goal_encoder.py` — Integration test (run first, no GPU required for logic check)

```python
# Verify end-to-end: obs → _encode_obs → goal_encoder.encode_per_object → actor_forward
# and that ABC loss backpropagates into GoalEncoder parameters after Fix 14

model = ActorCritic(...)
model.load("<ckpt>")

obs_batch = torch.randn(16, obs_dim)  # dummy batch

# Forward pass should not crash
enc, g_pooled = model._encode_obs(obs_batch, detach_goal_encoder=False)
assert g_pooled is not None, "FAIL: goal encoder returned None"
assert g_pooled.shape == (16, K_per_obj), f"FAIL: wrong embedding shape {g_pooled.shape}"

# Check gradient flows into GoalEncoder via ABC loss (Fix 14)
loss = g_pooled.sum()
loss.backward()
for name, p in model.goal_encoder.named_parameters():
    assert p.grad is not None, f"FAIL: no gradient in GoalEncoder.{name}"

print("[T4-integration] PASS")
```

### `test_goal_encoder_latent.py` — Embedding extraction + t-SNE

**Step 1 — Collect embeddings**

Load the trained model in inference mode. Run Alice's policy for 2000+ steps across 32 envs. After each Alice phase completes (goal validated), record:

```python
embeddings = []
labels     = []   # heuristic: "planar" / "lifted" / "rotated"

# Inside the data-collection loop:
if episode_manager.goal_states is not None and valid_goals.any():
    goal = episode_manager.goal_states[valid_goals]   # (k, num_obj * 6)
    curr = get_current_object_poses(env)[valid_goals] # (k, num_obj * 6)

    goal_flat = goal.reshape(len(goal), -1)
    curr_flat = curr.reshape(len(goal), -1)

    with torch.no_grad():
        g = model.goal_encoder(goal_flat, curr_flat)  # (k, K_per_obj)
    embeddings.append(g.cpu())

    # Heuristic label
    z_displacement = (goal[:, 2] - curr[:, 2]).abs()
    rot_diff = (goal[:, 3:6] - curr[:, 3:6]).norm(dim=-1)
    for i in range(len(goal)):
        if z_displacement[i] > 0.05:
            labels.append("lifted")
        elif rot_diff[i] > 0.3:
            labels.append("rotated")
        else:
            labels.append("planar")

embeddings = torch.cat(embeddings).numpy()   # (M, K_per_obj)
```

**Step 2 — t-SNE**

```python
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_2d = tsne.fit_transform(embeddings)   # (M, 2)

# Silhouette score on heuristic labels
label_ids = [{"planar": 0, "lifted": 1, "rotated": 2}[l] for l in labels]
sil = silhouette_score(X_2d, label_ids)
print(f"t-SNE silhouette score: {sil:.3f}  (target > 0.15)")

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
for lbl, color in [("planar", "blue"), ("lifted", "red"), ("rotated", "green")]:
    mask = [l == lbl for l in labels]
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=lbl, s=8, alpha=0.6)
ax.legend()
plt.savefig("goal_encoder_tsne.png", dpi=150)
```

**Step 3 — Noise invariance**

```python
sigma = 0.02  # 2 cm positional noise
perturbed_goal = goal_flat + sigma * torch.randn_like(goal_flat)
with torch.no_grad():
    g_perturbed = model.goal_encoder(perturbed_goal, curr_flat)
delta_norm = (g_perturbed - g).norm(dim=-1)  # (k,)

# Delta should be small relative to embedding magnitude
rel_change = (delta_norm / (g.norm(dim=-1) + 1e-6)).mean().item()
print(f"Relative embedding change under 2cm noise: {rel_change:.4f}  (target < 0.20)")
assert rel_change < 0.20, "FAIL: GoalEncoder is too sensitive to small positional perturbations"
```

**Step 4 — Embedding stats sanity check**

These are already logged to tensorboard during training (`GoalEncoder/embedding_norm`, `GoalEncoder/embedding_std`). Verify after 500 iterations:

- `embedding_norm` should be stable (not collapsing to 0, not exploding)
- `embedding_std` should be > 0 (encoder is not outputting constant vectors)

```python
norm_vals = df[df.tag == "GoalEncoder/embedding_norm"]["value"]
std_vals  = df[df.tag == "GoalEncoder/embedding_std"]["value"]
assert norm_vals.iloc[-20:].mean() > 0.1, "FAIL: embedding collapsed to zero"
assert std_vals.iloc[-20:].mean() > 0.01, "FAIL: encoder outputs constant vector"
```

### Pass / Fail

| Check | PASS | FAIL |
|---|---|---|
| Integration forward pass | No exceptions; g_pooled shape `(B, K)` | AttributeError, None, wrong shape |
| Gradient flow (Fix 14) | GoalEncoder params have `.grad` after backward | Any param grad is None |
| t-SNE silhouette | > 0.15 | < 0 (worse than random) |
| Noise invariance | Relative change < 0.20 under 2cm noise | > 0.50 (encoder not robust) |
| Embedding norm | Stable and > 0.1 | Collapses or diverges |

---

## Execution script

Create `asyncDualPlayPPO/diagnostics/run_diagnostics.sh`:

```bash
#!/usr/bin/env bash
set -e

ENVS=16
LOG="diagnostics_run"

echo "=== Test 1: Reward Pipeline ==="
python -m asyncDualPlayPPO.train --test_reward_pipeline --num_envs $ENVS --headless

echo "=== Test 2: Alice Sandbox ==="
python -m asyncDualPlayPPO.train --alice_sandbox --num_envs 32 --max_iterations 200 \
    --headless --experiment_name ${LOG}_t2

echo "=== Test 3: PPO/ABC Balance ==="
python -m asyncDualPlayPPO.train --num_envs 32 --max_iterations 50 \
    --headless --experiment_name ${LOG}_t3
python -m asyncDualPlayPPO.diagnostics.test_ppo_abc_balance --log_dir runs/${LOG}_t3
python -m asyncDualPlayPPO.diagnostics.test_checkpoint_chain --log_dir runs/${LOG}_t3

echo "=== Test 4: Goal Encoder ==="
# Requires a longer run — reuse t3 checkpoint if >200 iters, else run more
python -m asyncDualPlayPPO.diagnostics.test_abc_goal_encoder --ckpt runs/${LOG}_t3/model_50.pt
python -m asyncDualPlayPPO.diagnostics.test_goal_encoder_latent --ckpt runs/${LOG}_t3/model_50.pt

echo "=== All diagnostics passed ==="
```

---

## Dependencies

```
pip install tbparse sklearn matplotlib
```

`tbparse` reads tensorboard event files without running a TB server. If unavailable, use `tensorboard.backend.event_processing.event_accumulator` from the tensorboard package directly.
