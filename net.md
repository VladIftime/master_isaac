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
dims 0-2: XYZ Cartesian delta   → (bin − 5) / 5 × max_delta  (default 0.05 m)
dims 3-4: Rx, Ry rotation delta → (bin − 5) / 5 × 0.5 rad
dim  5:   Gripper               → sign(normalized) ∈ {−1, 0, +1}
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
— an advantage of Euler over quaternion for this structured subtraction.


storage.py — three fixes:

traj_maxlen is now a constructor parameter (default 200, YAML sets 500). With 512 envs and ~66% fail rate, the old 200 cycled faster than one iteration — no cross-iteration memory. 500 gives ~1.5 iterations of trajectory history, enough diversity without staleness.
save serialises _traj_store to CPU before pickling — _traj_store was silently lost on every checkpoint resume, meaning BC always restarted cold.
load restores _traj_store onto the correct device, respecting the current _traj_maxlen (so changing maxlen between runs still works).
train.py — ABC two-phase controller:

Phase 1 (iter < abc_anneal_iters): unchanged linear decay 0.5 → 0.0, bootstraps Bob when RL has no useful signal.
Phase 2: target = abc_coef_start * (1 - bob_sr) — 0.5 when Bob fails completely, 0.0 when Bob always succeeds. EMA with α=0.95 smooths transitions so a single bad SR window doesn't shock the optimizer. When Alice's entropy spikes and Bob's SR drops, abc_coef rises automatically within ~20 iterations.
ppo_continuous.yaml: abc_coef_ema: 0.95, abc_traj_maxlen: 500 added alongside the existing ABC params.

ining pipeline/home3/s3426394/master_isaac/README.md
/home3/s3426394/master_isaac/net.md

/home3/s3426394/master_isaac/asyncDualPlayPPO/train.py@contextScopeItemMention @contextScopeItemMention @contextScopeItemMention @contextScopeItemMention @contextScopeItemMention 

Find a top 10 of innfficiencies, in terms of memory and time. The priority is htat THEY DO NOT AFFECT THE FUNCTIONALITY OF THE PIPELINE,


I think the following analysis is wrong prove it if its not show how come:Top 10 Inefficiencies in the ASP Training Pipeline
NOTE

All proposed fixes are semantics-preserving — they do not change what the model learns, the reward structure, or the training dynamics. They only reduce wasted GPU time and memory.

1. 🔴 Per-Rollout-Step Tensor Allocation Storm (HIGH — ~30% of rollout time)
Files: 
train.py:940–944
, 
train.py:983–987
, 
train.py:1025–1029
, 
train.py:1076–1079
, 
train.py:1114–1117

Problem: Inside the rollout loop (for t in range(rollout_length)), ~20 tensors are allocated fresh via torch.zeros() every single step. With rollout_length = 256 + 200 = 456 steps, that's ~9,000 CUDA malloc calls per iteration. CUDA malloc is not free — each call hits the allocator, fragments the memory pool, and triggers occasional GC pauses.

Fix: Pre-allocate all these buffers once before the loop and .zero_() them at the start of each step (or just index into pre-allocated storage). The tensors are:

a_acts_active, a_logprob_active, a_val_active, a_mu_active, a_sigma_active
b_acts_active, b_logprob_active, b_val_active, b_mu_active, b_sigma_active
a_policy, b_policy
a_lp_full, a_val_full, a_mu_full, a_sigma_full
b_lp_full, b_val_full, b_mu_full, b_sigma_full
Safety: ✅ Pure allocation change. Same values written before use.

2. 🔴 Historical Policy Deep-Copy Every Iteration (HIGH — memory + time)
File: 
historical_pool.py:52

Problem: sample_policy() calls copy.deepcopy(reference_policy) every iteration, duplicating the entire ActorCritic (PI encoder 512-dim, LSTM 256, trunk, etc.) on GPU. This is ~20MB of GPU memory + significant CPU overhead for the deep copy + device transfers, done twice per iteration (once for Alice pool, once for Bob pool).

Fix: Keep a single persistent _hist_clone on the GPU and just load_state_dict() into it when sampling. No need to deep-copy the architecture each time — only the weights change.

python
def sample_policy(self, reference_policy, device):
    if not self._pool:
        return None
    if self._hist_clone is None:
        self._hist_clone = copy.deepcopy(reference_policy)  # once only
    snapshot = random.choice(self._pool)
    self._hist_clone.load_state_dict({k: v.to(device) for k, v in snapshot.items()})
    self._hist_clone.eval()
    return self._hist_clone
Safety: ✅ Same weights loaded, same inference. No gradient flows through historical policy.

3. 🟠 RolloutStorage.states Wastes 50% of Buffer Memory (MEDIUM — GPU memory)
File: 
storage.py:38-40

Problem: self.states is allocated as (num_transitions, num_envs, *states_shape) but asymmetric=False throughout the entire pipeline. This means states_shape == obs_shape, and states is never read (the critic always uses observations). For Alice (35D obs, 456 steps, 256 envs): that's 456 × 256 × 35 × 4 bytes = 16MB of dead GPU memory. Bob has even more (51D). Total: ~40MB wasted per agent.

Fix: When asymmetric=False, allocate states as a tiny scalar tensor (or skip it entirely and gate all .copy_() calls).

Safety: ✅ states is only read when self.asymmetric is True, which is never the case.

4. 🟠 ABC Buffer Stores 7 Unused Fields (MEDIUM — GPU memory)
File: 
storage.py:191-201

Problem: GPUDemonstrationBuffer allocates rewards, dones, values, returns, advantages, mu, sigma at capacity=50,000 — but none of these are ever used. Only observations, actions, and actions_log_prob matter for the ABC loss. The _traj_store deque is used for sequential LSTM evaluation, and sample() is never called.

With obs_shape=(51,), actions_shape=(6,):

Unused fields: rewards(50k×1) + dones(50k×1) + values(50k×1) + returns(50k×1) + advantages(50k×1) + mu(50k×6) + sigma(50k×6) = 50,000 × 17 × 4 bytes = 3.4MB wasted.
Plus observations and states (50k × 51 × 2 = 20.4MB) are also allocated but never read (only _traj_store is used). Total waste: ~24MB.

Fix: Remove unused fields from GPUDemonstrationBuffer, or replace the entire buffer with just the _traj_store deque (since sample() is never called in the pipeline).

Safety: ✅ No code path reads the unused fields.

5. 🟠 Per-Env Python Loop in ABC Buffer Population (MEDIUM — time)
File: 
train.py:1186-1209

Problem: The ABC demo insertion loop iterates for env_id in valid_ids: in Python, calling bob_ppo.actor_critic.evaluate() once per failed env. With 256 envs and ~66% fail rate, that's ~170 individual forward passes per iteration, each with tiny batch size 1.

Fix: Batch all failed envs together:

Pad all trajectories to the same length
Run a single evaluate() call with batch dimension = num_failed_envs
Add all trajectories to _traj_store in one pass
Safety: ✅ Same data goes into the buffer. Batching doesn't change the values.

6. 🟠 torch.searchsorted for Index Mapping (MEDIUM — time, per step)
File: 
train.py:963
, 
train.py:971
, 
train.py:1006
, 
train.py:1014

Problem: torch.searchsorted(alice_indices, curr_ids) is called 4× per rollout step to map global env IDs back to local indices. searchsorted launches a CUDA kernel each time, and the indices are small tensors (~200 elements), so the kernel launch overhead dominates.

Fix: Pre-compute a full reverse-index lookup table once before the loop:

python
alice_local_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
alice_local_idx[alice_indices] = torch.arange(len(alice_indices), device=device)
# Then: curr_local = alice_local_idx[curr_ids]  # O(1) gather, no kernel launch
Safety: ✅ Produces identical index mapping.

7. 🟡 Redundant .clone() Calls in Extras Dict (LOW — time)
File: 
wrapper.py:463-480

Problem: Every env.step() creates clones of 8 tensors for the extras["episode_manager"] dict:

python
"phase": self.episode_manager.current_phase.clone(),
"goal_count": self.episode_manager.goal_count.clone(),
"bob_success": self.episode_manager.bob_success.clone(),
...
Most of these are only read once downstream and never modified. Cloning forces a GPU memcpy each time — 8 × num_envs × 4 bytes per step, for 456 steps.

Fix: Only clone tensors that are actually mutated downstream. phase, goal_count, and max_contact_force are never modified after extraction — pass them directly. Only bob_success_this_step and bob_done_this_step (which are freshly allocated) need to stay.

Safety: ✅ Read-only access patterns confirmed by grep.

8. 🟡 Critic Forward Pass Runs on Full Raw Obs (Including Goal+Distance) (LOW — time)
File: 
module.py:308-309

Problem: Bob's critic MLP processes the full 51D observation including goal poses and distances. But the goal encoder was specifically designed to compress goals into an 8D embedding for the actor. The critic doesn't benefit from this bottleneck — but it also doesn't need the raw 22D-per-object goal+distance chunk. The first critic layer is Linear(51→512), which could be Linear(35→512) if the critic saw the same encoded representation.

This is more of a latent inefficiency — the critic has 51×512 = 26,112 parameters in layer 1 vs 35×512 = 17,920 if using encoded obs. The extra 8,192 parameters add ~0.5% to backward pass time.

Fix: Feed the critic the PI-encoded + goal-encoded representation instead of raw obs. This would require structural changes to the forward pass, so it's lower priority.

Safety: ⚠️ Changing critic input could slightly alter training dynamics. Best tested in a short comparison run.

9. 🟡 MultiCategorical Creates N Separate Categorical Objects (LOW — time)
File: 
module.py:33

Problem: MultiCategorical.__init__ creates a Python list of 6 Categorical distributions via a list comprehension. log_prob() and entropy() then iterate this list with sum(...), launching 6 small CUDA kernels each. PyTorch's Categorical could operate on a single (batch, 6, 11) tensor natively using a single kernel.

Fix: Replace the list of distributions with a single batched operation:

python
def log_prob(self, bin_indices):
    # logits: (batch, 6, 11), bin_indices: (batch, 6)
    return F.cross_entropy(
        self.logits.reshape(-1, 11), bin_indices.reshape(-1).long(), reduction='none'
    ).neg().reshape(-1, 6).sum(-1)
Safety: ✅ Mathematically identical. Just fused kernel execution.

10. 🟡 _alice_dense_accum Tensor Never Used for Anything (LOW — memory)
File: 
wrapper.py:170
, 
wrapper.py:929

Problem: _alice_dense_accum is accumulated via += dense[is_alice] every step, then zeroed at phase end. But it is never read — no code uses the accumulated value for logging, reward adjustment, or anything else.

Fix: Remove the tensor and the accumulation line entirely.

Safety: ✅ Dead code removal.
Show less
I'll read all the relevant files to verify each claim in the analysis.

Bash Find relevant Python files
IN
find /home3/s3426394/master_isaac/asyncDualPlayPPO -name "*.py" | grep -E "(train|historical_pool|storage|wrapper|module)" | head -20

OUT
/home3/s3426394/master_isaac/asyncDualPlayPPO/train.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/train_diffik.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/logs/analyze_training.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/utils/historical_pool.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/tasks/utils/dummy_alice_wrapper.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/tasks/utils/wrapper.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/algorithms/rl/ppo/module.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/algorithms/rl/ppo/storage.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/matplotlib/_constrained_layout.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/matplotlib/tests/test_constrainedlayout.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/pip/_vendor/cachecontrol/filewrapper.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/pip/_vendor/cachecontrol/wrapper.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/pip/_vendor/rich/constrain.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/numpy/_core/tests/test_simd_module.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/numpy/f2py/tests/test_modules.py
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/lib/python3.11/site-packages/numpy/typing/tests/data/pass/modules.py
Read train.py (lines 931-1130)
Read historical_pool.py
Read storage.py
Read wrapper.py (lines 451-510)
Read module.py (lines 2-61)
Read module.py (lines 281-340)
Read wrapper.py (lines 161-190)
Bash Find all usages of _alice_dense_accum
IN
grep -n "_alice_dense_accum" /home3/s3426394/master_isaac/asyncDualPlayPPO/tasks/utils/wrapper.py

OUT
170:        self._alice_dense_accum = torch.zeros(env.num_envs, device=self.device)
326:                self._alice_dense_accum[alice_term_ids] = 0.0
548:        self._alice_dense_accum[env_ids] = 0.0
929:            self._alice_dense_accum[is_alice] += dense[is_alice]
Read train.py (lines 1181-1240)
Bash Check asymmetric usage in train.py
IN
grep -n "asymmetric" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -20

OUT
456:        asymmetric=False,
477:        asymmetric=False,
508:        asymmetric=False,
523:        asymmetric=False,
Bash Check asymmetric usage in module.py
IN
grep -n "asymmetric\|states_shape\|states=" /home3/s3426394/master_isaac/asyncDualPlayPPO/algorithms/rl/ppo/module.py | head -20

OUT
150:        states_shape,
154:        asymmetric=False,
158:        self.asymmetric = asymmetric
308:        critic_in = states_shape[0] if asymmetric else obs_shape[0]
539:        value = self.critic(states if self.asymmetric else observations)
565:        value = self.critic(states if self.asymmetric else observations)
607:            states: states (for asymmetric critic) or None
616:        value = self.critic(states if self.asymmetric else observations)
Bash Check how add_transitions is called - what is passed as 'states'
IN
grep -n "add_transitions" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -10

OUT
1097:                alice_ppo.storage.add_transitions(
1137:                bob_ppo.storage.add_transitions(
Read train.py (lines 1097-1146)
Bash Check RolloutStorage initialization in train.py
IN
grep -n "RolloutStorage\|states_shape" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -20

OUT
561:        states_shape=env.bob_observation_space.shape,
Bash Check RolloutStorage initialization in ppo.py
IN
grep -n "RolloutStorage\|states_shape" /home3/s3426394/master_isaac/asyncDualPlayPPO/algorithms/rl/ppo/ppo.py 2>/dev/null | head -20

OUT
16:from .storage import RolloutStorage
84:        self.storage = RolloutStorage(
Read ppo.py (lines 85-104)
Read train.py (lines 456-470)
Read ppo.py (lines 101-220)
Read ppo.py (lines 261-360)
Bash Check how historical policies are sampled
IN
grep -n "sample_policy\|hist_alice\|hist_bob" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -30

OUT
925:        hist_alice = alice_pool.sample_policy(alice_ppo.actor_critic, env.device) if alice_pool.size > 0 else None
926:        hist_bob = bob_pool.sample_policy(bob_ppo.actor_critic, env.device) if bob_pool.size > 0 else None
957:                    if len(hist_ids) > 0 and hist_alice is not None:
958:                        (a_acts_hist, a_logprob_hist, a_val_hist, a_mu_hist, a_sigma_hist, _) = hist_alice.act_with_hidden(current_alice_obs[hist_ids], None, None)
1000:                    if len(hist_bids) > 0 and hist_bob is not None:
1001:                        (b_acts_hist, b_lp_hist, b_val_hist, b_mu_hist, b_sig_hist, _) = hist_bob.act_with_hidden(current_bob_obs[hist_bids], None, None)
Bash Check searchsorted usage
IN
grep -n "searchsorted" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py

OUT
963:                curr_local = torch.searchsorted(alice_indices, curr_ids)
971:                    hist_local = torch.searchsorted(alice_indices, hist_ids)
1006:                curr_bloc = torch.searchsorted(bob_indices, curr_bids)
1014:                    hist_bloc = torch.searchsorted(bob_indices, hist_bids)
Read train.py (lines 506-535)
Read wrapper.py (lines 921-940)
Bash Check if sample() or sample_trajectories() is called on abc_buffer
IN
grep -n "sample_trajectories\|abc_buffer\.sample\|demo_buffer\.sample" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -20

OUT
(Bash completed with no output)
Bash Check all abc_buffer usage in train.py
IN
grep -n "abc_buffer\|demo_buffer" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -30

OUT
197:        "iteration showing which section (env_step, alice_act, abc_buffer, …) "
558:    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
611:        _abc_buf_path = os.path.join(os.path.dirname(args.chkpt_bob), "abc_buffer.pt")
613:            bob_ppo.abc_buffer.load(_abc_buf_path)
615:                f"[Resume] Loaded ABC buffer ({bob_ppo.abc_buffer.size} entries) from {_abc_buf_path}"
809:            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
1181:                profiler.mark_start("abc_buffer")
1200:                    bob_ppo.abc_buffer.add_trajectory(
1210:                profiler.mark_stop("abc_buffer")
1296:            bob_ppo.abc_buffer.save(os.path.join(bob_ppo.log_dir, "abc_buffer.pt"))
1324:        _abc_buf_size = bob_ppo.abc_buffer.size
1335:            f"ABC buf: {bob_ppo.abc_buffer.size} | "
Bash Check if sample_trajectories is called
IN
grep -n "sample_trajectories\|abc_buffer\.sample\b" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py

OUT
(Bash completed with no output)
Bash Search for sample_trajectories usage across algorithms dir
IN
grep -rn "sample_trajectories\|abc_buffer\.sample\b" /home3/s3426394/master_isaac/asyncDualPlayPPO/algorithms/ /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py 2>/dev/null

OUT
/home3/s3426394/master_isaac/asyncDualPlayPPO/algorithms/rl/ppo/ppo_abc.py:124:            abc_trajs = self.abc_buffer.sample_trajectories(self.abc_n_trajs)
/home3/s3426394/master_isaac/asyncDualPlayPPO/algorithms/rl/ppo/storage.py:312:    def sample_trajectories(self, n_trajs: int):
Read ppo_abc.py (lines 111-160)
Bash Check if ppo_abc update function is called in train.py
IN
grep -n "ppo_abc\|PPOabc\|ABC\|abc" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -20

OUT
165:        "--test_abc_verbose",
167:        help="Test: print ABC demo content (goal shape, obs range) each time "
197:        "iteration showing which section (env_step, alice_act, abc_buffer, …) "
222:    from asyncDualPlayPPO.algorithms.rl.ppo.ppo_abc import PPOABC
265:            "abc_coef": 0.5,  # "ABC loss weight β"
276:            "abc_coef": learn.get("abc_coef", "?"),
502:    bob_ppo = PPOABC(
555:    # ABC Buffer for Alice's successful demonstrations
557:    _abc_act_shape = (num_cat_dims,) if use_mc else env.action_space.shape
558:    bob_ppo.abc_buffer = GPUDemonstrationBuffer(
562:        actions_shape=_abc_act_shape,
564:        traj_maxlen=ppo_cfg["params"]["learn"].get("abc_traj_maxlen", 500),
611:        _abc_buf_path = os.path.join(os.path.dirname(args.chkpt_bob), "abc_buffer.pt")
612:        if os.path.isfile(_abc_buf_path):
613:            bob_ppo.abc_buffer.load(_abc_buf_path)
615:                f"[Resume] Loaded ABC buffer ({bob_ppo.abc_buffer.size} entries) from {_abc_buf_path}"
632:            bob_ppo.abc_coef = float(_ts["abc_coef"])
636:                f"abc_coef={bob_ppo.abc_coef:.4f}, "
642:                f"entropy_coef/abc_coef reset to YAML defaults."
668:    last_alice_mean_rew = 0.0  # gating value for Bob's ABC loss warmup
Read wrapper.py (lines 321-335)
Read wrapper.py (lines 544-553)
Let me check the actual ABC buffer capacity and module forward pass:

Read train.py (lines 556-570)
Read module.py (lines 531-560)
Bash Check rollout length configuration
IN
grep -n "rollout_length\|alice_timesteps\|bob_timesteps\|nsteps\|num_transitions" /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -20

OUT
91:        "--nsteps",
113:        "Auto-computed as ceil(bob_timesteps / alice_timesteps) * max(1, 64 // num_envs) "
246:    alice_timesteps = task_cfg.get("alice_timesteps", 150)
247:    bob_timesteps = task_cfg.get("bob_timesteps", 200)
250:        f"[Config] Episode structure: alice_timesteps={alice_timesteps}, "
251:        f"bob_timesteps={bob_timesteps}, max_goals={max_goals_per_episode} "
296:    if args.nsteps is not None:
298:            f"[Config] Overriding nsteps: {ppo_cfg['params']['learn']['nsteps']} -> {args.nsteps}"
300:        ppo_cfg["params"]["learn"]["nsteps"] = args.nsteps
379:            alice_timesteps=alice_timesteps,
380:            bob_timesteps=bob_timesteps,
393:            alice_timesteps=alice_timesteps,
394:            bob_timesteps=bob_timesteps,
407:            alice_timesteps=alice_timesteps,
408:            bob_timesteps=bob_timesteps,
420:            alice_timesteps=alice_timesteps,
421:            bob_timesteps=bob_timesteps,
428:            alice_timesteps=alice_timesteps,
429:            bob_timesteps=bob_timesteps,
437:            alice_timesteps=alice_timesteps,
Bash Check config for rollout length
IN
grep -n "alice_timesteps\|bob_timesteps\|nsteps\|num_transitions" /home3/s3426394/master_isaac/asyncDualPlayPPO/cfg/train/FrankaTwoAgentsPPO.yaml 2>/dev/null | head -20

OUT
(Bash completed with no output)
Read train.py (lines 906-935)
Bash Find config files with timestep settings
IN
find /home3/s3426394/master_isaac/asyncDualPlayPPO/cfg -name "*.yaml" | xargs grep -l "alice_timesteps\|bob_timesteps" 2>/dev/null

OUT
/home3/s3426394/master_isaac/asyncDualPlayPPO/cfg/ppo/ppo_continuous.yaml
/home3/s3426394/master_isaac/asyncDualPlayPPO/cfg/task/AsyncDualPlay.yaml
Bash Check actual timestep values in config
IN
grep "alice_timesteps\|bob_timesteps" /home3/s3426394/master_isaac/asyncDualPlayPPO/cfg/task/AsyncDualPlay.yaml

OUT
alice_timesteps: 100
bob_timesteps: 200
Read train_small.slurm
Now I have enough to give a precise verdict. Here's what I found:

Verdict: 3 claims are wrong or materially misleading
Claim #1 — Wrong on numbers AND magnitude
The stated rollout length is wrong. The analysis says 256 + 200 = 456 steps. The actual values from the config are alice_timesteps=100, bob_timesteps=200, so rollout_length = 300 (train.py:928, AsyncDualPlay.yaml). The 256 comes from NSTEPS=256 in the slurm script — that's the storage buffer size, not the rollout length. The analysis confused the two.

"~30% of rollout time" is almost certainly wrong. PyTorch uses a caching memory allocator — torch.zeros() does not call cudaMalloc each time. It reuses freed blocks from a pool. The allocations are real overhead (especially for dynamically-sized tensors like (len(alice_indices),...) where the count varies step to step, preventing clean pool reuse), but calling it "30% of rollout time" with env.step() (full physics simulation with 256 envs) as the dominant cost is a massive overstatement. The inefficiency pattern is real; the magnitude claim is not.

Claim #3 — Wrong percentage
The analysis claims states wastes "50% of Buffer Memory". What's true: states is never read during updates when asymmetric=False (ppo_abc.py:135-141), and state_space = observation_space (train.py:459, train.py:511) means it duplicates observations. But "50%" implies states is half the total buffer. In reality, RolloutStorage has ~10 tensor fields (observations, states, rewards, actions, dones, masks, actions_log_prob, values, returns, advantages, mu, sigma). The wasted states is ~9% of total buffer memory, or 50% of the obs+states sub-portion only. The 16MB figure for Alice is correct; the "50%" framing is wrong.

Claim #8 — Should not be in this list at all
The analysis header explicitly states "All proposed fixes are semantics-preserving — they do not change what the model learns." Claim #8 then proposes changing the critic's input from raw 51D obs to the encoded representation and itself flags it with ⚠️: "Changing critic input could slightly alter training dynamics." This directly contradicts the stated premise of the entire analysis. Claim #8 is a model architecture change, not an efficiency fix, and has no place in a "functionality-preserving" optimization list.

Claims that are correct
#	Status	Note
#2	✅	copy.deepcopy once per iteration confirmed at train.py:925-926, fix is valid
#4	✅	sample() is never called in the pipeline; only sample_trajectories() via _traj_store is used (ppo_abc.py:124); flat 50k-capacity buffer is confirmed dead weight
#5	✅	Python loop at train.py:1186 confirmed, ~170 sequential evaluate() calls
#6	✅	4 searchsorted calls per step on GPU tensors confirmed at train.py:963,971,1006,1014
#7	✅	.clone() confirmed at wrapper.py:463-483, but impact is tiny
#9	✅	6-Categorical list at module.py:33 confirmed
#10	✅	_alice_dense_accum accumulated at wrapper.py:929, reset at lines 326 and 548, never read anywhere
implement the correct fiexes
Let me read the PPO __init__ and the parts of wrapper.py I need for the fixes:

Read ppo.py (lines 2-84)
Read ppo_abc.py (lines 2-111)
Let me read the remaining parts I need before making changes:

Read module.py (lines 61-110)
Read wrapper.py (lines 2-31)
Bash Check usage of top-level extras goal_valid and bob_success
IN
grep -n 'extras\["goal_valid"\]\|extras\["bob_success"\]' /home3/s3426394/master_isaac/asyncDualPlayPPO/train.py | head -20

OUT
(Bash completed with no output)
Read train.py (lines 1156-1185)
Read wrapper.py (lines 391-470)
Bash Find all max_contact_force modifications
IN
grep -n "max_contact_force" /home3/s3426394/master_isaac/asyncDualPlayPPO/tasks/utils/wrapper.py | head -20

OUT
457:                    self.episode_manager.max_contact_force = torch.where(
459:                        torch.max(self.episode_manager.max_contact_force, max_f),
460:                        self.episode_manager.max_contact_force,
479:            "max_contact_force": self.episode_manager.max_contact_force.clone(),
Bash Check in-place modifications of episode_manager fields after extras dict creation
IN
grep -n "bob_success\b\|current_phase\b\|goal_count\b\|goal_valid\b\|goal_states\b" /home3/s3426394/master_isaac/asyncDualPlayPPO/tasks/utils/wrapper.py | grep -v "#" | head -30

OUT
373:        step_bob_success = torch.zeros(
386:            step_bob_success[bob_done_ids] = success
413:            self.episode_manager.bob_success[completion_ids] = True
414:            step_bob_success[completion_ids] = True
418:            success, pos_err, rot_err = self._check_bob_success(
425:            completion_goals = self.episode_manager.goal_count[completion_ids].clone()
436:                self.episode_manager.goal_count[completion_ids]
464:            "phase": self.episode_manager.current_phase.clone(),
465:            "goal_count": self.episode_manager.goal_count.clone(),
466:            "bob_success": self.episode_manager.bob_success.clone(),
467:            "bob_success_this_step": step_bob_success,
471:            "goal_valid": self.episode_manager.goal_valid.clone(),
472:            "goal_states": (
473:                self.episode_manager.goal_states.clone()
474:                if self.episode_manager.goal_states is not None
482:        extras["goal_valid"] = self.episode_manager.goal_valid.clone()
483:        extras["bob_success"] = self.episode_manager.bob_success.clone()
568:        self.episode_manager.mark_goal_valid(env_ids, valid)
625:        success, pos_err, rot_err = self._check_bob_success(obs_dict, env_ids)
626:        self.episode_manager.mark_bob_success(env_ids, success)
647:            self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
674:            succeeded = self.episode_manager.bob_success[reset_ids]
694:            self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
724:    def _check_bob_success(self, obs_dict: Dict, env_ids: torch.Tensor) -> torch.Tensor:
806:        self, alice_obs: torch.Tensor, goal_states: torch.Tensor
811:        Assumes alice_obs and goal_states are already in the LOCAL environment frame.
812:        goal_states must be in Euler format: [pos(3)+euler(3)] per object = 6D each.
817:        if goal_states.dim() == 1:
818:            goal_states = goal_states.unsqueeze(0)
830:        goal_states_reshaped = goal_states.view(-1, self.num_objects, 6)
Now I have everything needed. Let me implement all 8 fixes across the 6 files in parallel:

Read wrapper.py (lines 371-385)
Read train.py (lines 1141-1155)
Now let me implement all fixes. Starting with the isolated files:

Edit historical_pool.py
Added 1 line
Edit historical_pool.py
Added 1 line
Now fix storage.py — Fix #3 (RolloutStorage asymmetric flag) and Fix #4 (strip GPUDemonstrationBuffer flat buffers):

Edit storage.py
Added 3 lines
Edit storage.py
Added 1 line
Now replace the entire GPUDemonstrationBuffer class with the stripped-down version:

Edit storage.py
Removed 141 lines
Now fix ppo.py to pass asymmetric to RolloutStorage:

Edit ppo.py
Added 1 line
Now fix module.py — Fix #9 (batched MultiCategorical):

Edit module.py
Added 1 line
Edit module.py
Added 6 lines
Now fix wrapper.py — Fix #7 (remove redundant clones) and Fix #10 (remove dead _alice_dense_accum):

Edit wrapper.py
Removed 3 lines
Edit wrapper.py
Removed 1 line
Edit wrapper.py
Removed 1 line
Edit wrapper.py
Removed 1 line
Now remove the redundant .clone() calls from the extras dict and the duplicate top-level entries:

Edit wrapper.py
Removed 3 lines
Now fix train.py — Fix #5 (batch ABC loop) and Fix #6 (searchsorted → lookup table):

Edit train.py
Added 8 lines
Now replace the 4 searchsorted calls with lookup table indexing:

Edit train.py
Modified
Edit train.py
Modified
Now replace the sequential ABC buffer loop with the batched version (Fix #5), also updating the add_trajectory call signature (Fix #4):

Edit train.py
Modified
Now let me verify all changes look correct:

Bash Verify storage, GPUDemonstrationBuffer, and MultiCategorical changes
IN
python -c "
import sys
sys.path.insert(0, 'asyncDualPlayPPO')

# Test storage changes
from asyncDualPlayPPO.algorithms.rl.ppo.storage import RolloutStorage, GPUDemonstrationBuffer
import torch

# Fix #3: states=None when asymmetric=False
s = RolloutStorage(4, 10, (5,), (5,), (3,), device='cpu', asymmetric=False)
assert s.states is None, 'states should be None when asymmetric=False'

# add_transitions should not crash with states=None
obs = torch.zeros(4, 5)
s.add_transitions(obs, obs, torch.zeros(4, 3), torch.zeros(4), torch.zeros(4).bool(), torch.zeros(4,1), torch.zeros(4), torch.zeros(4,3), torch.zeros(4,3))
print('Fix #3: RolloutStorage asymmetric=False OK')

# Fix #4: GPUDemonstrationBuffer simplified
buf = GPUDemonstrationBuffer(50000, (51,), (51,), (6,), device='cpu')
assert buf.size == 0
buf.add_trajectory(torch.zeros(10, 51), torch.zeros(10, 6), torch.zeros(10))
assert buf.size == 1
trajs = buf.sample_trajectories(1)
assert len(trajs) == 1 and 'obs' in trajs[0] and 'acts' in trajs[0] and 'old_lp' in trajs[0]
print('Fix #4: GPUDemonstrationBuffer simplified OK')

# Fix #9: MultiCategorical batched
from asyncDualPlayPPO.algorithms.rl.ppo.module import MultiCategorical
logits = torch.randn(32, 6, 11)
dist = MultiCategorical(logits)
samples = dist.sample()
assert samples.shape == (32, 6), f'sample shape wrong: {samples.shape}'
lp = dist.log_prob(samples)
assert lp.shape == (32,), f'log_prob shape wrong: {lp.shape}'
ent = dist.entropy()
assert ent.shape == (32,), f'entropy shape wrong: {ent.shape}'
print('Fix #9: MultiCategorical batched OK')
"

OUT
Exit code 127
/bin/bash: line 38: python: command not found
Bash Verify all Python-testable fixes
IN
cd /home3/s3426394/master_isaac/asyncDualPlayPPO && .venv/bin/python -c "
import sys, torch

# Fix #3: states=None when asymmetric=False
from algorithms.rl.ppo.storage import RolloutStorage, GPUDemonstrationBuffer
s = RolloutStorage(4, 10, (5,), (5,), (3,), device='cpu', asymmetric=False)
assert s.states is None
s.add_transitions(torch.zeros(4,5), torch.zeros(4,5), torch.zeros(4,3), torch.zeros(4), torch.zeros(4).bool(), torch.zeros(4,1), torch.zeros(4), torch.zeros(4,3), torch.zeros(4,3))
print('Fix #3: OK')

# Fix #4: GPUDemonstrationBuffer simplified
buf = GPUDemonstrationBuffer(50000, (51,), (51,), (6,), device='cpu')
assert buf.size == 0
buf.add_trajectory(torch.zeros(10,51), torch.zeros(10,6), torch.zeros(10))
assert buf.size == 1
trajs = buf.sample_trajectories(1)
assert len(trajs) == 1 and set(trajs[0].keys()) == {'obs','acts','old_lp'}
print('Fix #4: OK')

# Fix #9: MultiCategorical batched
from algorithms.rl.ppo.module import MultiCategorical
logits = torch.randn(32, 6, 11)
dist = MultiCategorical(logits)
samples = dist.sample()
assert samples.shape == (32, 6)
lp = dist.log_prob(samples)
assert lp.shape == (32,)
ent = dist.entropy()
assert ent.shape == (32,)
print('Fix #9: OK')

# Verify log_prob is numerically correct vs old implementation
from torch.distributions import Categorical
logits2 = torch.randn(8, 6, 11)
bins = torch.randint(0, 11, (8, 6))
old_lp = sum(Categorical(logits=logits2[:,i,:]).log_prob(bins[:,i]) for i in range(6))
new_lp = MultiCategorical(logits2).log_prob(bins)
assert torch.allclose(old_lp, new_lp, atol=1e-5), f'max diff={( old_lp - new_lp).abs().max()}'
print('Fix #9 numerical equivalence: OK')
"

OUT
Exit code 1
Traceback (most recent call last):
  File "<string>", line 2, in <module>
ModuleNotFoundError: No module named 'torch'
Bash Find Python executable
IN
find /home3/s3426394/master_isaac/asyncDualPlayPPO/.venv -name "python3" | head -3

OUT
/home3/s3426394/master_isaac/asyncDualPlayPPO/.venv/bin/python3
Bash Find system Python
IN
which python3 && python3 --version 2>/dev/null || which python3.11 2>/dev/null

OUT
/usr/bin/python3
Python 3.6.8
Bash Verify all fixes with correct Python path
IN
cd /home3/s3426394/master_isaac/asyncDualPlayPPO && .venv/bin/python3 -c "
import sys, torch
sys.path.insert(0, '.')

from algorithms.rl.ppo.storage import RolloutStorage, GPUDemonstrationBuffer

# Fix #3
s = RolloutStorage(4, 10, (5,), (5,), (3,), device='cpu', asymmetric=False)
assert s.states is None
s.add_transitions(torch.zeros(4,5), torch.zeros(4,5), torch.zeros(4,3), torch.zeros(4), torch.zeros(4).bool(), torch.zeros(4,1), torch.zeros(4), torch.zeros(4,3), torch.zeros(4,3))
print('Fix #3: OK')

# Fix #4
buf = GPUDemonstrationBuffer(50000, (51,), (51,), (6,), device='cpu')
assert buf.size == 0
buf.add_trajectory(torch.zeros(10,51), torch.zeros(10,6), torch.zeros(10))
assert buf.size == 1
trajs = buf.sample_trajectories(1)
assert len(trajs) == 1 and set(trajs[0].keys()) == {'obs','acts','old_lp'}
print('Fix #4: OK')

# Fix #9
from algorithms.rl.ppo.module import MultiCategorical
from torch.distributions import Categorical
logits = torch.randn(8, 6, 11)
bins = torch.randint(0, 11, (8, 6))
old_lp = sum(Categorical(logits=logits[:,i,:]).log_prob(bins[:,i]) for i in range(6))
new_lp = MultiCategorical(logits).log_prob(bins)
assert torch.allclose(old_lp, new_lp, atol=1e-5), f'diff={( old_lp - new_lp).abs().max()}'
samples = MultiCategorical(logits).sample()
assert samples.shape == (8, 6)
ent = MultiCategorical(logits).entropy()
assert ent.shape == (8,)
print('Fix #9: OK, numerical equivalence verified')

# Fix #2 - verify historical pool uses persistent clone
import copy
from utils.historical_pool import HistoricalPolicyPool
import torch.nn as nn
net = nn.Linear(4, 4)
pool = HistoricalPolicyPool(max_size=3)
pool.add(net)
result1 = pool.sample_policy(net, 'cpu')
result2 = pool.sample_policy(net, 'cpu')
assert result1 is result2, 'should reuse same clone object'
assert pool._hist_clone is not None
print('Fix #2: OK, persistent clone verified')
"

