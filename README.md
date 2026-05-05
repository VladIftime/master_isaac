# Asymmetric Self-Play with Goal Encoders for Dual-Arm Robotic Manipulation

This project combines two papers to build a hierarchical reinforcement learning system for dual-arm manipulation in Isaac Lab / Isaac Sim.

---

## Papers

### 1. Asymmetric Self-Play for Automatic Goal Discovery (OpenAI, Plappert et al. 2021)
`asymetric-self-play.pdf`

The core training framework. Two agents — **Alice** and **Bob** — play an adversarial game in a shared environment:

- **Alice** manipulates objects freely for T=100 steps to construct a goal state (a non-trivial object configuration).
- **Bob** then observes the resulting goal and must reproduce it from a fresh reset within T=200 steps.
- Goals repeat across 5 sub-goals per episode; each goal reuses the previous sub-goal's end state as the new start.

**Rewards** are sparse and symmetric:
- Bob receives **+1** for each object within the success threshold (0.04 m position, ~2° rotation).
- Bob receives **-1** if an object already at goal moves away.
- Bob receives a **+5 completion bonus** when all objects are simultaneously at goal.

**Alice Behavioral Cloning (ABC)**: When Bob fails a goal, Alice's trajectory for that sub-goal is stored in a BC replay buffer. Bob's PPO loss is augmented with a clipped imitation loss (ε=0.2, β=0.5) that clones Alice's actions, bootstrapping Bob's exploration.

**Historical policy pool**: 20% of episodes pit each agent against a randomly sampled past version of the opponent, improving training stability. The pool holds the 5 most recent snapshots.

---

### 2. Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning (Sukhbaatar et al. 2018)
`asymetric-self-play_charlie.pdf`

Introduces a **goal encoder** that compresses the goal description into a low-dimensional embedding, replacing naive goal-state concatenation in Bob's policy.

**Goal encoder** E: (s\*, s_t^B) → g_t ∈ ℝ^K maps the goal state s\* and Bob's current state s_t^B to a compact embedding g_t. Bob's policy becomes goal-conditioned: π'_B(s_t^B, g_t).

Two encoder architectures:
- **Difference form**: φ(s\*) − φ(s_t^B) — encodes relative progress toward goal.
- **Absolute form**: φ(s\*) — encodes goal independently of current state.

The encoder is integrated into Bob's second hidden layer:  
`h_2 = σ(W_2 σ(W_1 s_t) + W_g g_t)`

A high-level **Charlie controller** can also generate goal embeddings g_t to direct a pre-trained Bob, enabling hierarchical control without retraining the low-level policy.

---

## Project Contribution: Combining Both Papers for Dual-Arm Manipulation

This codebase merges the two frameworks and extends them to a **dual-arm robotic platform** (two UR5e robots with Robotiq grippers) simulated in **Isaac Lab / Isaac Sim**.

### Why the combination?

OpenAI's ASP provides an automatic curriculum (Alice discovers goals of increasing difficulty) and efficient exploration via ABC. However, raw goal-state concatenation in Bob's observation grows linearly with the number of objects and makes the policy sensitive to irrelevant goal dimensions.

Charlie's goal encoder solves this: the encoder compresses the goal into a fixed-size bottleneck g_t ∈ ℝ^K regardless of how many objects are in the scene. The embedding focuses Bob's attention on what still needs to change, rather than where everything currently is.

The dual-arm extension adds a second manipulator, doubling the number of objects in the workspace and making compact goal representations even more important.

### Architectural overview

```
Alice (PPO):
    Obs: EE pose(6) + gripper(1) + [obj_state(14)] × 2  = 35D
    Acts: joint velocity commands → RMPFlow controller
    Role: explore and construct interesting goal configurations

Bob (PPO + ABC + Goal Encoder):
    Obs (interleaved per-object): Robot(7) + [obj_state(14) + goal(6) + dist(2)] × 2 = 51D
    Goal encoder:      E(goal_state, current_state) → g ∈ ℝ^K
    Acts: same action space as Alice
    Role: reproduce the goal Alice left behind

Episode Manager:
    Stores Alice's final state as the goal for Bob (12D LOCAL Euler per episode)
    Validates that Alice moved at least one object (pos_threshold or rot_threshold)
    Manages phase transitions and ABC buffer writes
```

**State format** (12D per episode, LOCAL frame, Euler):  
`[target_pos(3), target_euler(3), cube_pos(3), cube_euler(3)]`

**Bob obs per object** (14D, from `object_states()`):  
`[pos(3), euler(3), lin_vel(3), ang_vel(3), dist_to_gripper(1), contact(1)]`

**Goal encoder** (`use_goal_encoder = True`, hardcoded in `train.py`): always active for Bob.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| ZYX Euler angles | Matches OpenAI paper Appendix A.2; avoids quaternion discontinuities in policy input |
| 12D goal state (pos+euler × 2 objects) | Compact; no velocities needed for goal definition |
| 51D Bob obs (interleaved) | Object state + goal + distance interleaved per-object; easier for encoder to associate |
| Alice entropy 1.0 → 0.10 over min(max_iters, 250) iters | Floor raised to 0.10 so Alice retains late-stage exploration; denominator clamped at 250 so annealing always completes |
| ABC filter: Bob-failure only | Follows paper §3.3; avoids cloning trivial successes |
| Historical pool 20% | Paper ratio; pool holds last 5 snapshots (max_size=5 in HistoricalPolicyPool) |
| Success threshold 0.05 m (pos), ~2° (rot) | 0.04 m from paper; relaxed to 0.05 m in wrapper |

---

## File Structure

```
asyncDualPlayPPO/
├── train.py                            # Main training loop (Alice+Bob PPO, ABC, historical pool)
├── run_diagnostic_tests.sh             # Three-test diagnostic suite (headless, logs to runs/diag_*)
├── optuna_sweep.py                     # Hyperparameter sweep with Optuna
├── buffers.py                          # Low-level buffer utilities
├── test_checkpoint_chain.py            # Checkpoint save/load smoke test
│
├── cfg/
│   ├── ppo/ppo_continuous.yaml         # PPO + ABC hyperparameters
│   └── task/AsyncDualPlay.yaml         # Episode structure (timesteps, goals per episode)
│
├── algorithms/
│   ├── goal_encoder.py                 # GoalEncoder φ MLP + aux distance-prediction head
│   └── rl/ppo/
│       ├── module.py                   # ActorCritic, PermInvEncoder, MultiCategorical
│       ├── ppo.py                      # Base PPO (Alice)
│       ├── ppo_abc.py                  # PPOABC: PPO + Alice Behavioral Cloning (Bob)
│       └── storage.py                  # RolloutStorage + GPUDemonstrationBuffer
│
├── tasks/
│   ├── async_dual_play.py              # IsaacLab env config (scene, observations, rewards)
│   └── utils/
│       ├── wrapper.py                  # AsyncDualPlayEnvWrapper: phase management, rewards
│       ├── observations.py             # Observation functions (EE, objects, goals, distances)
│       ├── rewards.py                  # Alice reward constants + reward functions
│       ├── events.py                   # Reset events (objects, robot joints)
│       ├── terminations.py             # Episode termination conditions
│       ├── dummy_alice_wrapper.py      # Diagnostic wrappers (DummyBob, DummyGoalDistance, …)
│       └── base/events.py              # Base reset event helpers
│
├── utils/
│   ├── episode_manager.py              # EpisodeManager: phase tracking, goal storage
│   ├── goal_validator.py               # validate_goal: movement threshold check
│   └── historical_pool.py             # HistoricalPolicyPool: past-5-snapshot ring buffer
│
├── tests/
│   ├── test_abc.py                     # End-to-end ABC pipeline tests
│   └── test_abc_goal_encoder.py        # Goal encoder integration tests
│
├── hpc/
│   ├── train_high.slurm                # Production HPC job (A100, 512 envs)
│   ├── train_medium.slurm              # Medium-scale HPC job
│   ├── train_low.slurm                 # Small-scale HPC job
│   └── run_interactive.sh              # Interactive session helper
│
├── extras/                             # Offline analysis / visualisation scripts
│   ├── visualize_logs.py
│   ├── plot_results.py
│   ├── diagnose_logs.py
│   └── extract_updates.py
│
└── paper-async/
    ├── asymetric-self-play.pdf         # OpenAI ASP paper (Plappert et al. 2021)
    └── asymetric-self-play_charlie.pdf # Charlie/HSP paper (Sukhbaatar et al. 2018)
```

---

## Running

### Local (headless)
```bash
python train.py --num_envs 16 --max_iterations 500 --exp_name test_run --headless
```

### HPC (Apptainer / Isaac Lab container)
```bash
sbatch hpc/train_high.slurm
```

### Diagnostic tests (3-test suite)
```bash
# Locally (from master_isaac/):
bash asyncDualPlayPPO/run_diagnostic_tests.sh

# Individual tests:
# Test 1 — reward pipeline (DummyBobWrapper teleports target→goal, expect SR > 0)
python -m asyncDualPlayPPO.train --headless --num_envs 16 --max_iterations 50 --test_bob_reward

# Test 2 — Alice exploration sandbox (watch ValidGoals climb)
python -m asyncDualPlayPPO.train --headless --num_envs 32 --max_iterations 200

# Test 3 — PPO vs ABC balance (watch Loss/Bob/ABC vs Loss/Bob/Surrogate)
python -m asyncDualPlayPPO.train --headless --num_envs 64 --max_iterations 300
```

---

## References

```bibtex
@article{plappert2021asymmetric,
  title={Asymmetric self-play for automatic goal discovery in robotic manipulation},
  author={Plappert, Matthias and Rajeswaran, Aravind and others},
  journal={arXiv preprint arXiv:2101.04882},
  year={2021}
}

@article{sukhbaatar2018learning,
  title={Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning},
  author={Sukhbaatar, Sainbayar and Lin, Zeming and Kostrikov, Ilya and Synnaeve, Gabriel and Szlam, Arthur and Fergus, Rob},
  journal={arXiv preprint arXiv:1811.09083},
  year={2018}
}
```
# cuRobo IK Integration — Feasibility Analysis

> **Context**: The project trains Alice+Bob PPO with Asymmetric Self-Play (Plappert et al. 2021)
> augmented with a Charlie-style GoalEncoder (Sukhbaatar et al. 2018) on two UR5e arms in Isaac Lab.
> Current controllers: **RMPFlow** (`train.py`) and **DifferentialIK** (`train_diffik.py`).
> Goal: a third variant `train_curobo.py` replacing the low-level controller with cuRobo IK.

---

## 1. How the Three Controllers Compare

| | RMPFlow (`train.py`) | DiffIK (`train_diffik.py`) | cuRobo IK (`train_curobo.py`) |
|---|---|---|---|
| **Policy output** | EE Cartesian delta (MultiCategorical, 6×11 bins) | EE Cartesian delta → DiffIK → Δθ | EE Cartesian delta → cuRobo → θ* |
| **Joint control** | Handled internally by RMPFlow | JointPositionActionCfg | JointPositionActionCfg |
| **IK quality** | RMPFlow (reactive, may drift) | First-order Jacobian pseudo-inverse (poor near singularities) | Batch MPPI/CuTorch solver, singularity-aware, seed-conditioned |
| **Policy obs** | EE pose (7D) — no joint angles | EE pose (7D) — no joint angles | EE pose (7D) — no joint angles |
| **Action space change** | — | None vs. train.py | None vs. train.py |
| **Speed per step** | Fast (native Isaac Lab) | Fast + Jacobian solve | Adds GPU IK solve (~1–3 ms/batch) |

---

## 2. Feasibility

**Yes, it is feasible.** The pattern is already proven in
`tests/test_curobo_follow_target.py`:

```
policy → EE Cartesian target
    → cuRobo solve_single / solve_batch (seed from current joints)
    → joint position command → Isaac Lab JointPositionActionCfg
```

The only structural difference from `train_diffik.py` is replacing IsaacLab's
DifferentialIK action term with an explicit cuRobo solve call inside the training loop,
identical to how the test script calls `ik_solver.solve_single()` at each sim step.

For training with N envs in parallel the correct call is `ik_solver.solve_batch()`:
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

`train_curobo.py` would be structurally identical to `train_diffik.py` with:
1. `from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig` (before AppLauncher)
2. `ik_solver` initialised once before the training loop (same as the test script)
3. The rollout step replaces the DiffIK action term with a manual cuRobo batch solve
4. Env action cfg uses `JointPositionActionCfg` (same as DiffIK)

---

## 3. Pros of cuRobo IK

### 3.1 Solution Quality
- **Singularity handling**: cuRobo's solver is aware of the manipulability metric and
  avoids degenerate configurations. DiffIK (Jacobian pseudo-inverse) degrades near
  singularities and can produce large, noisy joint velocity commands.
- **Seed conditioning**: seeding from current joints gives smooth, continuous joint
  trajectories — the robot does not jump to a distant configuration the way unconditioned
  IK can.
- **Guaranteed reachability check**: `result.success` flags infeasible targets per-env.
  The training loop can punish infeasible EE commands or fall back gracefully, which
  is cleaner than RMPFlow silently saturating velocities.

### 3.2 Training Dynamics
- **Consistent EE-to-joint mapping**: The policy outputs Cartesian deltas; cuRobo
  converts them deterministically. RMPFlow adds its own reactive planning noise which
  can confuse the policy about the consequence of its actions.
- **Better ASP goal fidelity**: Alice constructs goals by moving objects with the EE.
  If Alice's IK is higher quality, the goal states she creates are more physically
  consistent, giving Bob cleaner targets.
- **Workspace enforcement**: cuRobo respects joint limits natively, so the policy
  cannot accidentally drive the arm out of bounds in a way that the physics ignores.

### 3.3 HPC Scalability
- cuRobo is GPU-native and batched: `solve_batch(N envs)` runs as a single CUDA
  kernel launch, not N sequential solves. At 512 envs (A100 configuration in
  `train_high.slurm`) the overhead is ~2–5 ms per rollout step, which is negligible
  compared to the Isaac Lab physics step (~15–40 ms at 512 envs).
- The cuRobo CUDA graph can be warmed up once before the training loop, making
  subsequent calls ~0.5–1 ms.

### 3.4 Test-Time Evaluation
- **The test script already works**: `test_curobo_follow_target.py` runs interactive
  IK with the real environment. Loading a trained checkpoint and replacing the policy's
  random action with `actor_critic.act()` is a ~20-line change — the IK pipeline is
  identical.
- **Goal generation with the test script**: Trained Alice's Cartesian EE commands can
  drive the cuRobo solver in the test loop. Record final object poses → these become
  Bob's goals. The test script already has block spawning (`C` key) and resets
  (`R` key), so the evaluation loop practically writes itself.

---

## 4. Cons and Drawbacks

### 4.1 cuRobo Must Be Imported Before AppLauncher
This is a hard constraint (already handled in the test script). It requires a different
module import order from `train.py` / `train_diffik.py`. The `SuppressAllOutput`
pattern from `train_diffik.py` must wrap cuRobo's initialisation to prevent URDF
parser noise from contaminating slurm logs.

### 4.2 IK Failure Rate During Early Training
Early in training Alice takes random Cartesian actions, many of which land outside the
reachable workspace. cuRobo will return `result.success = False` for these — meaning
the robot holds pose instead of executing the action. This stalls Alice's exploration
in the first ~50 iterations. **Mitigation**: use `last_good_joints` fallback (already
in the test script) + add a small penalty to Alice's reward when IK fails, to steer
the policy away from unreachable targets.

### 4.3 Orientation Is Fixed in the Test Script
`test_curobo_follow_target.py` uses a fixed "tool pointing down" quaternion
(`[0,1,0,0]`). The current policy's action space includes Rx, Ry rotation bins (dims
3–4 in the MultiCategorical head). For `train_curobo.py` you must decide:
- **Option A**: also pass the orientation delta to cuRobo (full 6D EE target). This
  is more expressive but orientation IK is harder to satisfy.
- **Option B**: fix orientation to "down" and reduce the policy action space to 4D
  (XYZ + gripper). This simplifies the IK and matches the test script exactly.

Option B is recommended for a first implementation — it eliminates one failure mode
and the manipulation task (placing objects on a table) rarely requires non-downward
orientations.

### 4.4 CUDA Graph Warm-Up Adds ~30s to Startup
cuRobo traces CUDA graphs on first use. This is a one-time cost but makes the initial
seconds of a slurm job look stalled. Add a warm-up call before the training loop:
```python
_warmup_pose = Pose(position=torch.zeros(num_envs, 3, device=device),
                    quaternion=fixed_quat.expand(num_envs, 4))
ik_solver.solve_batch(_warmup_pose,
                      seed_config=torch.zeros(num_envs, 1, 6, device=device),
                      retract_config=torch.zeros(num_envs, 6, device=device))
```

### 4.5 cuRobo Requires Its Own Robot Config
The test script loads `ur5e.yml` from `get_robot_configs_path()`. The YAML must match
the URDF used by Isaac Lab exactly (joint limits, link names). If your dual-arm setup
uses a modified UR5e URDF, cuRobo's collision-aware solve may diverge from the
simulated robot. **Verification step**: run `test_curobo_follow_target.py` and confirm
the EE tracking error stays below ~5 mm before training.

### 4.6 Memory Overhead
cuRobo pre-allocates GPU tensors for its internal motion-generation buffers. At
N=512 envs the overhead is ~400–800 MB of VRAM. Verify this fits alongside Isaac Lab's
physics buffers on the A100 (80 GB) — it almost certainly does, but it is worth
checking on the RTX Pro 6000 (48 GB) used in `train_profile.slurm`.

### 4.7 No Drop-In for HPC Without Container Rebuild
`train_profile.slurm` and `train_high.slurm` use `isaac-lab.sif`. cuRobo must be
installed inside that Apptainer image. If it is not already present, the container
needs a rebuild (or a bind-mount of a cuRobo wheel). Check with:
```bash
apptainer exec --nv isaac-lab.sif python -c "import curobo; print(curobo.__version__)"
```

---

## 5. Relation to the ASP + ABC Training Loop

The Alice/Bob phase structure, ABC buffer, historical pool, GoalEncoder and
PPOABC loss are **controller-agnostic** — they operate entirely on observations and
rewards, which are computed from object poses, not from joint angles. Replacing the
low-level controller does not touch:

- `algorithms/rl/ppo/ppo_abc.py` (ABC loss)
- `algorithms/goal_encoder.py` (GoalEncoder)
- `utils/episode_manager.py` (phase management)
- `tasks/utils/wrapper.py` (reward shaping, goal validation)
- `utils/historical_pool.py` (policy snapshots)

The only files that need modification are:
1. `train_curobo.py` — new entry point (copy of `train_diffik.py` + cuRobo IK solve)
2. `tasks/async_dual_play.py` — swap action config to `JointPositionActionCfg`
   (already done for DiffIK; reuse that config)
3. `hpc/train_curobo.slurm` — new slurm script (copy of `train_profile.slurm`,
   increase `--time` to production length)

---

## 6. Evaluating a Trained Model With the Test Script

The evaluation pattern is:

```python
# After loading checkpoint:
alice_ppo.actor_critic.eval()
bob_ppo.actor_critic.eval()

# In the test loop (replace gamepad delta with policy):
with torch.no_grad():
    acts, _, _, _, _, _ = bob_ppo.actor_critic.act_with_hidden(obs, hidden_state, masks)
ee_delta = decode_multicategorical(acts)   # same decode as training rollout
ee_target = current_ee_pos + ee_delta[:3]
# → cuRobo solve → joint_cmd → env.step()
```

`test_curobo_follow_target.py` already has the cuRobo solve loop, workspace clamping,
and `last_good_joints` fallback. The additions needed:
1. Add `--chkpt` argument to load Bob's checkpoint
2. Replace the gamepad delta with Bob's policy output
3. Optionally: add Alice phase to auto-generate goals, then switch to Bob evaluation

**Metric of interest**: time-to-solve (steps until all objects within 0.05 m / 2°),
which maps directly to the `bob_sr` metric already logged during training.

---

## 7. HPC Profiling (`train_profile.slurm`)

The existing profile slurm runs 3 iterations at 2048 envs with `--profile`. For the
cuRobo variant:
- Add cuRobo warm-up step to the profiler timeline so it is not counted as training time
- The `profiler.mark_start("abc_buffer")` / `mark_stop` pattern already in `train.py`
  can be extended with `mark_start("curobo_ik")` / `mark_stop("curobo_ik")` to
  measure the IK solve fraction of each rollout step
- Expected: cuRobo IK accounts for <10% of rollout step time at 512 envs on an A100

---

## 8. Recommendation

**Start with `train_curobo.py` as a thin wrapper over `train_diffik.py`**:
1. Use `JointPositionActionCfg` (identical to DiffIK)
2. Fix orientation to "tool pointing down" (Option B above) — reduces IK failures
3. Insert the cuRobo batch solve between policy output and `env.step()`
4. Add `ik_fail_rate` to the per-iteration log (fraction of envs where IK returned
   `success=False`) — this is a key diagnostic for whether the policy is learning
   reachable targets
5. Run `test_curobo_follow_target.py` first to confirm tracking error < 5 mm, then
   move to training

The expected benefit over RMPFlow is smoother joint trajectories (better physical
plausibility of Alice's goals) and explicit workspace constraint enforcement. The
expected benefit over DiffIK is better IK quality near singularities, which matters
when Alice explores configurations at the edges of the workspace.

---

## Latest Updates (May 2026)

### 1. Environment & Object Spawning
*   **Startup Randomization**: The `TargetObject` and `Cube` are now randomly selected from a pool (`concave`, `cube`, `cylinder`, `rect`, `triangle`) at simulation startup. This provides visual and geometric variety for each training run while maintaining simulation stability.
*   **Visual Enhancements**:
    *   **Scale**: All blocks have been scaled up to **(1.5, 1.5, 1.5)** for significantly better visibility and easier interaction for the robot arms.
    *   **Color**: All target-related objects are now consistently colored **Green** `(0.1, 0.8, 0.1)`.
    *   **Table Feedback**: In non-headless mode, the table color now dynamically switches based on the active agent: **Red** for Alice and **Blue** for Bob.

### 2. cuRobo Training Refinements (`train_curobo.py`)
*   **Continuous Episodes (No IK Resets)**: Removed the immediate episode reset on IK failure. If the cuRobo solver cannot find a solution (e.g., the agent commands a pose inside the table), the robot now **holds its last valid pose** and the episode continues. This allows Alice to learn how to touch and push blocks without being interrupted by a reset.
*   **Target Persistence**: Removed target synchronization on IK failure. The target no longer "snaps" back to the hand if a command fails, forcing the agent to learn to move away from unreachable or colliding configurations.
*   **Stability**: Switched to **Startup Randomization** for USD references to prevent invalidating the PhysX tensor backend views during training rollouts. Resolved import errors (`NameError`, `AttributeError`) associated with custom spawner implementation.
