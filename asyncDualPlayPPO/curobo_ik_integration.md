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
reachable workspace. cuRobo will return `result.success = False` for these.
**Mitigation**: reset the episode immediately on IK failure rather than holding pose.
The existing dense reward (EE distance to object) is sufficient to teach Alice to
stay in the reachable workspace — no additional penalty term needed.

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

## 6. Human vs Model Evaluation Script

`tests/test_human_vs_bob.py` — a dedicated side-by-side evaluation tool.

**Two arenas share one goal.** `num_envs=2` gives two identical side-by-side arenas in
Isaac Sim. The same goal configuration is injected into both. The human drives arena 0
via gamepad + cuRobo IK; the loaded Bob model drives arena 1.

**Goal generation.** If `--chkpt_alice` is supplied, Alice's policy runs on arena 0
for `--alice_steps` steps (default 100) and the final object poses become the shared
goal. Without an Alice checkpoint, a random walk is used instead.

**Usage:**
```bash
python tests/test_human_vs_bob.py \
    --chkpt_bob  runs/my_run/bob/model_500.pt \
    --chkpt_alice runs/my_run/alice/model_500.pt \
    --num_objects 2 --max_vel 1.0
# Press R in-sim to regenerate a new goal and restart the race.
```

**Metric**: wall-clock seconds and step count until all objects are within 0.05 m /
0.20 rad of the goal — the same success criterion used during training. The terminal
prints a winner announcement when either side solves it.

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
