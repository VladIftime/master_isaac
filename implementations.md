# Implementation Record — ASP + GoalEncoder + Push-PPO Baseline + SAC/HER

**Branch**: `asp_goal_encoder`  
**Last updated**: 2026-06-29 (validation campaign complete — all 7 models evaluated on identical 30 T-block scenes; results consolidated to `results/`; P82 curriculum fix validated; A_simp 80.0% vs B_curr 76.7% definitive comparison)

---

## Table of Contents

1. [Overview & Comparison Targets](#overview)
2. [Development Timeline](#timeline)
3. [ASP + GoalEncoder Implementation](#asp-goalencoder)
4. [cuRobo IK Integration](#curobo-ik)
5. [HPC Setup & Run Guide](#hpc-setup)
6. [Push-PPO Baseline](#push-ppo)
7. [Push Primitive Test](#push-primitive-test)
8. [Validation Evaluation Suite](#validation-suite)
9. [SAC + HER Push Baseline (DirectRLEnv)](#sac-her)
10. [gym-pusht Controlled Testbed (Models A/B/C)](#gym-testbed)
11. [Validation Results (26.06.26–28)](#validation-results)
12. [Results Directory Structure](#results-directory)

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
| 4 | **Push-ASP** — ASP with object-relative push primitives | `train_push_asp.py` |
| 5 | **PBRS Model A** — PBRS dense reward, no curriculum | `train_push_pbrs_no_curr.py` |
| 6 | **PBRS Model B** — PBRS dense + forced pos→rot curriculum | `train_push_pbrs_curr.py` |
| 7 | **PBRS Model C** — PBRS dense + ASP curriculum (Alice/Bob) | `train_push_pbrs_asp.py` |
| 8 | **PBRS Model D** — Model C with GoalEncoder ablated (PI-encoder only for Bob) | `train_push_pbrs_asp_no_ge.py` |
| 9 | **PBRS Model E** — Model C with SE(2) d_pose metric (T-block) | `train_push_pbrs_asp_dpose.py` |
| 10 | **PBRS Model F** — Model E with disc object (position-only d_pose) | `train_push_pbrs_asp_disc.py` |
| 11 | **PBRS Model G** — Model E with Sukhbaatar time-based Alice reward (T-block) | `train_push_pbrs_tasp_dpose.py` |
| 12 | **PBRS Model H** — Model F with Sukhbaatar time-based Alice reward (Disc) | `train_push_pbrs_tasp_disc.py` |
| 13 | **PBRS Model I** *(planned)* — Model G + Bob time penalty R_B += -gamma_sp*t_B | — |
| 14 | **PBRS Model J** *(planned)* — Model H + Bob time penalty R_B += -gamma_sp*t_B | — |
| 15 | **PBRS Model K** *(planned)* — Model G + ABC enabled (beta=0.5) | — |
| 16 | **PBRS Model L** *(planned)* — Model H + ABC enabled (beta=0.5) | — |
| 17 | **SAC+HER Push Baseline** — SAC (Haarnoja et al.) + HER (Andrychowicz et al.) push primitive, DirectRLEnv + SB3 | `train_push_sac_her.py` |
| 18 | **gym-pusht Model A** — PBRS single-agent in gym-pusht (controlled CPU testbed) | `train_a_gym_pbrs_simple.py` |
| 19 | **gym-pusht Model B** — PBRS + P82 pos→rot curriculum in gym-pusht | `train_b_gym_pbrs_curriculum.py` |
| 20 | **gym-pusht Model C** — PBRS + ASP (Alice/Bob) in gym-pusht | `train_c_gym_pbrs_asp.py` |

### Stack Versions

| Component | Version |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 (commit `6c151ea`) |
| **cuRobo** | **0.7.5** — last release with `IKSolver`/`solve_batch`/`Pose` API |
| PyTorch | 2.7.0+cu128 |
| Python | 3.11.5 |
| **Stable-Baselines3** | **2.7.0** — SAC + HerReplayBuffer for goal-conditioned push |
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
| 2026-05-28 | **Fix P45 (zero-length push)**: `decode_push_action` clamped push length to `[0.01, 0.20]`, making a "hold position" action impossible. After pushing the object to the goal on push 3, pushes 4–5 would push it away again. The +5 completion bonus became permanent (gated by `_gave_completion`) and the agent learned to collect it mid-episode without caring where the object ended up. Lowered min clamp `0.01 → 0.0` so the policy can output `length=0` to signal "I'm done." When `Xf=Xs, Yf=Ys`, the push phase holds at contact height — the object stays put. `action_push.py:145-153`. |
| 2026-05-28 | **Fix P46 (completion terminates episode)**: `check_done` previously let the episode run to `max_pushes` even after pos_err<0.05m, so rotation refinement could continue. In practice, subsequent pushes overshot the goal. Now `at_goal_pos` terminates immediately — the agent gets a clean return signal and resets to practice on a fresh goal. This also fixes the `_gave_completion` gating hack: the bonus is never "permanent" because reaching the goal ends the episode. `wrapper_push.py:261-274`. |
| 2026-05-28 | **Fix P47 (rotation reward rebalanced)**: `PUSH_DENSE_ROT_ALPHA` 5.0→1.0. At α_pos=12 (0.01m→0.12 reward) vs α_rot=5 (0.5rad→2.5 reward), rotation gradients dominated position by ~20×. The observed result (RotSR=42%, PosErr flat at 0.25m) exactly matched this prediction. At α_rot=1.0, a 0.5rad improvement earns 0.50 — comparable to a 0.04m position improvement earning 0.48. Both gradients now live in the same magnitude range. `wrapper_push.py:26`. |
| 2026-05-28 | **Fix P48 (object-relative push actions for Push-ASP)**: Absolute `(Xs, Ys)` had ~2% contact probability → Alice never bootstrapped. New `decode_push_action_relative()` parameterizes approach as `(r, φ)` relative to object center/yaw (guaranteed contact), with push direction `θ` in world frame (decoupled, easy translation). Bob obs gains relative goal `[delta_x, delta_y, rel_yaw, pos_dist, rot_dist]` (5D world-frame) replacing absolute goal+dist (8D). Alice obs 21→20D (gripper removed), Bob obs 29→25D. New file: `tasks/utils/action_push_relative.py`. |
| 2026-05-28 | **Fix P49 (Push-ASP IK-death fix)**: IK failures during waypoint execution used `prev_joint_cmd` as fallback instead of the arm's actual physics state `cur_joints`. When `prev_joint_cmd` differed from the true physics position (e.g., after gravity drift or partial IK solve), the fallback commanded the arm to jump to a stale position, potentially through the table. Now IK failures hold `cur_joints` (actual physics state), and `prev_joint_cmd` is updated from `cur_joints` on failure so subsequent waypoints also freeze correctly. Arm stays put on IK fail until the push trajectory completes — no teleport, no table penetration, no object-launching. `train_push_asp.py:947-960`. |
| 2026-05-29 | **Fix P50 (object-relative observation switch for Push-PPO baseline)**: The 28D flat observation requires the policy to learn `atan2(goal_y-obj_y, goal_x-obj_x)` internally. New `--rel-obs` flag appends `[rel_dx, rel_dy] = goal_pos[:2] - obj_pos[:2]` to the observation (28D→30D), making the push direction trivially available. Backward-compatible (off by default). New HPC script: `hpc/train_push_rel.slurm`. |
| 2026-06-04 | **Fix P51 (slurm auto-resume iteration counter bug)**: Both `train_push.slurm` and `train_push_rel.slurm` passed `--chkpt` (loading weights) but never `--resume_iteration`, so every job in the chain restarted the iteration counter from 0. 3-job chain with 1,851 iters each = 5,550 paid-for iterations but only 1,851 worth of progress. Both scripts now extract the iteration number from the checkpoint filename and pass `--resume_iteration $ITER_NUM`. |
| 2026-06-04 | **Fix P52 (episode start-state logging)**: Per-env bin logging was spammy and uninformative. Now: (a) per-env lines only fire for `length > 0` (actual pushes, not zero-length holds); (b) per-episode log includes `start=(x,y,z)` position and `yaw` orientation so the full trajectory (start → goal → final) is visible in one line; (c) iteration summary ends with `| rel` or `| abs` depending on observation mode. `wrapper_push.py`, `train_push.py`. |
| 2026-06-04 | **Fix P53 (Bob dense per-push improvement reward for Push-ASP)**: Bob's sparse-only `{+1/−1/+5}` reward fired on 0.14% of pushes, producing zero-mean GAE noise. Added `compute_bob_dense_push_reward()` in `wrapper_push_asp.py` mirroring `wrapper_push.py`'s dense improvement formula: `R = α·(d_prev−d_now) + α_rot·(y_prev−y_now) − β·d_now − β_rot·y_now` with α=12, α_rot=1, β=0.5, β_rot=0.25. All components clamped per Fix P30. Dense reward has NO completion bonus — that comes from sparse to avoid double-counting the +5. Bob's per-push reward = dense + sparse. Completion detection (`bob_achieved_completion`) still gated on sparse only to prevent false triggers from large dense improvements. Alice stays sparse-only (preserves adversarial curriculum). `wrapper_push_asp.py`, `train_push_asp.py`. |
| 2026-05-28 | **Fix P54 (Push-ASP observation dimension fix)**: Gripper was removed from push-task observations in Fix P39 (`OBS_ROBOT_DIM 7→6`) but `wrapper_push_asp.py` still defined `_OBS_ROBOT_DIM=7`. Mismatch caused `RuntimeError: size of tensor a (29) must match size of tensor b (28)`. Fixed `_OBS_ROBOT_DIM 7→6`, `robot_state_dim 7→6` (two places in `train_push_asp.py`), and GoalEncoder sampling offset. Alice obs 21→20D, Bob obs 29→28D. |
| 2026-05-28 | **Fix P55 (Push-ASP missing PPO update calls)**: `perform_alice_update()` and `perform_bob_update()` were defined at lines 549/568 but never called in the main loop. Training printed `[Iter 0]` forever — the counter never incremented. Added `perform_alice_update(); perform_bob_update(current_bob_obs)` after the rollout loop. |
| 2026-05-28 | **Fix P56 (Push-ASP tight spawn bounds + random yaw)**: Objects spawned with default wide bounds `X∈[-0.35,0.35] × Y∈[0.45,0.80]` causing 46% invalid goal rate from OOB pushes. New `_rand_reset_objs()` helper with `X∈[-0.04,0.04] × Y∈[0.35,0.45]` (8cm×10cm box) + `random_yaw=True` (uniform [-π,π]). `_initial_states_from_spawn` updated to track yaw via `target_yaw` return from spawn. All 7 reset call sites routed through `_rand_reset_objs`. `reset_objects_to_random_safe_pose` gained `random_yaw` parameter (default False, backward-compatible). `events.py`, `wrapper_push_asp.py`. |
| 2026-05-28 | **Fix P57 (Push-ASP workspace clamping, EE sync, Bob safety penalty, slurm resubmit)**: (a) Push start/end `(Xs,Ys,Xf,Yf)` clamped to workspace ±0.02m margin with recomputed length/theta — credit matches execution; (b) `ee_pos_local`/`ee_quat_w`/`prev_joint_cmd` synced to physics after each push matching `train_push.py:687-688`; (c) Bob gets `-5.0` penalty for `obj_lifted || robot_through_table` pushes — same magnitude as +5 completion bonus for symmetric gradient; (d) slurm resubmit path fixed from `$PROJECT_ROOT/asyncDualPlayPPO/hpc/...` (doubled) to `$PROJECT_ROOT/hpc/...`. `train_push_asp.py`, `train_push_asp.slurm`. |
| 2026-06-06 | **Fix P58 (randomised object spawn for Push-PPO)**: Object spawned at fixed `(0, 0.5, 0.05)` — policy memorised `Xs≈0, Ys≈0.5` without reading the observation. Random spawn `X∈[-0.4,0.4] Y∈[0.3,0.7] yaw∈[0,2π]` forces the policy to extract position from observation to compute approach point. Once those gradients flow, θ features (goal positions, rel_dx/dy) can also receive attention — breaking the fixed-θ local optimum. Object teleported via `write_root_pose_to_sim` after `env.env.reset()`; fresh observation recomputed by `_get_push_obs()`. |
| 2026-06-06 | **Fix P59 (trivial-goal filter)**: Independent spawn and goal randomisation (same ranges) produced ~2.5% episodes where the object was already within 0.05 m of the goal at start — episode terminated on first `check_done` with +5 bonus and zero pushes. New `_sample_goals_filtered()` reads object scene position after spawn and rejects goals within 0.05 m. Goal sampling moved from `reset_done_envs` (before object position known) to `train_push.py` reset block (after object randomisation). Diagnostic print `[P4 filter] N goals too close` fires on resamples. |
| 2026-06-06 | **Fix P60 (_get_push_obs compute check)**: `observation_manager.compute()` called outside `env.step()` cycle — diagnostic print `[P5 check] _get_push_obs` fires on every call to verify frequency (should only fire during reset, not polling in step loop) and confirm no side-effects. Push-PPO has one obs group with no sensors. Remove print after verification run. |
| 2026-06-06 | **Fix P61 (_ep_started cleared after reset completes)**: `_ep_started` cleared in `reset_done_envs()` BEFORE the base reset — if anything threw between clear and reset, next `capture_pre_push` would snapshot a stale pose as "episode start." `_ep_started[ids]=False` moved to `train_push.py` reset block, only firing after the full reset+randomise+sample sequence succeeds. |
| 2026-06-06 | **Fix P62 (object-relative action decode for Push-PPO)**: New `--rel-act` flag swaps `decode_push_action` for `decode_push_action_relative` (from Fix P48, `action_push_relative.py`). Instead of absolute Xs ∈ [-0.50,0.50], Ys ∈ [0.25,0.70], the same 4D × 21 bins parameterize `(r, φ, len, θ)` where `r ∈ [0.02,0.08]m` is radial offset from object center, `φ ∈ [-π,π]` is approach angle in object frame, `len ∈ [0,0.20]m`, and `θ ∈ [-π,π]` is push direction in world frame. `Xs = obj_x + r·cos(obj_yaw+φ)`, `Ys = obj_y + r·sin(obj_yaw+φ)`. Guarantees ~100% contact regardless of object spawn position. World-frame θ aligned with `--rel-obs` delta features for trivial direction learning. New HPC script: `hpc/train_push_rel_full.slurm` with both flags. Per-push log shows `r=` (effective approach offset) instead of `Xs=/Ys=` in rel_act mode. Iteration summary shows `rel_full` / `rel_act` / `rel_obs` / `abs`. |
| 2026-06-06 | **Fix P63 (normalised fractional reward)**: `PUSH_DENSE_ALPHA=12` (position) and `PUSH_DENSE_ROT_ALPHA=1` (rotation) replaced by single `PUSH_DENSE_ALPHA=3.0` with normalised deltas: `pos_imp = α·(d_prev−d_now)/d_prev`, `rot_imp = α·(y_prev−y_now)/y_prev`. Both are unitless fractions of remaining error — a push that halves the distance earns α×0.5 regardless of whether the domain is position or rotation. Denominators clamped at 0.01 to prevent division by zero. One coefficient instead of two, no hand-tuning required. |
| 2026-06-08 | **Fix P64 (Push-ASP dense reward normalised)** — `compute_bob_dense_push_reward()` in `wrapper_push_asp.py` was still using the old non-normalised formula (`R = 12·Δd + 1·Δyaw − 0.5·d_now − 0.25·y_now`) from Fix P53, while `wrapper_push.py` had already been upgraded to the normalised fractional formula in Fix P63. A 1cm improvement from 30cm away earned `12×0.01 − 0.5×0.29 = −0.025` — **negative** even for a progress-producing push. Random pushes averaging ~0cm improvement produced expected reward ≈ −0.15 with near-zero variance, starving Bob's PPO gradient. Now uses α=3.0 with division by d_prev/y_prev (clamped at 0.01). A push halving the distance earns +1.5. | `wrapper_push_asp.py:485-516` |
| 2026-06-08 | **Fix P65 (Push-ASP obj_lifted criterion removed)** — Per-waypoint-substep Z-check (`obj_lifted |= _z_obs[:, _OBS_ROBOT_DIM + 2] > 0.10`) was firing on 45–55% of pushes, terminating episodes with −5 penalty. Push-PPO (`train_push.py`) does NOT have this check at all — it experiences the same physics but simply ignores momentary object tipping. The check was a false positive: objects briefly tip during pushes then settle back to the table. Removing it brings Push-ASP in line with Push-PPO's execution path, restoring 100% of push budget for Bob. Also removed `robot_through_table` check and the `_abnormal` handling block (which paid Alice +5 for Bob's "lifted" pushes, leaking reward). | `train_push_asp.py:933-1018` |
| 2026-06-08 | **Fix P66 (Push-ASP --rel-obs for Bob)** — Bob's 28D observation `[ee_pose(6) | obj(14) | goal(6) | dist(2)]` required the LSTM+MLP to learn `atan2(goal_y−obj_y, goal_x−obj_x)` internally to predict the correct push θ. Same issue as Push-PPO Fix P50, solved differently here to avoid breaking `module.py`'s structured observation slicing (`_encode_obs` chunks by `_ge_raw_per_obj=22D`): when `--rel-obs` is on, `_get_push_obs()` replaces the `goal_dist(2)` slot (pos_dist, rot_dist) with `[rel_dx, rel_dy] = goal_xy − obj_xy`. Same 28D dimension — no model architecture changes. World-frame push direction θ is now trivially inferable from the observation. Backward-compatible (off by default). | `wrapper_push_asp.py:240-256`, `train_push_asp.py:120-123,202` |
| 2026-06-12 | **PBRS reward redesign** — Three new standalone training scripts implementing Potential-Based Reward Shaping (Ng et al. 1999, Grzes & Kudenko 2009) as a thesis experiment. Replaces the fractional improvement formula (`α·Δd/d_prev`) with bounded exponential potentials (`Φ(s)=exp(-k·d²)`) using `gamma_shaping=1.0` (valid for episodic MDPs). Key parameters: `k_p=30, k_r=5, w_pos=w_rot=10.0`. Cosine angular distance replaces `_yaw_distance_rad` for smooth rotation metric. Episode terminates on both-threshold (pos<0.05m AND cos_rot<0.01) not position-only. Arm-through-table detection added. New files: `tasks/utils/reward_pbrs.py`, `train_push_pbrs_a.py` (Model A: PBRS only), `train_push_pbrs_b.py` (Model B: PBRS + pos→rot curriculum ramp), `train_push_pbrs_c.py` (Model C: PBRS + ASP with bug fixes). HPC scripts: `hpc/train_push_pbrs_{a,b,c}.slurm`. Design documented in `thesis_impl.md`. |
| 2026-06-12 | **PBRS Model A** — Single-agent PPO with PBRS dense reward. Hardcoded `rel_obs=True, rel_act=True`. Dense: `F=Φ(s')−Φ(s)` (zero-sum cycles, no near-goal negativity). Sparse: +5 pos-only (no termination), +2 both (terminates). Penalties: −5 tip/launch/OOB/table (terminates). Per-env PBRS logging when `num_envs≤50`. `train_push_pbrs_a.py` (836 lines). |
| 2026-06-12 | **PBRS Model B** — Model A + forced curriculum controller. Phase 1: `w_rot=0`, position-only termination at `pos_term_threshold=0.05` (fast cycles). Phase 2 (triggered by episodic position-only SR ≥ 0.5 for 20 consecutive iters; fixed 2026-06-26): `w_rot` ramps 0→10 over 200 iters, `pos_term_threshold` set to 0.0 on trigger (disables pos-only termination so rotation can be practiced). Phase 3: full multi-objective. `train_push_pbrs_curr.py`. |
| 2026-06-12 | **PBRS Model C** — Fork of `train_push_asp.py` with PBRS for Bob. Fixes: `bob_done_now` set on early completion, LSTM zeroed on early completion, `_bob_gave_rot_bonus` buffer added, progress reward removed (redundant with PBRS). Bob sparse changed: +5 position-only gate (was both-threshold), +2 both-threshold (terminates phase). Approach params tightened: `min_r=0.02, max_r=0.08, max_l=0.20`. Phase-end logs use `_prev_initial` snapshot for correct start positions. `train_push_pbrs_asp.py`. |
| 2026-06-16 | **PBRS Model D (GoalEncoder ablation)** — Fork of Model C with Bob's GoalEncoder removed. Bob config: `use_goal_encoder=False`, `pi_obj_dim=22` (PI-encoder sees full 22D per-object chunk: obj_state 14D + goal_pose 6D + dist 2D). Bob still uses PI-encoder + LSTM + MultiCategorical 4D×21 bins + PPOABC. Alice unchanged. Purpose: isolate whether the GoalEncoder's 8D latent compression bottleneck prevents Bob from learning translation under PBRS + adversarial Alice curriculum. If Bob translates without GoalEncoder, the 8D bottleneck was the cause. If not, the issue lies in ASP dynamics or PI-encoder inductive bias. `train_push_pbrs_asp_no_ge.py`, `hpc/train_push_pbrs_asp_noge.slurm`. |
| 2026-06-12 | **Fix P67 (Model B missing push_count increment)** — `env.push_count += 1` was absent from Model B's push loop. `check_done_pbrs` never triggered `max_push_done` → episodes had no budget cap, `episode_push_counts` always reported 0, `gave_completion`/`gave_rot_bonus` never reset between logical episodes. Added `env.push_count += 1` before `check_done_pbrs`. `train_push_pbrs_curr.py`. |
| 2026-06-12 | **Fix P68 (Model C catastrophe early termination)** — `_bob_should_end` and `_bob_early_end` were computed but never used (dead code). Bob continued pushing tipped/launched/OOB objects for the full phase budget. Removed dead code; added `_cata_early` handling after completion block: calls `handle_bob_phase_end` for catastrophe envs, sets `bob_done_now=True`, zeros `_bob_gave_rot_bonus`, includes catastrophe envs in EE reset set (`needs_ee_reset |= bob_done_now`). `train_push_pbrs_asp.py`. |
| 2026-06-12 | **Fix P69 (Model A double-scaled logging)** — `env._last_completion` subtracted `dense_pos * PBRS_W_POS` but `dense_pos` already includes `w_pos` scaling → squared the weight. Logging-only bug. Fixed to `reward - dense_pos - dense_rot`. `train_push_pbrs_no_curr.py`. |
| 2026-06-12 | **Fix P70 (launched objects getting positive rewards)** — PBRS reward computed BEFORE `check_done_pbrs` detected LAUNCHED/TIPPED/OOB. A launched object at Z=0.30m with favourable XY projection earned positive dense reward (+3.4). The `-10.0` override only applied to physics-terminated envs, not to LAUNCHED detected by `check_done_pbrs`. Fix: after `check_done_pbrs`, apply `reward[catastrophe & ~terminated] = -10.0` for launched/tipped/OOB. Move `episode_reward += reward` to after the override. All three PBRS scripts. Model C changed from additive `+= -5` to hard override `= -10.0` for consistency. `train_push_pbrs_no_curr.py`, `train_push_pbrs_curr.py`, `train_push_pbrs_asp.py`. |
| 2026-06-12 | **Fix P71 (goal sampling too far + OOB 2D)** — (a) `_sample_goals_filtered` only rejected goals < 0.05m from object. Goals > 0.45m away caused episodes to start already OOB → terminated on push 1 with -10 penalty. Added max-distance filter: rejects goals > 0.45m, resamples up to 10×. (b) `check_done_pbrs` OOB check used 3D distance (`obj_pos - goal_pos`), inflated by Z component (obj Z=0.05, goal Z=0.02, tipped Z=0.27). Changed to 2D: `obj_pos[:,:2] - goal_pos[:,:2]`. `wrapper_push.py`, `reward_pbrs.py`. |
| 2026-06-12 | **Fix P72 (termination reason logging)** — `check_done_pbrs` now returns `(done, reasons)` tuple where `reasons` is a dict of per-condition boolean masks (`terminated`, `max_pushes`, `success`, `launched`, `tipped`, `oob`, `pos_only`). Episode end lines in Models A & B show `end=SUCCESS+MAX_PUSH`, `end=LAUNCHED`, `end=OOB`, etc. `reward_pbrs.py`, `train_push_pbrs_no_curr.py`, `train_push_pbrs_curr.py`. |
| 2026-06-12 | **Fix P73 (Alice logging in Model C)** — Per-push Alice debug lines expanded from `push=N` to `push=N obj=(x,y,z) yaw=... len=... θ=...` showing object position and push parameters. `[ALICE END]` lines now include inferred invalid-goal reason: `INVALID(AIRBORNE)`, `INVALID(NO_DISP)`, `INVALID(SHALLOW)`, `INVALID(OOZ)`, or combinations. `train_push_pbrs_asp.py`. |
| 2026-06-12 | **Fix P74 (checkpoint overhaul)** — Periodic checkpoints changed from `model_{iteration}.pt` (accumulating files) to `latest_checkpoint.pt` (single file, overwritten) + `latest_iter.txt` (plain text iteration number). `model_best.pt` unchanged. Emergency and final saves use the same scheme. HPC slurm scripts updated: resume logic reads `latest_checkpoint.pt` + `latest_iter.txt` instead of parsing `model_*.pt` filenames. All three PBRS training scripts + all three PBRS HPC scripts. |
| 2026-06-12 | **Fix P75 (ppo.py load crash on latest_checkpoint.pt)** — `PPO.load()` parsed iteration from filename via `int(path.split("_")[-1].split(".")[0])`. With `latest_checkpoint.pt`, this produced `int("checkpoint")` → `ValueError`. Wrapped in try/except, defaults to 0. The actual iteration is always set via `--resume_iteration` from the training script. `algorithms/rl/ppo/ppo.py:117`. |
| 2026-06-16 | **PBRS Model D (GoalEncoder ablation)** — Fork of Model C with Bob's GoalEncoder removed. Bob's PI-encoder sees full 22D per-object chunk (obj_state + goal_pose + dist) directly, without the 8D latent compression bottleneck. All other architecture unchanged — ASP two-phase loop, Alice unchanged, PBRS dense reward, PPOABC + ABC buffer, historical pool. `train_push_pbrs_asp_no_ge.py`, `hpc/train_push_pbrs_asp_noge.slurm`. |
| 2026-06-16 | **Model B pos_term_threshold floor** — `pos_term_threshold` fade changed from `0.05→0.0` to `0.05→0.02` during curriculum ramp. Prevents position-only termination from becoming infinitely tight, preserving some position-only gating in Phase 3. `train_push_pbrs_curr.py:891`. |
| 2026-06-15 | **Fix P76 (validation scripts + configs + plotting)** — (a) Created `tasks/utils/validation_configs.py` with 20 predefined test scenes across easy/medium/hard difficulty with varied start→goal directions. (b) Restored `tests/validate_push.py` with original module-level import structure, added: `VisualizationMarkers` for goal flat-T-block and push arrows (green start, red end, blue cylinder), airborne/tipped/OOB detection (`obj_z > 0.10`, `|roll| > 0.3`, `|pitch| > 0.3`, `||obj_xy - goal_xy|| > 0.5`), per-push prediction logging (`bins=... r=... len=... θ=... pos=... rot=... z=...`), termination reason tracking (`stop_reason`), `--csv` flag for CSV output. (c) Created `tests/validate_push_asp.py` — separate script for ASP Model C Bob using `PPOABC` with GoalEncoder. Same markers + detection + logging. Loads `episode_manager` state for goal management. (d) Created `tests/plot_validation.py` — reads CSV files from multiple models and generates comparison plots: `overall_sr.png`, `sr_easy.png`/`medium.png`/`hard.png`, `sr_by_difficulty_grouped.png`, `avg_pushes.png`, `per_test_comparison.txt`, `summary.md` (markdown table). All four files created. |
| 2026-06-18 | **PBRS Model E (T-block + SE(2) d_pose)** — Replaces separate position/rotation PBRS potentials with a single SE(2) metric `d_pose = sqrt(dx² + dy² + L²·dθ²)` where `L=0.07m` is the T-block characteristic length. Observation slot `goal_dist(2)` replaced with `[d_pose, bearing]` where `bearing = atan2(dy, dx)`. Single PBRS potential `Φ(s) = exp(-k·d_pose²)` with `k=30, w=10`. Success: `d_pose < 0.055m`. Eliminates competing position/rotation gradients. No rotation bonus, no `_bob_gave_rot_bonus`. Progress reward uses single d_pose metric. `wrapper_push_asp.py` gains `dpose_obs`, `char_length`, `dpose_threshold` params (backwards compatible). `reward_pbrs.py` gains `compute_dpose()`, `potential_dpose()`, `compute_pbrs_reward_dpose()`. `train_push_pbrs_asp_dpose.py`, `hpc/train_push_pbrs_asp_dpose.slurm`. |
| 2026-06-18 | **PBRS Model F (Disc + position-only d_pose)** — Fork of Model E with rotationally-symmetric disc (10cm diameter, 6cm tall, `CylinderCfg`) replacing the T-block. `char_length=0.0` collapses d_pose to pure 2D position distance `sqrt(dx² + dy²)`. Rotation still observed but not rewarded — network learns to ignore meaningless yaw. New env config `PushTaskCuRoboDiscEnvCfg` in `tasks/push_task_curobo_disc.py` uses `sim_utils.CylinderCfg(radius=0.05, height=0.06)` with `collision_props`, `mass_props(density=300)`, `physics_material(friction=0.6)`. Success: `d_pose < 0.05m`. `events.py` `reset_objects_to_random_safe_pose()` gains `spawn_z`, `settled_z` params for disc half-height (0.03m). `train_push_pbrs_asp_disc.py`, `hpc/train_push_pbrs_asp_disc.slurm`. |
| 2026-06-18 | **Fix P77 (workspace border markers corrected)** — Zone border cuboids in `push_primitive_1arm_env.py` were at X=±0.65, Y=0.20/0.75 — 15cm wider in X and 5cm wider in Y than the actual IK workspace. Corrected to match `_WS_X=[-0.50,0.50], _WS_Y=[0.25,0.70]`: top Y 0.75→0.70, bottom Y 0.20→0.25, left/right X ±0.65→±0.50, bar widths 1.32→1.02, bar heights 0.57→0.47. Fix propagates to all task envs via `Push1ArmSceneCfg` inheritance. |
| 2026-06-18 | **Fix P78 (table resized)** — Table shrunk from 2.0×2.0m to 1.40×1.00m, center moved (0,0.5)→(0,0.40). New edges: X∈[-0.70,0.70], Y∈[-0.10,0.90] — 20cm margin from IK workspace on left/right/far, 35cm on robot side. Eliminates wasted table area that was never reachable. `push_primitive_1arm_env.py`. |
| 2026-06-18 | **Fix P79 (placement_bounds aligned to IK workspace)** — `placement_bounds` X∈[-0.75,0.75]→[-0.50,0.50], Y∈[0.20,1.00]→[0.25,0.70] — now identical to IK workspace. `table_bounds` tightened to match new table: X∈[-1.0,1.0]→[-0.70,0.70], Y∈[-0.5,1.5]→[-0.10,0.90]. Goals outside IK workspace are now invalid. `wrapper_push_asp.py`, `wrapper.py`, `train_curobo.py`. |
| 2026-06-18 | **Fix P80 (absolute workspace OOB termination)** — Old OOB check used relative distance from goal (`‖obj−goal‖ > 0.5m`); object 0.49m from goal but outside IK workspace was not detected. Replaced with absolute IK workspace bounds check `_oob_ws = (obj_x < -0.50) ∣ (obj_x > 0.50) ∣ (obj_y < 0.25) ∣ (obj_y > 0.70)`. For Bob: feeds into existing catastrophe detection → −10 penalty, early phase end. For Alice: new `_alice_oob` block immediately resets the episode with −3.0 penalty when object leaves IK workspace during Alice's phase (object unreachable). Applied to all four PBRS ASP scripts: Models C, D, E, F. |
| 2026-06-18 | **Fix P81 (approach radius tuned per object)** — T-block scripts (Models C, D, E): `min_r` 0.02→0.03m. Disc script (Model F): `min_r` 0.02→0.06m, `max_r` 0.08→0.12m. The disc has radius 0.05m — old `min_r=0.02` placed the push start inside the disc. New `min_r=0.06` keeps the gripper 1cm outside the disc surface. `max_r=0.12` gives the agent room to build speed before contact. T-block `min_r=0.03` adds 1cm safety margin from the narrowest part of the T-block stem. `train_push_pbrs_asp.py`, `train_push_pbrs_asp_dpose.py`, `train_push_pbrs_asp_disc.py`, `train_push_pbrs_asp_no_ge.py`. |
| 2026-06-19 | **PBRS Model G (T-block + time-based ASP)** — Replaces outcome-based Alice reward (+5 fail/−1 succeed, Plappert 2021) with Sukhbaatar's time-based reward `R_A = γ_sp · max(0, t_B − t_A)` where `t_A` = Alice's push count at phase end, `t_B` = Bob's push count at phase end or early completion, `γ_sp = 0.5`. Scale chosen to match Bob's +5 completion bonus: `γ_sp = 5 / max(t_B − t_A) = 5/9 ≈ 0.5`. Restores Sukhbaatar's self-regulating curriculum: Alice is incentivized to find goals she can create efficiently (fewer pushes) that Bob takes many pushes to solve — naturally sitting at the frontier of Bob's capability. Shallow goal penalty removed (`skip_shallow_penalty=True` in `validate_goal()`) — time-based reward makes it redundant since easy goals yield small `t_B − t_A`. Bob reward unchanged (PBRS d_pose). ABC disabled. `wrapper_push_asp.py` gains `time_based_alice`, `alice_reward_scale`, `alice_phase_push_count` (backward-compatible). `goal_validator.py` gains `skip_shallow_penalty` flag. `train_push_pbrs_tasp_dpose.py`, `hpc/train_push_pbrs_tasp_dpose.slurm`. |
| 2026-06-19 | **PBRS Model H (Disc + time-based ASP)** — Fork of Model G with disc object. `char_length=0.0`, `dpose_threshold=0.05`. Same time-based Alice reward. Tests whether Sukhbaatar's self-regulation prevents the toxic curriculum collapse observed in Model F (Alice creating impossibly hard goals for Bob while Bob's SR declined from 29% to 6.5%). `train_push_pbrs_tasp_disc.py`, `hpc/train_push_pbrs_tasp_disc.slurm`. |
| 2026-06-19 | **Future models planned** — Model I/J: G/H + Bob time penalty `R_B += −γ_sp · t_B` (full Sukhbaatar reward for both agents, adds urgency for Bob to solve quickly). Model K/L: G/H + ABC enabled (`β=0.5`, PPO-style BC clipping per Plappert 2021) to test whether time-based Alice + ABC produces stronger curriculum than either alone. |
| 2026-06-26 | **Fix P82 (Model B curriculum trigger fixed)** — The old Phase‑2 trigger thresholded `mean_pos_err` (iteration‑mean of *all* per‑push position errors) against `CURRICULUM_POS_THRESHOLD=0.08m` with a brittle 50‑iteration AND gate. The per‑push mean was dominated by early/mid‑episode pushes far from a freshly randomised goal and contaminated by catastrophe envs (launched/tipped/OOB), creating a structural floor well above 0.08m. `curriculum_active` stayed `False` forever, `w_rot=0.0` permanently, and rotation received zero reward signal. Fix: (1) Replaced metric with **episodic position‑only SR** (= episodes terminated by `pos_only` or `success` / total episodes per iteration), computed from `done_reasons["pos_only"] | done_reasons["success"]` in the done‑handling block. (2) Trigger fires when all 20 consecutive iterations have `pos_ep_sr ≥ 0.5` (`CURRICULUM_ENTER_THRESHOLD=0.5`, `CURRICULUM_LOOKBACK=20`). (3) On trigger, `pos_term_threshold` is set to `0.0` (disables pos‑only termination), so episodes run to `max_pushes` and rotation can be practiced. (4) Phase‑2 ramp: `w_rot` 0→10 over 200 iters; `pos_term_threshold` stays at 0.0 (no fade). Phase‑2 Model B is now identical to Model A except the staged ramp — a clean "curriculum vs no curriculum" comparison. `train_push_pbrs_curr.py` (lines 493‑503,759‑761,880‑901,912). SLURM updated: `SEED=42`, `MAX_ITERATIONS=2600`, `EXP_NAME` → `hpc_pbrs_curr_${NUM_ENVS}env_fixed`. |
| 2026-06-25 | **SAC+HER Push Baseline planning** — Designed DirectRLEnv push environment based on `throwing_enviroment` architecture (macro-action decimation, SB3 `model.learn()` delegation, custom `DirectRLVecEnv`). Objective: off-policy SAC with hindsight relabeling as external calibration baseline vs published planar-pushing results (Haarnoja 2018, Andrychowicz 2017). |
| 2026-06-26 | **SAC+HER Push Baseline implemented** — Created `tasks/push_direct_env.py` (553 lines, `PushDirectEnv(DirectRLEnv)`), `tasks/push_direct_env_cfg.py` (240 lines, `PushDirectEnvCfg`), `tasks/sb3_vec_env.py` (128 lines, `DirectRLVecEnv(VecEnv)`), `tasks/utils/action_push_continuous.py` (78 lines), `train_push_sac_her.py` (182 lines), `hpc/train_push_sac_her.slurm` (155 lines), `tests/validate_push_sac.py` (579 lines). One outer step = one complete push macro-action (72 substeps via cuRobo IK inside `_apply_action()`). Observation: dict with `desired_goal`/`achieved_goal` for SB3 `HerReplayBuffer`. No ManagerBasedRLEnv overhead. |
| 2026-06-26 | **Definitive validation campaign kickoff** — Re-evaluated all 7 Isaac models (A_simp, B_curr, E–H) on identical 30 T-block scenes using the current `validation_configs.py` (tests 1–10: R_* rotation-heavy, tests 11–20: pos_only, tests 21–30: pos_rot). B_curr validated from the P82-fixed `hpc_pbrs_curr_528env_fixed` checkpoint (iter 2600) at both 26.06.26 and 26.06.28. |
| 2026-06-28 | **Definitive head-to-head comparison** — A_simp (26.06.20 checkpoint, iter 2400) run on the identical 30 T-block scene set used for B_curr. Clean comparison: A_simp 80.0% scene SR (100% pos-only, 70% pos+rot) vs B_curr 76.7% (100% pos-only, 65% pos+rot). Single-agent beats curriculum by 3.3pp on identical scenes. Both models fully solve position-only (100%). |
| 2026-06-29 | **Results organization** — All validation CSVs, plots, comparison outputs, and legacy data consolidated into `/home/vladi/IsaacLab/master_isaac/results/` organized by model (A_simp, B_curr, C_asp, E_asp_dpose, F_asp_disc, G_tasp_dpose, H_tasp_disc) with a `comparison/` subfolder for cross-model plots. See §12. P82 curriculum fix confirmed working: B_curr achieves 76.7% with properly triggered Phase 1→2 progression. |
| 2026-06-29 | **Validation config change documented** — The current `validation_configs.py` has tests 1–10 as T-block R_* rotation scenes (pos_rot). The 26.06.20 validation used an earlier version where tests 1–10 were D_* disc scenes (disc_pos). The 26.06.26/28 campaign uses the current config exclusively for fair comparison. Gym-pusht CSVs and legacy disc-protocol CSVs are preserved in `results/legacy/`. |
| 2026-06-26 | **SAC+HER Fix S1 (Q-function divergence)** — First HPC run (528 envs, 3000 iters) revealed SAC critic loss exploding (56.5 → 1130) and actor loss diverging (−2.53 → 79.3). Root causes: (a) `gamma=0.99` too high for 5-step episodes — all pushes got near-equal weight in return, making credit assignment impossible. (b) `completion_bonus=5.0` + `dense_alpha=3.0` produced per-push rewards exceeding 6.0, causing Q-target variance. Fixes: `gamma 0.99→0.95`, `completion_bonus 5.0→2.0`, `dense_alpha 3.0→1.0`, `buffer_size 200K→100K`. |
| 2026-06-26 | **SAC+HER Fix S2 (TensorBoard async crash)** — SB3's TensorBoard `SummaryWriter` background thread crashed with `FileNotFoundError` on the TMPDIR-bind-mounted events file mid-training, killing `model.learn()` at 626K/1.58M timesteps (40%). Fixed: `tensorboard_log=None` (disabled TB entirely), wrapped `model.learn()` in `try/except FileNotFoundError` as safety net. |
| 2026-06-27 | **gym-pusht testbed: Models A/B/C ported** — Native gym-pusht counterparts of PBRS Models A/B/C, to get a fast, controlled CPU testbed that isolates the *reward/curriculum* question from the robotic-task confounds (compute mismatch, IK gating, contact). Reuses the SAME custom `PPO`/`PPOABC`/`ActorCriticPush` + `EpisodeManager` + `validate_goal` + `reward_pbrs` unchanged — only the environment differs. New files: `tasks/utils/gym_push_primitive_env.py` (smart `gym.Env`: 1 step = 1 push macro, PBRS reward + thesis-gate done computed inside; never signals terminated/truncated, self-resets, reports via `info`; `TorchVecAdapter` wraps `AsyncVectorEnv` for the custom-PPO tensor contract), `tasks/utils/gym_push_asp_env.py` (single-process ASP env reusing EpisodeManager/validate_goal/PBRS), `train_{a,b,c}_gym_pbrs_*.py`. A/B use AsyncVectorEnv (CPU-parallel); C is single-process (ASP's Alice↔Bob cross-phase delayed reward forces central orchestration). All run in `.master_venv` (no Isaac/cuRobo). See §10. |
| 2026-06-27 | **gym freeze fix + thread caps + benchmark** — A debug benchmark **froze the desktop**: a missing `if __name__=="__main__"` guard made `spawn` workers re-import `__main__` and recursively re-spawn (a spawn fork-bomb), compounded by `fork` + torch's 6-threads-per-process default (32 workers × 6 ≈ 192 threads on 12 logical CPUs). The *training* scripts already guard env creation inside `main()` (unaffected); added `OMP/MKL/OPENBLAS/NUMEXPR=1` + `torch.set_num_threads(1)` thread caps to all gym scripts + the env module as defence, and `AsyncVectorEnv(autoreset_mode=DISABLED, context="spawn")`. Benchmark (Ryzen 5 5600X, 6c/12t): N=6 AsyncVectorEnv = **91 push-macros/s**; single-process ASP (C) = **22.4 push/s**. |
| 2026-06-27 | **gym A diagnostic: compute-starved at N=6, not broken** — A local 11k-iter A-gym run showed flat PosErr (~0.26 m) / Best SR ~0.006. Oracle-push probe proved the env is fine: an ideal goal-directed push moves the object **+0.23 m goal-ward at any approach radius**, and the trained checkpoint moved it **+0.029 m/push vs +0.002 random (15× learning)** — but slowly because N=6 gives a tiny PPO batch (6×15=90 samples/update vs Isaac's 7920) and only ~144k pushes (~1% of Isaac's stabilization budget). Added `--push_nsteps` arg (bigger batch without more processes). Decision: **move gym A/B/C to HPC** (many CPU cores → big batch → faithful `k_p=30` learns). New HPC scripts `hpc/train_gym_{a,b,c}.slurm` (CPU partition, `isaac-lab.sif` + `gym_overlay.img` + bound `gym-pusht`, `num_envs=$SLURM_CPUS_PER_TASK`, `device=cpu`, auto-resubmit). |
| 2026-06-27 | **"Isaac Lab is the right sim for ASP" — measured** — Controlled throughput (one continuous slurm segment, 528-env): single-agent **A**, curriculum **B**, and ASP **C** all run at **~0.022 it/s (~172 push-macros/s)** — ASP's 2-agent machinery (2 PPO updates, GoalEncoder, ABC, historical pool) adds **~0 per-iteration wall-clock** (the shared 528-env cuRobo-IK/physics dominates), so the Isaac A-vs-ASP comparison is genuinely **compute-matched**. Isaac reaches 1M pushes in **~1.6 h** (batch 7920); single-process gym ASP in **~12.4 h (~7.7× slower)** — ASP parallelises for free on GPU-batched sim but is forced single-process on CPU. Resolves critique **C2** (≈**1.6 days / 3000 iters**, not "a few hours"). Distinct from the Slide-11 ManagerBasedRLEnv-vs-DirectRLEnv API-overhead claim. |

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
- Bob reward: sparse `{+1/−1/+5}` + **dense per-push improvement** (Fix P53 → Fix P64): `R = α·(d_prev−d_now)/d_prev + α·(y_prev−y_now)/y_prev − β·d_now − β_rot·y_now` where α=3.0, β=0.5, β_rot=0.25 (normalised fractional, matching Fix P63/P64). Dense reward has no completion bonus to avoid double-counting. + phase-end progress reward `r = clamp(0.6·Δpos/init_pos + 0.4·Δrot/init_rot, −1, +1)` paid once at Bob termination. Init errors captured on first Bob step. Gives 100% episode coverage — every Bob trial produces a grade plus per-push directional signal. `bob_timesteps` reduced 200→100 to halve credit-assignment horizon for single-object push task. (Dense per-step delta was tested v2–v5 and reverted per Fix P27 on the cuRobo per-step variant — the push-primitive dense reward is per-push, not per-step, and uses improvement deltas not potential shaping.)
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
|  |---|
| `train_curobo.slurm` | Production cuRobo training run (baseline sparse) |
| `train_curobo_shaping.slurm` | Production cuRobo + `--diag_alice_shaping` (Fix P31 — EE→obj proximity) |
| `train_curobo_large.slurm` | Large-scale (512+ envs) |
| `train_curobo_profile.slurm` | 3-iteration profiler run |
| `train_push.slurm` | Push-PPO baseline (absolute world-coord obs) |
| `train_push_rel.slurm` | Push-PPO baseline + `--rel-obs` (30D object-relative delta appended) |
| `train_push_rel_full.slurm` | Push-PPO baseline + `--rel-obs --rel-act` (full object-relative: obs + actions) |
| `train_push_asp.slurm` | Push-ASP — object-relative push primitives with Alice/Bob self-play |
| `train_push_pbrs_simp.slurm` | PBRS Model A — PBRS only, 528 envs, 3000 iters |
| `train_push_pbrs_curr.slurm` | PBRS Model B — PBRS + episodic-pos-SR curriculum, 528 envs, 2600 iters, seed=42 |
| `train_push_pbrs_asp.slurm` | PBRS Model C — PBRS + ASP, 528 envs, 3000 iters |
| `train_push_pbrs_asp_noge.slurm` | PBRS Model D — PBRS + ASP, GoalEncoder ablated, 528 envs, 3000 iters |
| `train_push_pbrs_asp_dpose.slurm` | PBRS Model E — T-block + SE(2) d_pose, 528 envs, 3000 iters |
| `train_push_pbrs_asp_disc.slurm` | PBRS Model F — Disc + position-only d_pose, 528 envs, 3000 iters |
| `diagnostic_tests.slurm` | Full 4-test suite on HPC |
| `test1_ppo_reward.slurm` | Test 1 only |
| `test2_alice_exploration.slurm` | Test 2 only (200 iters, random Bob) |
| `test3_asp_tug_of_war.slurm` | Test 3 only (50 iters, full pipeline) |

#### Supporting Utilities

- `utils/historical_pool.py`: ring buffer of past 5 snapshots, `sample_env_subset()`
- `utils/episode_manager.py`: phase tracking, goal storage, checkpoint support
- `utils/profiler.py`: `TrainingProfiler` with `section()` context manager, `get_section_frac()`
- `utils/goal_validator.py`: `validate_goal()` — minimum displacement threshold check
- `tasks/utils/reward_pbrs.py`: PBRS utility — bounded exponential potentials (`k_p=30, k_r=5`), cosine angular distance, `compute_pbrs_reward()`, `check_done_pbrs()`, `gamma_shaping=1.0` (episodic PBRS)
- `tasks/utils/validation_configs.py`: 20 predefined test scenes (easy/medium/hard, varied directions) for push model validation evaluation
- `tests/validate_push.py`: Push-PPO model validator — loads `ActorCriticPush`, runs test scenes, outputs per-push predictions, detects airborne/tipped/OOB
- `tests/validate_push_asp.py`: ASP Model C Bob validator — loads `PPOABC` + GoalEncoder, runs test scenes with same features
- `tests/plot_validation.py`: Validation result plotter — reads CSV files, generates SR/pushes/difficulty comparison plots and markdown summary
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
| **Fix P45** | **Zero-length push allowed** — `decode_push_action` clamp `[0.01, 0.20] → [0.0, 0.20]`. The agent can now output `length=0` to hold position instead of always pushing. Fixes the MDP where the agent physically could never stop, causing objects to overshoot goals. With length=0, `Xf=Xs, Yf=Ys`, the push phase holds at contact height and the object stays put. | **Critical (Push)** | ✅ Fixed | `action_push.py:145-153` |
| **Fix P46** | **Completion terminates episode** — `check_done` now includes `at_goal_pos` (pos_err < 0.05m) so the episode ends immediately on success. Previously the episode ran to max_pushes, allowing subsequent pushes to overshoot the goal. The clean return signal (+5 bonus on the last push before reset) gives PPO a sharper gradient toward achieving and stopping. | **Critical (Push)** | ✅ Fixed | `wrapper_push.py:261-274` |
| **Fix P47** | **Rotation reward rebalanced** — `PUSH_DENSE_ROT_ALPHA` 5.0→1.0. At the old value, rotation improvement (up to 2.5 reward per push) dominated position improvement (typically 0.12 per push) by ~20×. PPO learned rotation control (RotSR=42%) while ignoring position (PosErr=0.25m flat). Now both components produce comparable gradients at observed step sizes. | **Critical (Push)** | ✅ Fixed | `wrapper_push.py:26` |
| **Fix P48** | **Object-relative push actions for Push-ASP** — absolute `(Xs, Ys)` action space had ~2% contact probability per push, making Alice unable to bootstrap (object never moved → no reward → no gradient). New parameterization: `(r, φ, length, θ)` where `r∈[0.02, 0.08]m` is offset from object center, `φ∈[-π,π]` is approach angle in object's frame, `length∈[0,0.20]m`, and `θ∈[-π,π]` is push direction in world frame (decoupled from approach). Conversion: `Xs=obj_x+r·cos(obj_yaw+φ)`, `Ys=obj_y+r·sin(obj_yaw+φ)`, `Xf=Xs+len·cos(θ)`, `Yf=Ys+len·sin(θ)`. Guarantees ~95%+ contact rate. World-frame θ makes translation trivially learnable (θ≈atan2(goal_y-obj_y, goal_x-obj_x)). Object-frame φ provides rotational equivariance for contact-point selection. `compute_push_waypoints()` unchanged — only decode step replaced. Bob observation updated to include relative goal features: `[delta_x, delta_y, rel_yaw, pos_dist, rot_dist]` (5D) with world-frame deltas aligned to world-frame θ for direct policy mapping. Alice obs: 20D (6+14), Bob obs: 25D (6+14+5). | **Critical (Push-ASP)** | ✅ Fixed | `tasks/utils/action_push_relative.py` (new), `train_push_asp.py`, `tasks/utils/wrapper_push_asp.py`, `net.md` |
| **Fix P49** | **Push-ASP IK-death fix** — IK failures in waypoint loop fell back to `prev_joint_cmd` (last commanded position), which could differ from the arm's actual physics state `cur_joints`. When gravity or contact forces displaced the arm from the previously-commanded position, the fallback would command a snap-back teleport, potentially through the table or into the object. Now IK failures hold `cur_joints` (actual physics state), and `prev_joint_cmd` is overwritten from `cur_joints` on failure so subsequent waypoints also freeze in place. The arm stays at its current physical pose until the push trajectory completes — no teleport, no table penetration, no phantom object-launching on IK fail. | **Critical (Push-ASP)** | ✅ Fixed | `train_push_asp.py:947-960` |
| **Fix P53** | **Bob dense per-push improvement reward (Push-ASP)** — Bob's sparse-only `{+1/−1/+5}` fired on ~0.14% of pushes, producing zero-mean GAE advantages that gave PPO no directional signal. Added `compute_bob_dense_push_reward()` mirroring `wrapper_push.py`'s dense improvement formula: `R = α·(d_prev−d_now) + α_rot·(y_prev−y_now) − β·d_now − β_rot·y_now` with `α=12, α_rot=1, β=0.5, β_rot=0.25`. All components clamped per Fix P30. **No completion bonus** in dense — that comes from sparse, avoiding double-counting of the +5 that caused critic instability (Fix P18, P29). Bob's total per-push reward = dense + sparse. Completion detection (`bob_achieved_completion >= 4.0`) still gated on sparse only to prevent false triggers from large dense improvements on near-miss pushes. Alice stays sparse-only (preserves adversarial curriculum). `_yaw_distance_rad` helper added alongside existing `_rot_distance_rad`. | **Critical (Push-ASP)** | ✅ Fixed | `wrapper_push_asp.py:484-523`, `train_push_asp.py:994-1004` |
| **Fix P54** | **Push-ASP observation dimension fix** — `wrapper_push_asp.py` still used `_OBS_ROBOT_DIM=7` (with gripper) after Fix P39 removed the gripper from push-task observations (Isaac Lab outputs 28D: `ee_pose(6)+obj(14)+goal(6)+dist(2)`). Caused `RuntimeError` at `storage.add_transitions`: tensor a (29) must match tensor b (28). Fixed `_OBS_ROBOT_DIM 7→6` in wrapper, `robot_state_dim 7→6` in two PPO config blocks, and GoalEncoder sampling offset `_robot_dim 7→6`. Alice obs 21→20D, Bob obs 29→28D. | **Critical (Push-ASP)** | ✅ Fixed | `wrapper_push_asp.py:38-44`, `train_push_asp.py:387,431,580` |
| **Fix P55** | **Push-ASP missing PPO update calls** — `perform_alice_update()` and `perform_bob_update()` were defined but never invoked in the main `while bob_updates < args.max_iterations:` loop. Training printed `[Iter 0]` forever — `bob_updates` never incremented, no PPO updates ran. Added both calls after the rollout loop: `perform_alice_update(); perform_bob_update(current_bob_obs)`. | **Critical (Push-ASP)** | ✅ Fixed | `train_push_asp.py:1307-1308` |
| **Fix P56** | **Push-ASP tight spawn bounds + random yaw** — Default placement bounds `X∈[-0.35,0.35]×Y∈[0.45,0.80]` let objects spawn near workspace edges, causing 46% invalid goal rate from single-push OOB. Objects spawned with fixed identity quaternion (yaw=0), so all goals/bob-starts had identical orientation — rotation curriculum didn't exist. Added `_rand_reset_objs()` helper with `X∈[-0.04,0.04]×Y∈[0.35,0.45]` (8cm×10cm box at table center) + `random_yaw=True` (uniform [-π,π]). `_initial_states_from_spawn` updated to read `target_yaw` from spawn result. `reset_objects_to_random_safe_pose` gained optional `random_yaw` param (default False, backward-compatible). All 7 reset call sites routed through `_rand_reset_objs`. | **Critical (Push-ASP)** | ✅ Fixed | `wrapper_push_asp.py:183-194,310-324`, `events.py:121-168` |
| **Fix P57** | **Push-ASP workspace clamp + EE sync + Bob safety penalty + slurm path** — (a) Push start/end `(Xs,Ys,Xf,Yf)` clamped to workspace ±0.02m with recomputed length/theta so credit matches executed trajectory; (b) `ee_pos_local`/`ee_quat_w`/`prev_joint_cmd` synced to physics after each push (matching `train_push.py:687-688`), fixing stale-approach-start bug; (c) Bob gets `-5.0` penalty for `obj_lifted || robot_through_table` pushes — same magnitude as +5 completion, symmetric gradient; (d) slurm resubmit path was `$PROJECT_ROOT/asyncDualPlayPPO/hpc/...` (doubled `asyncDualPlayPPO`) → fixed to `$PROJECT_ROOT/hpc/...`. | **Critical (Push-ASP)** | ✅ Fixed | `train_push_asp.py:908-917,984-987,998-1001`, `train_push_asp.slurm:36` |
| **Fix P50** | **Object-relative observations for Push-PPO baseline** — the 28D flat observation `[ee_pose(6) | obj_state(14) | goal_pose(6) | goal_dist(2)]` contains raw world-frame positions, forcing the network to learn `atan2(goal_y-obj_y, goal_x-obj_x)` internally from a fully-connected MLP before it can predict the correct push direction `θ`. The result: the policy converges to a fixed θ regardless of goal position, succeeding only when goals happen to be in the landing zone of that fixed push (SR ~5%). New `--rel-obs` flag appends `[rel_dx, rel_dy] = goal_pos[:2] - obj_pos[:2]` (2D world-frame delta) to the observation (28D→30D), giving the policy direct access to the answer. The network sees the direction-to-goal explicitly and maps it to a push θ bin — `atan2` is trivialized. The flag is fully backward-compatible (off by default). New HPC script: `hpc/train_push_rel.slurm`. | **Critical (Push)** | ✅ Fixed | `wrapper_push.py`, `train_push.py`, `hpc/train_push_rel.slurm` (new) |
| **Fix P51** | **Slurm auto-resume iteration counter bug** — both `train_push.slurm` and `train_push_rel.slurm` passed `--chkpt` (which loads model weights and optimizer state) but never passed `--resume_iteration`. Because `train_push.py` initializes `iteration = args.resume_iteration if args.chkpt else 0`, every resumed job started counting from iteration 0 regardless of how many iterations the previous job completed. A 3-job chain with ~1,850 iters each produced 5,550 paid-for iterations but only 1,850 worth of training progress. Both scripts now extract the iteration number from the checkpoint filename and pass `--resume_iteration $ITER_NUM`. | **Critical (Infra)** | ✅ Fixed | `hpc/train_push.slurm`, `hpc/train_push_rel.slurm` |
| **Fix P52** | **Episode start-state logging** — per-env bin logging was spammy (all 512 envs every push) and uninformative (couldn't tell if object actually moved). Now: (a) per-env lines only fire for pushes with `length > 0` (filters out zero-length holds); (b) per-episode log includes `start=(x,y,z) yaw=` position/orientation so the full trajectory (start → goal → final) is visible in one line without needing to cross-reference; (c) iteration summary ends with `| rel` or `| abs` to disambiguate which observation mode is running. `_ep_start_pos`/`_ep_start_euler` buffers added to `PushEnvWrapper`, snapshot on first `capture_pre_push` of each episode. | **Medium (Push)** | ✅ Fixed | `wrapper_push.py`, `train_push.py` |
| **Fix P58** | **Randomised object spawn for Push-PPO** — Object always spawned at fixed `(0, 0.5, 0.05)` allowing policy to memorise approach point. Random spawn `X∈[-0.4,0.4] Y∈[0.3,0.7] yaw∈[0,2π]` forces object-position-dependent learning. `_randomize_object_spawn()` teleports via `write_root_pose_to_sim`; `_get_push_obs()` recomputes observation after teleport. Always active (no flag). | **Critical (Push)** | ✅ Fixed | `wrapper_push.py`, `train_push.py` |
| **Fix P59** | **Trivial-goal filter for Push-PPO** — Independent randomisation of spawn and goal produced ~2.5% episodes where object was already within 0.05m of goal → instant +5 bonus. `_sample_goals_filtered()` reads object scene position and rejects goals too close. Goal sampling moved from `reset_done_envs` to `train_push.py` reset block (after object randomisation). Diagnostic print `[P4 filter]` on resample. | **Medium (Push)** | ✅ Fixed | `wrapper_push.py`, `train_push.py` |
| **Fix P60** | **Push-PPO compute check** — `observation_manager.compute()` called outside `env.step()` in `_get_push_obs()`. Diagnostic `[P5 check]` print confirms safe (push_policy group has no sensors/cameras). Remove print after verification run. | **Low (Push)** | ✅ Fixed | `wrapper_push.py` |
| **Fix P61** | **Push-PPO _ep_started cleared after reset completes** — `_ep_started` was cleared in `reset_done_envs()` BEFORE the base reset. `_ep_started[ids]=False` moved to `train_push.py` reset block after reset succeeds. | **Medium (Push)** | ✅ Fixed | `wrapper_push.py`, `train_push.py` |
| **Fix P62** | **Object-relative action decode (--rel-act) for Push-PPO** — New flag swaps `decode_push_action` for `decode_push_action_relative`. 4D×21 bins now parameterize `(r, φ, len, θ)` with `r∈[0.02,0.08]m` offset from object center, `φ∈[-π,π]` approach angle in object frame. `Xs = obj_x + r·cos(obj_yaw+φ)` — guaranteed ~100% contact. Works alone or with `--rel-obs`; combined as `--rel-obs --rel-act` gives full object-relative pipeline. World-frame θ aligned with rel-obs delta features. New HPC script: `hpc/train_push_rel_full.slurm`. Iteration summary shows `rel_full` / `rel_act` / `rel_obs` / `abs`. Per-push log shows `r=` instead of `Xs=/Ys=`. | **Critical (Push)** | ✅ Fixed | `train_push.py`, `hpc/train_push_rel_full.slurm` (new) |
| **Fix P63** | **Normalised fractional reward** — `PUSH_DENSE_ALPHA=12` (position) and `PUSH_DENSE_ROT_ALPHA=1` (rotation) replaced by single `PUSH_DENSE_ALPHA=3.0` with normalised deltas: `pos_imp = α·(d_prev−d_now)/d_prev`, `rot_imp = α·(y_prev−y_now)/y_prev`. Both are unitless fractions of remaining error — halving the distance earns α×0.5 regardless of domain. Denominators clamped at 0.01. One hyperparameter, no domain-specific scaling. | **Medium (Push)** | ✅ Fixed | `wrapper_push.py:25-26,266-267` |
| **Fix P64** | **Push-ASP dense reward was old non-normalised formula** — `compute_bob_dense_push_reward()` still used α=12, α_rot=1 without division by d_prev. Even improvement-producing pushes earned negative reward: a 1cm push from 30cm away cost `12×0.01 − 0.5×0.29 = −0.025`. Random pushes produced expected reward ≈ −0.15 with near-zero variance → Bob's PPO gradient starved. Normalised to α=3.0 with d_prev/y_prev division (clamped at 0.01), matching Fix P63. Dead copy-paste code (18 unreachable lines past return) removed. | **Critical (Push-ASP)** | ✅ Fixed | `wrapper_push_asp.py:487-517` |
| **Fix P65** | **Push-ASP obj_lifted false positives killing episodes** — Per-waypoint-substep Z‑check (`obj_lifted |= Z > 0.10`) fired on 45–55% of pushes, terminating episodes with −5 penalty. Push‑PPO (`train_push.py`) has no such check; it experiences identical physics but ignores momentary tipping. Objects briefly tip during pushes then settle — the check was a false alarm. Removed obj_lifted/robot_through_table tracking, the `_abnormal` handling block (which leaked +5 reward to Alice for Bob's "lifted" pushes), and the per-substep observation read. Also removed unused `ALICE_BOB_FAIL_REWARD`/`ALICE_BOB_SUCCESS_REWARD` import. | **Critical (Push-ASP)** | ✅ Fixed | `train_push_asp.py:688-691,932-979,989-992,997-1018,1180,1355-1356` |
| **Fix P66** | **Push-ASP Bob observation missing relative goal delta** — 28D flat observation required LSTM to learn `atan2` internally before predicting push θ. New `--rel-obs` flag replaces `goal_dist(2)` slot with `[rel_dx, rel_dy] = goal_xy − obj_xy`. Same 28D dimension — no architecture changes needed (avoids breaking `module.py`'s structured slicing). `PushASPEnvWrapper._get_push_obs()` overwrites the last 2D when enabled. Backward‑compatible (off by default). Iteration summary shows `| rel_obs`. | **Critical (Push-ASP)** | ✅ Fixed | `wrapper_push_asp.py:240-256`, `train_push_asp.py:120-123,202` |

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
Phase 1: Approach   (12 steps)  EE → above push start at approach height, tool-down
Phase 2: Descend    (16 steps)  EE down to contact height, tool-down
Phase 3: Push       (20 steps)  EE moves: contact_xy → push_endpoint_xy, tool-down
Phase 4: Retract    (16 steps)  EE up to approach height, tool-down
Phase 5: Return     ( 8 steps)  EE back to pre-push position at approach height, tool-down

Total: 72 substeps per push macro-action. Gripper always closed — no engage/release phases.
(Fix P34: 4D action + gripper-always-closed, Fix P32: substeps scaled from 115)
```

### 6.4 Action Space

MultiCategorical: **4D × 21 bins**  (Fix P6: was 11 bins → 0.03m/bin resolution, below 0.05m success threshold; Fix P34: 6D→4D)

| Dim | Parameter | Range | Description |
|-----|-----------|-------|-------------|
| 0 | `Xs` | `[-0.50, 0.50]` m | Push start X world coords |
| 1 | `Ys` | `[0.25, 0.70]` m | Push start Y world coords |
| 2 | `length` | `[0.00, 0.20]` m | Push length (zero allowed — Fix P45) |
| 3 | `theta` | `[-π, π]` rad | Push direction in world frame |

Xf = Xs + length × cos(theta),  Yf = Ys + length × sin(theta)

### 6.5 Observation Space

```
Base (28D, --rel-obs off):    [ee_pose(6) | obj_state(14) | goal_pose(6) | goal_dist(2)]
Relative (30D, --rel-obs on): [ee_pose(6) | obj_state(14) | goal_pose(6) | goal_dist(2) | rel_dx(1) | rel_dy(1)]
```

where `rel_dx = goal_x - obj_x`, `rel_dy = goal_y - obj_y` are the world-frame goal-offset deltas.
These give the policy direct access to the push direction without requiring the MLP to learn
`atan2` internally. The 2D delta is a sufficient statistic for the optimal push θ (ignoring
rotational goals): if the policy needs to push the object toward the goal, θ should
approximately equal `atan2(rel_dy, rel_dx)`.

The `--rel-obs` flag controls which mode is used. Both modes share the same `action_push.py`
decode logic and reward structure. Only the wrapper's `_build_obs()` and `obs_dim` differ.
The network (`module_push.py`) auto-adapts to observation dimension — no architecture changes
needed between modes.

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
- `γ = 1.0` — rotation improvement gain — reduced from 5.0 (Fix P47: rotation dominated position ~20×, agent converged to rigid-body rotation trick while ignoring translation)
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
obs (28D or 30D depending on --rel-obs)
  │
  ├─ Linear(obs_dim → 512) → ReLU     ← orthogonal init gain=sqrt(2) (Fix P1)
  ├─ Linear(512 → 256) → ReLU    ← orthogonal init gain=sqrt(2) (Fix P1)
  ├─ Linear(256 → 128) → ReLU    ← orthogonal init gain=sqrt(2) (Fix P1)
  ├─ LSTM(128 → 256)             ← hidden propagates across pushes within episode (Fix P3)
  │
  ├─ Actor head:  Linear(256 → 84) → (4, 21) → MultiCategorical  ← gain=0.01 (Fix P34: 6D→4D)
  └─ Critic head: Linear(obs_dim → 512) → ReLU → Linear(512 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 1)
```

**Weight init**: trunk layers use `orthogonal_(gain=sqrt(2))` for ReLU activations; actor head uses `gain=0.01` only (Fix P1). Previously gain=0.01 was applied to all layers, making activations ~100× too small and killing gradient signal.

**LSTM sequencing**: hidden state propagates push-to-push within an episode; zeroed only at episode done boundaries. Hidden states are stored at rollout time and yielded during PPO mini‑batch updates so `evaluate()` recomputes action log‑probs with the correct temporal context — the ratio `π_new/π_old` reflects genuine weight changes, not LSTM amnesia (Fix P13).

### 6.8 Files

| File | Purpose |
|  |---|
| `tasks/push_task_curobo.py` | Environment config (single agent, single object) |
| `tasks/utils/wrapper_push.py` | Push env wrapper: obs (28D abs / 30D rel), reward, reset, goals |
| `tasks/utils/action_push.py` | Push primitive: trajectory generation + decode (4D, 21 bins) |
| `algorithms/rl/ppo/module_push.py` | ActorCritic (flat MLP + LSTM, auto-adapts to obs_dim) |
| `train_push.py` | Training script (`--rel-obs` flag for object-relative δ) |
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

---

## 8. Validation Evaluation Suite <a name="validation-suite"></a>

> Added 2026-06-11. Standardized evaluation suite for fair comparison of all push-T model variants.

### 8.1 Overview

A 10-test validation suite that evaluates trained models on identical, deterministically-seeded (start, goal) pairs. Supports all three model types in two scripts:

| Script | Model Types | Action Pipeline |
|--------|-------------|-----------------|
| `tests/eval_suite.py` | Push-PPO (abs/rel), Push-ASP Bob | Push macro-actions (4D × 21 bins) → 72-substep waypoints → cuRobo IK |
| `tests/eval_suite_curobo.py` | ASP Bob (cuRobo step) | Per-step EE deltas (6D × 11 bins) → accumulate → cuRobo IK |

Both produce identical CSV format for merged analysis via `tests/eval_plot.py`.

### 8.2 Test Definitions (`tests/eval_test_defs.py`)

| # | Name | Goal Offset | N Episodes | Tests |
|---|------|-------------|------------|-------|
| 1 | `short_translation` | Δpos=0.08m, Δyaw=0 | 20 | Basic position accuracy |
| 2 | `long_translation` | Δpos=0.25m, Δyaw=0 | 20 | Multi-push sequencing |
| 3 | `small_rotation` | Δpos=0, Δyaw=30° | 20 | Rotation without translation |
| 4 | `large_rotation` | Δpos=0, Δyaw=90° | 20 | Large rotation control |
| 5 | `combined_easy` | Δpos=0.10m, Δyaw=30° | 20 | Basic combined |
| 6 | `combined_hard` | Δpos=0.25m, Δyaw=90° | 20 | Full difficulty |
| 7 | `precision` | Δpos=0.03m, Δyaw=10° | 20 | Fine correction |
| 8 | `boundary_push` | Goal near workspace edge | 20 | Edge reachability |
| 9 | `random_easy` | pos∈[0.05,0.15]m, rot∈[0,0.5]rad | 100 | Statistical (easy) |
| 10 | `random_hard` | pos∈[0.15,0.35]m, rot∈[0.5,2.5]rad | 100 | Statistical (hard) |

**Seeding**: `master_seed * 1000 + test_id * 100 + episode_idx` — all models face identical episodes.

**Spawn noise** (tests 1-8): Gaussian position jitter σ=0.02m XY, σ=0.17rad yaw around workspace center (0, 0.50). Test 7 (precision) uses reduced noise σ=0.01m, σ=0.09rad.

### 8.3 Success Criteria & Safety Guards

| Criterion | Threshold |
|-----------|-----------|
| Position success | pos_err < 0.05m |
| Rotation success | rot_err < 0.2rad (max-axis Euler wraparound) |
| Max pushes (push models) | 10 |
| Max steps (cuRobo Bob) | 100 |

**Safety termination (per-waypoint checks):**
- TCP Z < 0.01m → robot penetrating table → abort push
- Object Z > 0.08m → gripper lifted/scooped object → abort push
- Object XY out of bounds (|X|>0.75, Y<0.1 or Y>1.0) or Z<-0.1 → off-table/explosion → abort episode

### 8.4 Model Configuration

Models are specified as a list at the top of each script:

```python
MODELS = [
    {
        "name": "Push-PPO (abs)",
        "type": "push_ppo",        # ActorCriticPush, no GoalEncoder
        "rel_act": False,           # decode_push_action (absolute Xs, Ys)
        "rel_obs": False,           # 28D observation
        "checkpoint": "runs/.../agent/model_best.pt",
    },
    {
        "name": "Push-PPO (rel_full)",
        "type": "push_ppo",
        "rel_act": True,            # decode_push_action_relative
        "rel_obs": True,            # 30D observation (appends rel_dx, rel_dy)
        "checkpoint": "runs/.../agent/model_best.pt",
    },
    {
        "name": "Push-ASP Bob",
        "type": "push_asp_bob",    # ActorCritic with GoalEncoder + PI encoder
        "rel_act": True,            # always object-relative
        "rel_obs": True,            # 28D (replaces last 2 dims with rel_dx, rel_dy)
        "checkpoint": "runs/.../bob/model_best.pt",
    },
]
```

### 8.5 Execution

```bash
# Push-based models (non-headless, single env, visual markers)
python -m asyncDualPlayPPO.tests.eval_suite

# cuRobo step-based Bob
python -m asyncDualPlayPPO.tests.eval_suite_curobo

# Generate comparison plot from CSVs
python -m asyncDualPlayPPO.tests.eval_plot
```

### 8.6 Output

**CSV** (`results/eval_push_results.csv`): One row per episode with columns `model_name, model_type, checkpoint, test_id, test_name, episode_idx, seed, success, pos_error, rot_error, pushes_used`.

**Terminal**: Per-episode summary line showing start/goal/final positions+yaw, pushes used, final errors, termination reason (`success`, `max_pushes`, `off_table`, `terminated`).

**Plot** (`results/model_comparison.png`): 2×2 grouped bar chart — SR%, PosErr, RotErr, AvgPushes per test, one color per model.

### 8.7 Visualization

Visual markers (goal ghost + push start/end spheres + direction arrow) are shown during evaluation (non-headless mode).

Non-headless by default (`HEADLESS = False`). The evaluation viewport shows:
- **Orange flat T-block** at goal pose (updated per episode)
- **Green sphere** at push start point (Xs, Ys)
- **Red sphere** at push end point (Xf, Yf)
- **Blue cylinder arrow** showing push direction

### 8.8 Files

| File | Purpose |
|------|---------|
| `tests/eval_test_defs.py` | 10 test definitions + seeded episode generator |
| `tests/eval_suite.py` | Push-PPO & Push-ASP Bob evaluation |
| `tests/eval_suite_curobo.py` | cuRobo step-based Bob evaluation |
| `tests/eval_plot.py` | CSV reader + matplotlib comparison plot |
| `tests/validate_push.py` | Simpler single-model validator — per-push predictions, airborne detection, CSV output |
| `tests/validate_push_asp.py` | Simpler ASP Bob validator — same features, loads GoalEncoder |
| `tests/plot_validation.py` | Reads CSVs from validate scripts, generates comparison plots + markdown summary |
| `tasks/utils/validation_configs.py` | 20 predefined test scenes (easy/medium/hard) |

---

## 9. SAC + HER Push Baseline (DirectRLEnv) <a name="sac-her"></a>

> Implemented 2026-06-26.  Provides an off-policy continuous-action comparison point
> against all PPO push baselines.  Uses SB3 SAC with `HerReplayBuffer` for
> hindsight goal relabeling, running on `DirectRLEnv` (no manager overhead).

### 9.1 Motivation

Prior push baselines (Models 3–16) were all PPO variants with discrete 4D×21-bin
multi-categorical actions.  The reviewer noted: *"Missing the obvious working
baselines. No HER (the standard sparse goal-conditioned pushing baseline), no SAC
(despite citing Haarnoja). So '80% is good' has no external calibration vs
published planar-pushing results."*

SAC + HER addresses both gaps: SAC (Haarnoja et al. 2018) is the standard
off-policy continuous-action algorithm, and HER (Andrychowicz et al. 2017) is the
standard sparse-goal pushing baseline.  Together they provide a published-aligned
comparison point that calibrates the thesis's 80% SR claims.

### 9.2 Architecture

The architecture follows the `throwing_enviroment` project's proven
`DirectRLEnv` + SB3 SAC pattern, adapted for push with cuRobo IK:

```
SB3 SAC agent (4D Box action)
  → env.step(action)  ←── one push macro-action (72 decorrelated substeps)
    → _pre_physics_step(): decode (Xs, Ys, length, theta), build 72 waypoints
    → _apply_action() × 72: cuRobo solve_batch(N_envs) → joint targets → physics step
    → _get_dones(): max_pushes, at_goal, launched, tipped, OOB
    → _get_rewards(): dense fractional improvement + completion bonus + tip penalty
    → _get_observations(): dict {observation, achieved_goal, desired_goal} for HER
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| `DirectRLEnv` not `ManagerBasedRLEnv` | Eliminates 792 wasted manager calls per push (72 substeps × 11 manager pipelines, all outputs discarded). `throwing_enviroment` runs 4096 envs with this pattern. |
| `decimation=72` | One env step = one complete push across all 72 waypoint substeps. No external training loop orchestration. |
| cuRobo IK in `_apply_action()` | IK solver is an env member, warm-up once with `N=num_envs`. Each `_apply_action()` call advances one waypoint via `solve_batch` seeded from `prev_joint_cmd`. |
| Dict observations for HER | `_get_observations()` returns `{"observation": obs22D, "achieved_goal": [obj_xy, obj_yaw], "desired_goal": [goal_xy, goal_yaw]}`. SB3's `HerReplayBuffer` uses `goal_selection_strategy="future"` with `n_sampled_goal=4`. |
| Continuous action space | 4D `Box(-1,1)` decoded via `action_push_continuous.py` → `(Xs,Ys,len,theta)`. Object-relative decode (`decode_push_action_relative_continuous`) available via `--rel-act`. |
| `DirectRLVecEnv` wrapper | Custom `VecEnv` subclass (128 lines, copied from `throwing_enviroment/tasks/sb3_vec_env.py`). Wraps one `DirectRLEnv` instance that handles N envs internally. Converts numpy ↔ torch. |
| SB3 `model.learn()` delegation | No custom training loop. SB3 handles replay buffer, SAC gradient updates, logging. `LatestCheckpointCallback` + SIGTERM handler for HPC preemption. |

### 9.3 Files

| File | Lines | Purpose |
|------|-------|---------|
| `tasks/push_direct_env_cfg.py` | 240 | `PushDirectEnvCfg(DirectRLEnvCfg)` — scene, MDP, reward, IK config |
| `tasks/push_direct_env.py` | 553 | `PushDirectEnv(DirectRLEnv)` — `_apply_action` IK state machine, obs/reward/done methods, `compute_reward` for HER |
| `tasks/sb3_vec_env.py` | 128 | `DirectRLVecEnv(VecEnv)` — SB3 compatibility wrapper |
| `tasks/utils/action_push_continuous.py` | 78 | `decode_push_action_continuous()`, `decode_push_action_relative_continuous()` — Box action → push params |
| `train_push_sac_her.py` | 182 | Training launcher: `PushDirectEnv` → `DirectRLVecEnv` → `SAC(MultiInputPolicy, ...)` with `HerReplayBuffer` |
| `hpc/train_push_sac_her.slurm` | 155 | SLURM job: 528 envs, 3000 iters, 4h, auto-resume chain |
| `tests/validate_push_sac.py` | 579 | Validation: loads SB3 SAC checkpoint, runs against 20 test scenes |
| `tests/record_push_video.py` | 538 | Video recorder (reuses PPO checkpoints, same physics pathway) |

### 9.4 Reward Function

Fractional improvement reward (same formula as Push-PPO Fix P63, with adjusted scales):

```
d_prev = ‖obj_xy_prev − goal_xy‖,   d_now = ‖obj_xy_now − goal_xy‖
y_prev = |yaw_prev − goal_yaw|,      y_now = |yaw_now − goal_yaw|

pos_imp  = α · (d_prev − d_now) / d_prev     (α=1.0, clamped ±5.0)
rot_imp  = α · (y_prev − y_now) / y_prev     (α=1.0, clamped ±4.0)
penalty  = −β · d_now                          (β=0.5, clamped [−2.0, 0])
rot_pen  = −β_rot · y_now                      (β_rot=0.25, clamped [−1.0, 0])

r_dense  = pos_imp + rot_imp + penalty + rot_pen
r_sparse = +2.0  if pos_err < 0.05                (completion bonus)
         = +2.0  if pos_err < 0.05 AND rot_err < 0.2  (rotation sub-bonus)
         = −5.0  if tipped (|roll|>0.3 or |pitch|>0.3)
```

**HER compatibility note**: `compute_reward(achieved_goal, desired_goal, infos)` is
implemented on `PushDirectEnv` and delegated from `DirectRLVecEnv`.  It recomputes
the dense fractional improvement using `prev_achieved_goal` from `info` dict,
supporting SB3's hindsight relabeling with sparse + dense reward terms.

### 9.5 SAC Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Standard SB3 default |
| `buffer_size` | 100,000 | Sufficient for 5-step episodes (100K transitions ≈ 20K episodes at 5 pushes each) |
| `batch_size` | 256 | Standard SB3 default |
| `tau` | 0.005 | Polyak averaging for target networks |
| `gamma` | 0.95 | Matches PPO baselines; 1/(1−0.95)=20-step effective horizon vs 5-step episodes |
| `ent_coef` | `"auto"` | SAC auto-tunes entropy coefficient |
| `net_arch` | `[256, 256]` | MLP with ReLU activations |
| HER `n_sampled_goal` | 4 | Standard HER setting |
| HER `goal_selection_strategy` | `"future"` | Relabels with goals from later steps in same episode |

### 9.6 Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| **Q-function divergence at gamma=0.99** | **Fixed (Fix S1)** | `completion_bonus=5.0` + `dense_alpha=3.0` + `gamma=0.99` caused critic loss 56.5→1130 and actor loss −2.53→79.3 within 626K steps. Reduced to `completion_bonus=2.0`, `dense_alpha=1.0`, `gamma=0.95`. |
| **TensorBoard async writer crash** | **Fixed (Fix S2)** | SB3's `SummaryWriter` background thread lost file descriptor to TMPDIR-bind-mounted events file → `FileNotFoundError` → killed `model.learn()`. Disabled via `tensorboard_log=None`. |
| **`push_count` in wrong lifecycle method** | Open | Incremented in `_get_rewards()` which runs *after* `_get_dones()` in `DirectRLEnv.step()`. Practically gives 1 extra push before max-push termination (5 pushes instead of 4+done). Minor — does not affect learning. |
| **`observation_space=22` in config is misleading** | Open | Config declares `int` but env returns `Dict`. `DirectRLVecEnv` auto-detects dict from `_get_observations()` output, so training works. Cosmetic. |
| **Cannot run smoke test locally without container** | Open | Isaac Lab requires Isaac Sim container. Local testing needs `docker pull nvcr.io/nvidia/isaac-lab:2.3.0` + `pip install curobo==0.7.5`. |

### 9.7 Observation Layout

Flat observation for the policy (22D with `rel_obs=True`):

```
[ee_pos(3) | ee_euler(3) | obj_pos(3) | obj_euler(3) | obj_linvel(3) |
 obj_angvel(3) | dist_to_ee(1) | contact(1) | rel_dx(1) | rel_dy(1)] = 22D
```

HER goal tensors (3D each):

```
achieved_goal = [obj_x, obj_y, obj_yaw]        ← actually achieved
desired_goal  = [goal_x, goal_y, goal_yaw]     ← from sampled goal
```

---

## 10. gym-pusht Controlled Testbed (Models A/B/C) <a name="gym-testbed"></a>

**Branch**: `asp_goal_encoder`  
**Last updated**: 2026-06-27  

Native gym-pusht counterparts of PBRS Models A/B/C to get a **fast, controlled CPU testbed**
that isolates the **reward/curriculum question** from the Isaac robotic-task confounds
(compute mismatch between single-agent and ASP runs, IK gating, contact physics).
All three models reuse the **identical custom PPO/PPOABC/ActorCriticPush
+ EpisodeManager + validate_goal + reward_pbrs** as the Isaac models; only the
environment (gym-pusht 2D point-pusher) differs.

### 10.1 Rationale

The Isaac runs are the **headline** evidence (robotic UR5e, cuRobo IK, contact-rich).
But the Isaac A-vs-ASP comparison is confounded: ASP changed 3 things at once
(goal distribution, budget, agent count; critique **C6**), and the single-agent
checkpoints were compute-mismatched vs the ASP runs.  The gym testbed fixes this
by running A, B, and C in **one identical, fast environment at one identical budget**,
reusing the same learner — only the curriculum (none vs forced vs ASP) differs.

A/B answer the **reward/curriculum** question cleanly in hours.  
ASP (C) is single-process on CPU (~22 push/s) and needs days for the same
experience; the ASP evidence at scale comes from the **Isaac runs** (GPU-batched,
identical compute to the single-agent).  This env split is principled and
strengthens critique **C2** (scale) and **C9** (overhead).

### 10.2 Files (new, no Isaac/cuRobo dependency)

| File | Role |
|---|---|
| `tasks/utils/gym_push_primitive_env.py` | Smart `gym.Env` for A/B: 1-step push + PBRS + done. `TorchVecAdapter` wraps `AsyncVectorEnv`. |
| `tasks/utils/gym_push_asp_env.py` | Single-process ASP env for C (reuses `EpisodeManager` + `validate_goal` + PBRS). |
| `train_a_gym_pbrs_simple.py` | Model A-gym (single-agent PBRS, no curriculum). |
| `train_b_gym_pbrs_curriculum.py` | Model B-gym (PBRS + P82 pos→rot curriculum via `set_curriculum`). |
| `train_c_gym_pbrs_asp.py` | Model C-gym (ASP: Alice PPO + Bob PPOABC/GoalEncoder/ABC/historical pool). |
| `hpc/train_gym_a.slurm` | SLURM for gym-A on HPC (CPU partition). |
| `hpc/train_gym_b.slurm` | SLURM for gym-B on HPC. |
| `hpc/train_gym_c.slurm` | SLURM for gym-C on HPC (single-process, slow). |

### 10.3 Smart gym.Env design (A/B)

`GymPushPrimitiveEnv(gym.Env)` — one `step(action_bins)` = one macro push:

1. Decode 4D×21 bins → object-relative push `(r, φ, len, θ)` → (Xs,Ys,Xf,Yf) in meters.
2. Execute the push: `APPROACH_STEPS=40` + `PUSH_STEPS=60` PD-control steps on the underlying gym‑pusht `PushTEnv`.
3. Build the 30‑D observation (`rel_obs` layout: `[ee(6)|obj(14)|goal(6)|goal_dist(2)|rel_goal(2)]`).
4. Compute **PBRS reward** (`compute_pbrs_reward`) + `check_done_pbrs` with `terminated=False` always.
5. **Self‑reset on done** — sample new block+goal, return the reset observation.
6. `done`, `pos_err`, `cos_rot_err`, `at_goal`, `pos_only`, `success` all go into `info`.

The env **never signals `terminated`/`truncated` to the vector wrapper**, so
`AsyncVectorEnv(autoreset_mode=DISABLED)` does not interfere.  Curriculum B
calls `env.set_curriculum(w_rot, pos_term, enable_rot_sparse)` each iteration
via `AsyncVectorEnv.call`; Model A runs with defaults (w_rot=10, no curriculum).

**TorchVecAdapter** wraps `gym.vector.AsyncVectorEnv` (`context="spawn"`,
`autoreset=DISABLED`), numpy↔torch, exposes the custom‑PPO `vec_env` tensor contract.

### 10.4 ASP env design (C)

`GymPushASPEnv` is **single-process synchronous** — the Alice↔Bob cross‑phase delayed
reward forces central ASP orchestration on a batched `EpisodeManager(num_envs=N)`.
It ports `PushASPEnvWrapper`'s methods to gym‑pusht physics:

- Object read/write → pymunk `block.position/angle` + `env.reset(reset_to_state)`.
- Observation build → 28‑D (Alice 20‑D, Bob 28‑D, same as Isaac PBRS-C).
- Reuses `EpisodeManager` + `validate_goal` + `PBRS` verbatim (env‑agnostic).
- `execute_push(Xs,Ys,len,θ)` loops N pymunk envs serially each push‑step.

**Training script** (`train_c_gym_pbrs_asp.py`) mirrors `train_c_pbrs_asp.py`
with the cuRobo/IK inner loop replaced by `env.execute_push(...)`.  All ASP
machinery — Alice/Bob phase routing, delayed reward attribution to Alice's
last valid transition, ABC buffer, historical pool, two separate PPO updates —
are copied faithfully from `train_c_pbrs_asp.py`.

### 10.5 Freeze fix + safe benchmark

**Root cause:** a debug benchmark script lacked `if __name__ == "__main__":`,
so `spawn` workers re‑imported `__main__` and recursively created a `TorchVecAdapter`
→ **recursive process spawning** (a spawn fork‑bomb).  Compounded by `fork` +
torch's 6‑threads‑per‑process default (32 workers × 6 ≈ 192 threads on 12 logical
CPUs) → **system freeze**.  The training scripts are unaffected (guard is in place).

**Fix:** added thread caps to all gym scripts + the env module:
```
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
...
torch.set_num_threads(1)
```

**Benchmark** (Ryzen 5 5600X, 6c/12t, 31 GB RAM):

| Config | push-macros/s | time to 1M pushes |
|---|---|---|
| gym AsyncVectorEnv N=6 (A/B) | 91 | ~3.0 h |
| gym single-process ASP (C) | 22.4 | ~12.4 h |

### 10.6 Diagnostic findings

**Oracle‑push probe** (approach behind the object centroid along the obj→goal
axis, push toward goal): the object moves **+0.23 m goal‑ward at every approach
radius** r ∈ {0.03…0.16 m} — the push primitive works fine; contact is not the issue.

**Random‑policy probe:** mean per‑push goal‑ward displacement = **+0.002 m**
(pure noise), mean |displacement| = 0.07 m.

**Trained checkpoint eval** (A‑gym, iter 1916, N=6, push_nsteps=15, 144k pushes):
per‑push goal‑ward = **+0.029 m → 15× the random baseline.**  The policy IS
learning, but **very slowly** because:
- N=6 gives a tiny PPO batch (6×15=90 samples/update vs Isaac's 7920) → noisy updates.
- Only 144k pushes (~1% of Isaac's 12M‑push stabilization budget at k_p=30).
- Faithful k_p=30 makes Φ essentially flat beyond ~0.25 m, so dense gradient is
  weak until the object gets close — requiring many pushes to bootstrap.

**Action:** added `--push_nsteps` arg (bigger PPO batch without more processes)
and moved A/B/C to HPC (many CPU cores → PPO batch scales → faithful k_p=30
learns in hours).

### 10.7 HPC setup (one-time, Habrók login node)

Build a writable overlay with the extra Python deps gym needs (the isaac‑lab.sif
container already has torch + numpy + yaml + tensorboard):

```bash
cd $PROJECT_ROOT                         # asyncDualPlayPPO (where isaac-lab.sif lives)

# NOTE: the container exposes python ONLY via /workspace/isaaclab/isaaclab.sh -p
#       (bare `python`/`pip` are NOT on $PATH in the container).

# 1. Create the overlay — let it FINISH (~20-30s zeroing; do NOT Ctrl-C):
apptainer overlay create --size 2048 gym_overlay.img

# 2. Install gym deps into the overlay. PIN pymunk==6.11.0 — pymunk 7.x removed
#    space.add_collision_handler(), which gym-pusht's _setup() calls (crash otherwise):
apptainer exec --overlay gym_overlay.img:rw isaac-lab.sif \
  /workspace/isaaclab/isaaclab.sh -p -m pip install \
  'pymunk==6.11.0' pygame shapely opencv-python-headless

# 3. gym-pusht (modified arena_width) is provided as the BOUND SOURCE via PYTHONPATH
#    (the slurm sets PYTHONPATH=$CROOT/gym-pusht) — no pip install of gym_pusht needed.
#    Just clone/rsync master_isaac/gym-pusht onto HPC at $PROJECT_ROOT/../gym-pusht.

# 4. Verify — construct + RESET the env (the crash site that needs pymunk 6.x):
apptainer exec --overlay gym_overlay.img:ro \
  --bind "$PROJECT_ROOT/../gym-pusht":/ws/gp --env PYTHONPATH=/ws/gp \
  isaac-lab.sif /workspace/isaaclab/isaaclab.sh -p -c \
  "import pymunk; print('pymunk', pymunk.version); from gym_pusht.envs.pusht import PushTEnv; e=PushTEnv(obs_type='state'); e.reset(); print('reset OK')"
```

**Run** (once `gym_overlay.img` + `gym-pusht` are in place):
```bash
sbatch hpc/train_gym_a.slurm      # Model A (single-agent)
sbatch hpc/train_gym_b.slurm      # Model B (curriculum)
sbatch hpc/train_gym_c.slurm      # Model C (ASP; slow, many resubmit chains)
```

Each SLURM script uses `gym_overlay.img`, binds the modified `gym-pusht` source,
sets `SDL_VIDEODRIVER=dummy`, and calls `isaaclab.sh -p train_*_gym.py` (the
container's proven Python with torch).  Time limits 12h per job + auto‑resubmit
on `SIGUSR1` for long runs.  Checkpoints under `runs/{exp_name}/` are rsynced
to `PROJECT_ROOT/runs` at cleanup.

### 10.8 Measured Isaac-vs-gym throughput (controlled)

| Sim (hardware) | parallelism | push‑macros/s | **1M pushes** | batch/update | ASP overhead vs A |
|---|---|---|---|---|---|
| **Isaac Lab** (1 GPU, 528 env) | GPU‑batched | **172** | **~1.6 h** | 7920 | **~1.0× (none)** |
| gym‑pusht (desktop, 6 CPU, A/B) | 6 CPU procs | 91 | ~3.0 h | 90–360 | — |
| gym‑pusht ASP (single‑proc) | 1 CPU proc | 22.4 | ~12.4 h | ~480 | ~7.7× *slower* |

**Key findings:**
- Isaac single‑agent A, curriculum B, and ASP C all run at **~0.022 it/s (~172 push/s)** — the 2‑agent ASP machinery adds **~no per‑iteration wall‑clock** (the shared 528‑env cuRobo‑IK/physics dominates). So the Isaac A‑vs‑ASP comparison is genuinely **compute‑matched** (same hardware, same it/s, same 528 envs).
- Isaac reaches ASP‑scale experience (~10M pushes) in ~16h (batch 7920); gym single‑process ASP in ~124h (~5 days).  ASP parallelises for free on GPU‑batched sim but is forced single‑process on CPU.
- **Conclusion:** Isaac Lab is the appropriate environment to evaluate ASP — it gives ASP its best shot on a single‑GPU budget (identical compute to the winning single‑agent), providing the large batch the non‑stationary two‑agent objective needs (Makoviychuk et al. 2021; Rudin et al. 2022). That ASP still collapses to 0–7% there is strong evidence the failure is **structural, not under‑resourcing.**  The `5–10× overhead` claim on Slide 11 refers to the **ManagerBasedRLEnv‑vs‑DirectRLEnv** API overhead (a separate, valid claim; not refuted by this data).
- Resolves supervisor critique **C2**: ~1.6 days / 3000 iters, not "a few hours."  The deck should show a **new slide** ("Why Isaac Lab is a good sim for robotic tasks") with the 3 pillars — GPU‑batched parallelism, robotic fidelity, time‑to‑scale for ASP — backed by this measured table. See `presentation_plan.md` §4 for the narrative revision.

---

## 11. Validation Results (26.06.26–28) <a name="validation-results"></a>

### 11.1 Definitive Head-to-Head Comparison

All 7 Isaac models evaluated on **identical 30 T-block scenes** using the current
`validation_configs.py` (tests 1–10: R_* rotation-heavy pos_rot, tests 11–20: pos_only,
tests 21–30: pos_rot). Success gate: `pos_err < 0.05 m AND rot_err < 0.2 rad` (thesis gate).
Best-of-20 trials per scene, max 30 pushes.

| Model | Scene SR | Pos-only | Pos+rot | PosErr | RotErr | Avg Pushes |
|-------|----------|----------|---------|--------|--------|-----------|
| **A_simp** (no curriculum) | **80.0%** | 100% | 70% | 0.032 m | 0.568 rad | 23.5 |
| **B_curr** (P82 curriculum) | **76.7%** | 100% | 65% | 0.023 m | 0.663 rad | 26.7 |
| G_tasp_dpose (TASP + d_pose) | 16.7% | 30% | 10% | 0.158 m | 1.457 rad | 12.2 |
| H_tasp_disc (TASP + disc) | 10.0% | 30% | 0% | 0.186 m | 1.260 rad | 12.6 |
| E_asp_dpose (ASP + d_pose) | 6.7% | 20% | 0% | 0.143 m | 1.612 rad | 12.8 |
| F_asp_disc (ASP + disc) | 6.7% | 20% | 0% | 0.197 m | 1.576 rad | 11.9 |

**Key findings:**

1. **Single-agent PBRS (A_simp) beats the P82-fixed curriculum (B_curr) by 3.3pp scene SR.**
   Both achieve 100% pos-only SR. The curriculum improves position precision (0.023m vs
   0.032m PosErr) but slightly degrades rotation (0.663 vs 0.568 rad RotErr). The simpler
   model wins — curriculum adds staging complexity without net gain.

2. **ASP collapses across all variants.** The best ASP model (G_tasp_dpose) reaches 16.7% —
   a 4.8× gap from single-agent. Time-based Alice (G/H) outperforms outcome-based Alice
   (E/F) by 2-10pp. T-block (G) outperforms disc (H) by 6.7pp. All ASP models terminate
   early (~12 pushes vs 23–27 for single-agent models) due to catastrophe detection.

3. **Position-only is solved (100% for A and B).** The combined pos+rot gate is the
   bottleneck — capping at 70% (A) / 65% (B). Rotation remains the unsolved dimension
   across all models.

4. **The P82 curriculum trigger fix produced a functional model.** B_curr went from 35% SR
   (26.06.12, mis-specified trigger) to 76.7% (26.06.28, episodic pos-SR trigger). The
   curriculum now activates and progresses through Phase 1→2, but the resulting model does
   not beat the simpler no-curriculum baseline.

### 11.2 Checkpoints Used

| Model | Date | Checkpoint | Iter | Validated |
|-------|------|-----------|------|-----------|
| A_simp | 26.06.20 | `ppo_pbrs_reward/26.06.20/runs/hpc_pbrs_simp_528env/agent/latest_checkpoint.pt` | 2400 | 2026-06-28 |
| B_curr | 26.06.26 | `hpc_pbrs_curr_528env_fixed/agent/model_best.pt` | 2600 | 2026-06-28 |
| E_asp_dpose | 26.06.26 | `hpc_pbrs_asp_dpose_528env/bob/model_best.pt` | — | 2026-06-26 |
| F_asp_disc | 26.06.26 | `hpc_pbrs_asp_disc_528env/bob/model_best.pt` | — | 2026-06-26 |
| G_tasp_dpose | 26.06.26 | `hpc_pbrs_tasp_dpose_528env/bob/model_best.pt` | — | 2026-06-26 |
| H_tasp_disc | 26.06.26 | `hpc_pbrs_tasp_disc_528env/bob/model_best.pt` | — | 2026-06-26 |

### 11.3 Validation Config Compatibility

The current `validation_configs.py` (post-2026-06-26) has tests 1–10 as T-block R_*
rotation scenes (`pos_rot`). The 26.06.20 validation used an earlier version where
tests 1–10 were D_* disc scenes (`disc_pos`, `object_type=disc`). The 26.06.26/28
campaign uses the current config exclusively. Legacy disc-protocol CSVs are preserved
in `results/legacy/`.

Two validation protocols exist:
- **Isaac** (thesis gate): `pos_err < 0.05m AND rot_err < 0.2 rad` — `success` column
  computed by `validate_push.py`/`validate_push_asp.py`
- **gym-pusht** (coverage gate): `coverage >= 0.95` — `success` column computed by
  `validate_pusht_gym.py`

SR values between protocols are not directly comparable; raw error metrics (PosErr,
RotErr) are.

---

## 12. Results Directory Structure <a name="results-directory"></a>

All validation results consolidated to `/home/vladi/IsaacLab/master_isaac/results/`:

```
results/
  SUMMARY.md              — Aggregate comparison table + findings
  A_simp/                 — PBRS single-agent (all dates/protocols)
    20_isaac_30t.csv      — Head-to-head: 26.06.20 ckpt on current 30 T-block
    12_isaac_20t.csv      — 26.06.12: old 20 T-block scenes
    20_disc_isaac.csv     — 26.06.20: old disc scenes (tests 1–10 = disc)
    20_gympusht.csv       — 26.06.20: gym-pusht
    26_gympusht.csv       — 26.06.26: gym-pusht
    gym_gympusht.csv      — gym-pusht trained variant
  B_curr/                 — PBRS + curriculum
    28_isaac_30t.csv      — Head-to-head: 26.06.28 ckpt on current 30 T-block
    12_isaac_20t.csv      — 26.06.12: old 20 T-block (broken trigger)
    26_gympusht.csv       — 26.06.26: gym-pusht
  C_asp/                  — PBRS + ASP (Alice/Bob)
    gympusht.csv          — gym-pusht: 0% SR, avg 8.9 pushes
    12_isaac_20t.csv      — 26.06.12: old 20 T-block
  E_asp_dpose/            — ASP + SE(2) d_pose (T-block)
    26_isaac.csv          — 26.06.26: 6.7% SR
    20_isaac.csv          — 26.06.20: 0% SR (early checkpoint, it 2600)
    gympusht.csv          — gym-pusht
  F_asp_disc/             — ASP + d_pose (disc)
    26_isaac.csv          — 26.06.26: 6.7% SR
    20_isaac.csv          — 26.06.20: 0% SR (early checkpoint, it 2400)
    gympusht.csv          — gym-pusht
  G_tasp_dpose/           — TASP + d_pose (T-block)
    26_isaac.csv          — 26.06.26: 16.7% SR — best ASP
    20_isaac.csv          — 26.06.20: 0% SR (early checkpoint, it 1200)
    gympusht.csv          — gym-pusht
  H_tasp_disc/            — TASP + d_pose (disc)
    26_isaac.csv          — 26.06.26: 10.0% SR
    20_isaac.csv          — 26.06.20: 0% SR (early checkpoint, it 1200)
    gympusht.csv          — gym-pusht
  comparison/             — Cross-model comparison plots + per_test_comparison.txt
```

**Naming convention:** `{date-short}_{protocol}_{scenes-short}.csv` — e.g.
`28_isaac_30t.csv` = 26.06.28, Isaac protocol, 30 T-block scenes.

Models **not evaluated**: D (ASP-no-GE, excluded), SAC (training incomplete).