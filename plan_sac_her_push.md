# Plan: SAC + HER Push Baseline (DirectRLEnv)

## Rationale

The current push PPO baselines use `ManagerBasedRLEnv` with cuRobo IK executed in an external training loop. Every push macro-action (72 substeps) calls `ManagerBasedRLEnv.step()` 72 times, each running the full manager pipeline (action/recorder/termination/reward/observation/command/event managers). None of those outputs are used — the wrapper discards them. ~792 wasted manager calls per push.

The `Isaaclab-TableTennisRobot` and `throwing_enviroment` projects both prove that `DirectRLEnv` eliminates this overhead by collapsing multi-substep macro-actions into a single `step()` call with internal physics substeps, and delegates training entirely to Stable-Baselines3's `model.learn()`.

**Goal**: A fast SAC + HER push agent that reuses the validated `throwing_enviroment` architecture, adding cuRobo IK, multi-step episodes, and goal-conditioned rewards with hindsight relabeling.

---

## Architecture reference: throwing_enviroment

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| Config | `tasks/throwing_direct_env_cfg.py` | 223 | `DirectRLEnvCfg`, decimation=320, action_space=4, observation_space=10 |
| Env | `tasks/throwing_direct_env.py` | 354 | `ThrowingDirectEnv(DirectRLEnv)` |
| SB3 wrapper | `tasks/sb3_vec_env.py` | 76 | `DirectRLVecEnv(VecEnv)` — wraps one DirectRLEnv for SB3 |
| Training | `scripts/train_sac.py` | 175 | `SAC("MlpPolicy")` → `model.learn()`, `LatestCheckpointCallback` |
| SLURM | `hpc/train_sac_direct.slurm` | 180 | 4096 envs, 4h, auto-resume chain, SIGTERM forwarding |

### How throwing_environment.step() works

```
SAC agent outputs 4D action (Box, [-1,1])
  → _pre_physics_step(): store, decode, build joint target sequence
  → _apply_action() state machine across 320 substeps:
      PHASE_STABILIZE (10)  → hold crane pose, drink kinematically attached
      PHASE_GO_TO_INIT (20) → move to init joint pose
      PHASE_GO_TO_INITIAL (20) → move to wind-up pose (agent param 0)
      PHASE_THROW (≤120)    → linear joint interpolation init→end, release at release_time
      PHASE_FLIGHT (150)    → hold end pose, object flies free
  → _get_dones()   → always terminated (one-shot episode)
  → _get_rewards() → dual-Gaussian: 0.9*exp(-d²/0.1) + 0.1*exp(-d²/0.5), +1 at d<0.15
  → _get_observations() → [robot_indicator, target_xyz, obj_xyz, dist, dist_xy] / 3.0
  → _reset_idx()   → randomize target, reset robot, reset state machine
```

### Key decisions proven by throwing_environment

1. **SB3 `model.learn()` works with macro-action DirectRLEnv** — no custom training loop needed
2. **Custom `DirectRLVecEnv(VecEnv)` is simpler than `Sb3VecEnvWrapper`** for DirectRLEnv
3. **`LatestCheckpointCallback` + SIGTERM handler** reliably saves/reloads replay buffer across SLURM job chains
4. **4096 envs at 320 substeps per step** runs fine on RTX Pro 6000 (96 GB VRAM)

---

## Files to create (5 new, 0 modified)

```
asyncDualPlayPPO/
├── tasks/
│   ├── sb3_vec_env.py              # NEW: copy from throwing_enviroment/tasks/sb3_vec_env.py
│   ├── push_direct_env_cfg.py      # NEW: DirectRLEnvCfg for push
│   └── push_direct_env.py          # NEW: PushDirectEnv(DirectRLEnv)
├── train_push_sac_her.py           # NEW: SAC+HER training script
└── hpc/
    └── train_push_sac_her.slurm    # NEW: SLURM job
```

### What stays unchanged

| File | Why |
|------|-----|
| `action_push.py` | `compute_push_waypoints()` reused inside `_apply_action()` |
| `action_push_relative.py` | `decode_push_action_relative()` for object-relative actions |
| `reward_pbrs.py` | PBRS reward functions (optional, for reward port) |
| `events.py` | `reset_objects_to_random_safe_pose()`, `reset_robot_joints()` — minor signature tweak |
| `push_primitive_1arm_env.py` | Asset configs (`UR5e_CFG`, table dimensions, zone borders) |
| `goal_validator.py` | `validate_goal()` — pure tensor function |

---

## File 1: `tasks/sb3_vec_env.py` (~80 lines)

Copy from `throwing_enviroment/tasks/sb3_vec_env.py` with zero changes.

```python
class DirectRLVecEnv(VecEnv):
    """Wraps a single DirectRLEnv (which handles N envs internally) for SB3."""
    def __init__(self, env: DirectRLEnv):
        self.env = env
        self.num_envs = env.num_envs
        observation_space = gymnasium.spaces.Box(...)  # from env.single_observation_space
        action_space = gymnasium.spaces.Box(...)        # from env.single_action_space
        super().__init__(self.num_envs, observation_space, action_space)

    def step_async(self, actions: np.ndarray):    # store
    def step_wait(self):                           # numpy→torch→env.step()→numpy
    def reset(self):                               # env.reset() → numpy
    def close(self):
    def get_attr(self, attr_name, indices=None):
    def set_attr(self, attr_name, value, indices=None):
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
```

---

## File 2: `tasks/push_direct_env_cfg.py` (~120 lines)

```python
@configclass
class PushDirectEnvCfg(DirectRLEnvCfg):
    decimation: int = 72          # one push = one env step, internally 72 physics ticks
    episode_length_s: float = 10.0    # safety timeout (5 pushes × ~2s each)

    action_space: int = 4        # continuous Box(4): Xs, Ys, length, theta
    observation_space: int = 28  # [ee_pose(6) | obj_state(14) | goal_pose(6) | goal_dist(2)]
    state_space: int = 0

    num_objects: int = 1
    max_pushes_per_episode: int = 5

    # RL algorithm hint for gym registration
    rl_task_config = {"agent_cfg_entry_point": None}

    # Reward parameters
    reward_type: str = "dense"       # "dense" or "pbrs"
    dense_alpha: float = 3.0         # fractional improvement scale
    push_success_threshold_pos: float = 0.05
    push_success_threshold_rot: float = 0.2

    # Workspace
    ws_x: tuple = (-0.50, 0.50)
    ws_y: tuple = (0.25, 0.70)
    ws_z: tuple = (0.25, 0.55)

    # Spawn / goal ranges (local frame)
    spawn_x_range: tuple = (-0.40, 0.40)
    spawn_y_range: tuple = (0.30, 0.70)
    goal_x_range: tuple = (-0.40, 0.40)
    goal_y_range: tuple = (0.30, 0.70)
    goal_min_dist: float = 0.05
    goal_max_dist: float = 0.45

    # Object
    object_spawn_z: float = 0.05
    object_type: str = "t_shape"   # "t_shape" or "cylinder" (disc)

    # Robot
    robot_usd_path: str = ...
    robot_arm_joint_names: list = [...]

    sim: SimulationCfg = SimulationCfg(
        dt=0.02,
        render_interval=decimation,
        use_fabric=True,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=1024 * 1024,
            gpu_max_rigid_contact_count=1024 * 1024,
            gpu_max_rigid_patch_count=81920 * 4,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True,
    )
```

---

## File 3: `tasks/push_direct_env.py` (~500 lines)

Heavily modelled on `throwing_direct_env.py`. Key methods:

### `__init__`

```python
class PushDirectEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, render_mode)
        # Pre-allocate all per-env tensors
        self.push_count = torch.zeros(self.num_envs, dtype=torch.long, ...)
        self.goal_pos_euler = torch.zeros(self.num_envs, 6, ...)  # local frame
        self.prev_obj_pos = torch.zeros(self.num_envs, 3, ...)
        self.prev_obj_euler = torch.zeros(self.num_envs, 3, ...)
        self.prev_phi_pos = torch.zeros(self.num_envs, ...)       # for PBRS
        self.prev_phi_rot = torch.zeros(self.num_envs, ...)
        self.gave_completion = torch.zeros(self.num_envs, dtype=torch.bool, ...)
        self.gave_rot_bonus = torch.zeros(self.num_envs, dtype=torch.bool, ...)
        self.at_goal = torch.zeros(self.num_envs, dtype=torch.bool, ...)
        self.ep_start_pos = torch.zeros(self.num_envs, 3, ...)
        self.ep_start_euler = torch.zeros(self.num_envs, 3, ...)
        # IK state-machine tensors
        self._step_idx = torch.zeros(self.num_envs, dtype=torch.int, ...)
        self._waypoints_pos = torch.zeros(self.num_envs, 72, 3, ...)
        self._waypoints_quat = torch.zeros(self.num_envs, 72, 4, ...)
        self._joint_cmd = torch.zeros(self.num_envs, 6, ...)  # prev joint cmd for IK seed
        # cuRobo IK solver init
        self._init_ik_solver()
```

### `_setup_scene`

```python
def _setup_scene(self):
    # Create assets
    self._robot = Articulation(self.cfg.robot_cfg)
    self._table = RigidObject(self.cfg.table_cfg)
    self._target_object = RigidObject(self.cfg.object_cfg)
    # Register
    self.scene.articulations["robot"] = self._robot
    self.scene.rigid_objects["table"] = self._table
    self.scene.rigid_objects["target_object"] = self._target_object
    # Ground, lights
    spawn_ground_plane(...)
    # Clone for N envs
    self.scene.clone_environments(copy_from_source=False)
    # Goal ghost as VisualizationMarkers (no physics)
    self._goal_viz = VisualizationMarkers(...)
    # Resolve body/joint indices
    self._arm_jids, _ = self._robot.find_joints(self.cfg.robot_arm_joint_names)
    _, self._ee_body_idx = self._robot.find_bodies("wrist_3_link")
```

### `_pre_physics_step`

```python
def _pre_physics_step(self, actions):
    self.actions = actions.clone()
    # Decode 4D Box action → (Xs, Ys, length, theta) with workspace clamping
    # Uses decode_push_action_relative() if relative mode, else absolute decode
    # Build waypoints for all envs: compute_push_waypoints(...)
    self._waypoints_pos[:, :, :] = ...
    self._waypoints_quat[:, :, :] = ...
    self._step_idx.zero_()
    # Snapshot pre-push object position
    obj_root = self._target_object.data.root_pos_w - self.scene.env_origins
    self.prev_obj_pos[:] = obj_root
    self.prev_obj_euler[:] = _quat_to_euler_xyz(...)
```

### `_apply_action` (the critical method)

```python
def _apply_action(self):
    step = self._step_idx
    wp_pos = self._waypoints_pos[torch.arange(self.num_envs), step]
    wp_quat = self._waypoints_quat[torch.arange(self.num_envs), step]
    # cuRobo IK solve batch
    result = self.ik_solver.solve_batch(
        CuroboPose(position=wp_pos - self._IK_ERROR, quaternion=wp_quat),
        seed_config=self._joint_cmd.unsqueeze(1),
        retract_config=self._joint_cmd,
    )
    solved = result.solution.view(self.num_envs, 6)
    ik_ok = result.success.squeeze(-1)
    elbow_bad = solved[:, 2] < 0.0
    ik_ok[elbow_bad] = False
    joints = torch.where(ik_ok.unsqueeze(-1), solved, self._joint_cmd)
    self._joint_cmd = joints.detach().clone()
    # Apply
    self._robot.set_joint_position_target(joints, joint_ids=self._arm_jids)
    self._robot.write_joint_position_to_sim(
        self._robot.data.joint_pos[:, self._arm_jids], joint_ids=self._arm_jids)
    # Gripper always closed
    self._gripper_cmd[:] = -1.0   # close
    self._step_idx += 1
```

**Note**: `_apply_action()` is called once per substep (decimation=72 times per env step). Each call advances one waypoint via IK solve. The state machine (`_step_idx` tracking across substeps) replaces the external training loop's `for wp_idx, (wp_pos, wp_quat, ...) in enumerate(waypoints)` pattern.

### `_get_observations`

Direct tensor reads — NO observation manager:

```python
def _get_observations(self) -> dict:
    ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
    ee_euler = _quat_to_euler_xyz(...)
    obj_pos = self._target_object.data.root_pos_w[:, :3] - self.scene.env_origins
    obj_euler = _quat_to_euler_xyz(self._target_object.data.root_quat_w)
    obj_linvel = self._target_object.data.root_lin_vel_w
    obj_angvel = self._target_object.data.root_ang_vel_w
    dist_to_ee = torch.norm(obj_pos - ee_pos, dim=-1, keepdim=True)
    contact = ...  # from contact sensor or distance threshold
    goal_xyz = self.goal_pos_euler[:, :3]
    goal_euler = self.goal_pos_euler[:, 3:6]
    pos_dist = torch.norm(obj_pos - goal_xyz, dim=-1, keepdim=True)
    rot_dist = _yaw_distance(obj_euler[:, 2], goal_euler[:, 2])
    obs = torch.cat([ee_pos(3), ee_euler(3), obj_pos(3), obj_euler(3),
                     obj_linvel(3), obj_angvel(3), dist_to_ee(1), contact(1),
                     goal_xyz(3), goal_euler(3), pos_dist(1), rot_dist(1)], dim=-1)
    return {"policy": obs}
```

### `_get_rewards`

Port from `wrapper_push.py:232-312` or `reward_pbrs.py`:

```python
def _get_rewards(self) -> torch.Tensor:
    obj_pos = self._target_object.data.root_pos_w[:, :3] - self.scene.env_origins
    obj_euler = _quat_to_euler_xyz(self._target_object.data.root_quat_w)
    goal_pos = self.goal_pos_euler[:, :3]
    goal_euler = self.goal_pos_euler[:, 3:6]

    # Dense improvement reward (fractional, normalized)
    d_prev = torch.norm(self.prev_obj_pos[:, :2] - goal_pos[:, :2], dim=-1)
    d_now  = torch.norm(obj_pos[:, :2] - goal_pos[:, :2], dim=-1)
    y_prev = torch.abs(_yaw_distance_rad(self.prev_obj_euler[:, 2], goal_euler[:, 2]))
    y_now  = torch.abs(_yaw_distance_rad(obj_euler[:, 2], goal_euler[:, 2]))
    pos_imp = self.cfg.dense_alpha * (d_prev - d_now) / d_prev.clamp(min=0.01)
    rot_imp = self.cfg.dense_alpha * (y_prev - y_now) / y_prev.clamp(min=0.01)

    # Penalties
    pos_penalty = 0.5 * d_now
    rot_penalty = 0.25 * y_now

    reward = pos_imp + rot_imp - pos_penalty - rot_penalty

    # Completion bonuses
    pos_ok = (d_now < self.cfg.push_success_threshold_pos)
    rot_ok = (y_now < self.cfg.push_success_threshold_rot)
    both_ok = pos_ok & rot_ok
    new_pos_ok = pos_ok & ~self.gave_completion
    reward[new_pos_ok] += 5.0
    self.gave_completion |= new_pos_ok
    new_rot_ok = both_ok & ~self.gave_rot_bonus
    reward[new_rot_ok] += 2.0
    self.gave_rot_bonus |= new_rot_ok
    self.at_goal = pos_ok

    # Catastrophe detection
    roll  = obj_euler[:, 0].abs()
    pitch = obj_euler[:, 1].abs()
    obj_z = obj_pos[:, 2]
    tipped  = (roll > 0.3) | (pitch > 0.3)
    launched = obj_z > 0.15
    oob = (obj_pos[:, 0] < self.cfg.ws_x[0]) | (obj_pos[:, 0] > self.cfg.ws_x[1]) | \
          (obj_pos[:, 1] < self.cfg.ws_y[0]) | (obj_pos[:, 1] > self.cfg.ws_y[1])

    return reward
```

### `_get_dones`

Multi-step episode logic (unlike throwing's always-terminated):

```python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    # Physics termination (from base env)
    terminated = ...  # from _target_object root state
    # Max pushes
    max_push = self.push_count >= self.cfg.max_pushes_per_episode
    # At goal (position threshold — terminates episode)
    at_goal = self.at_goal  # set in _get_rewards
    # Catastrophe
    catastrophe = tipped | launched | oob  # computed in _get_rewards, stored on self
    # Timeout
    time_out = self.episode_length_buf >= self.max_episode_length - 1
    return (terminated | max_push | at_goal | catastrophe, time_out)
```

### `_reset_idx`

```python
def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)  # scene reset
    # Reset robot
    self._robot.write_joint_position_to_sim(
        self._robot.data.default_joint_pos[env_ids], env_ids=env_ids)
    self._robot.write_joint_velocity_to_sim(
        torch.zeros_like(...), env_ids=env_ids)
    # Randomize object spawn
    reset_objects_to_random_safe_pose(self, env_ids, ...)
    # Sample filtered goal
    self._sample_goals(env_ids)
    # Move goal ghost marker
    self._update_goal_markers(env_ids)
    # Reset bookkeeping
    self.push_count[env_ids] = 0
    self.gave_completion[env_ids] = False
    self.gave_rot_bonus[env_ids] = False
    self.at_goal[env_ids] = False
    self._step_idx[env_ids] = 0
    self._joint_cmd[env_ids] = self._robot.data.joint_pos[:, self._arm_jids][env_ids]
    # Capture episode start position
    obj_root = self._target_object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
    self.ep_start_pos[env_ids] = obj_root
    ...
```

---

## File 4: `train_push_sac_her.py` (~200 lines)

Modelled on `throwing_enviroment/scripts/train_sac.py` (175 lines). Key differences: HER support, multi-step episode logging.

```python
def main():
    # Parse args: --num_envs, --max_iterations, --seed, --exp_name, --checkpoint, --headless
    # AppLauncher as usual
    
    cfg = PushDirectEnvCfg()
    cfg.scene.num_envs = args.num_envs
    
    # Create env
    env = PushDirectEnv(cfg=cfg)
    env_wrapped = DirectRLVecEnv(env)
    
    # --- HER: wrap observation with goal-aware wrapper ---
    # SB3's HerReplayBuffer needs the env to expose compute_reward() and
    # return observations with separated "observation" and "achieved_goal" and "desired_goal".
    # We create a thin wrapper that:
    #   - Splits obs[20:26] as desired_goal
    #   - Splits obs[6:12] as achieved_goal  
    #   - Splits obs (with goals zeroed) as observation
    #   - Exposes env.compute_reward(achieved_goal, desired_goal, info)
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
    env_wrapped = HerGoalEnvWrapper(env_wrapped, goal_indices={
        "desired_goal": slice(20, 26),
        "achieved_goal": slice(6, 12),
    })
    
    # SAC
    model = SAC(
        "MultiInputPolicy",  # required for dict obs with HER
        env_wrapped,
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            online_sampling=True,
            max_episode_length=env.cfg.max_pushes_per_episode,
        ),
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, (args.max_iterations * env.num_envs) // 10),
        save_path=log_dir,
        name_prefix="agent",
    )
    latest_ckpt_callback = LatestCheckpointCallback(
        save_path=os.path.join(log_dir, "latest_checkpoint.zip"),
        save_freq=max(1, (args.max_iterations * env.num_envs) // 20),
        include=["replay_buffer"],
    )
    
    # Resume
    if args.checkpoint:
        model = SAC.load(args.checkpoint, env=env_wrapped, ...)
    
    # Train
    total_timesteps = args.max_iterations * args.num_envs
    model.learn(total_timesteps, callback=[checkpoint_callback, latest_ckpt_callback])
    model.save(os.path.join(log_dir, "agent_final"))
```

---

## File 5: `hpc/train_push_sac_her.slurm` (~130 lines)

Direct copy of `throwing_enviroment/hpc/train_sac_direct.slurm` with these substitutions:

| throwing_enviroment | push |
|---------------------|------|
| `throw_sac_d` | `push_sac_her` |
| `logs/sac/throwing_primitive` | `logs/sac/push_primitive` |
| `CONTAINER_PROJECT="/workspace/isaaclab/.../throwing_enviroment"` | `CONTAINER_PROJECT="/workspace/isaaclab/.../asyncDualPlayPPO"` |
| `scripts/train_sac.py` | `asyncDualPlayPPO/train_push_sac_her.py` |
| `--playing_arm_side` | removed (not applicable) |
| `NUM_ENVS=4096` | `NUM_ENVS=4096` (same) |
| `MAX_ITERATIONS=100000` | `MAX_ITERATIONS=2000` (push episodes are multi-step, fewer outer steps) |
| `PYTHONPATH` | `asyncDualPlayPPO:$CONTAINER_PROJECT` |
| `SIF_IMAGE` | path adjusted for asyncDualPlayPPO parent |

Keep: SIGTERM forwarding, cleanup trap with NFS sync, auto-resume chain with `latest_checkpoint.zip`, `MAX_RESUBMITS=50`.

Container mounts: add cuRobo overlay bind (read-only), Isaac Lab cache dirs.

---

## Build order

| Step | File | Action | Depends on |
|------|------|--------|------------|
| 1 | `tasks/sb3_vec_env.py` | Copy from throwing, verify imports | — |
| 2 | `tasks/push_direct_env_cfg.py` | Write config | — |
| 3 | `tasks/push_direct_env.py` | `_setup_scene` + `__init__` | 1, 2 |
| 4 | `tasks/push_direct_env.py` | `_apply_action` (IK integration) | 3 |
| 5 | `tasks/push_direct_env.py` | `_get_observations` | 3 |
| 6 | `tasks/push_direct_env.py` | `_get_rewards` + `_get_dones` | 5 |
| 7 | `tasks/push_direct_env.py` | `_reset_idx` + goal management | 6 |
| 8 | `train_push_sac_her.py` | Training script + HER wrapper | 7 |
| 9 | `hpc/train_push_sac_her.slurm` | SLURM script | 8 |

---

## Risk areas

| Risk | Severity | Mitigation |
|------|----------|------------|
| **cuRobo IK in `_apply_action()` CUDA graph recompilation** on partial env reset (different N per call). `solve_batch` was pre-warmed with fixed N. | High | Test with partial resets early. If IK fails on variable N, fall back: call `solve_batch` with full N_envs always, but mask invalid rows for dead envs. |
| **`replicate_physics=True` with IK** — does cuRobo's `solve_batch` work when physics is replicated (single USD stage)? | Medium | The current push task also uses replicated physics. cuRobo solves in joint space per-env with per-env EE targets. Should work since IK is geometry-only (no physics state dependency). |
| **HER reward recomputation** — SB3's `HerReplayBuffer` replaces goals but doesn't recompute dense improvement rewards which depend on `prev_obj_pos`. | High | Store `prev_obj_pos`/`prev_obj_euler` in the `info` dict per transition. In `compute_reward(achieved_goal, desired_goal, info)`, use stored prev_obj to recompute dense reward with the relabeled goal. |
| **Decimation=72 is large** — long step() blocks SB3's training thread. SB3 batches N_envs transitions per step, but the step itself takes ~72 IK solves. | Low | Same as current training loop. throwing_environment runs decimation=320 fine. |
| **MultiCategorical action space** — current push uses 4D×21 bins. SAC needs continuous Box. | Low | Action space becomes 4D Box `[-1,1]`. Decoding maps to (Xs, Ys, len, theta) with workspace clamping. This is a superset of the discrete space — more expressive, less constrained. |
| **`goal_ghost` as `VisualizationMarkers`** — no physics body for the goal marker. If the reward function needs to read a RigidObject for the goal, this breaks. | Low | Goals are stored as tensors (`self.goal_pos_euler`), not read from a RigidObject. `VisualizationMarkers` are visual-only, which is correct. |

---

## Total effort: 1–2 days

The `throwing_enviroment` provides a validated template that eliminates architectural risk. The novel code is limited to cuRobo IK integration inside `_apply_action()` and HER relabeling support.
