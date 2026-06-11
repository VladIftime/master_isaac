"""
Push environment wrapper for single-agent push-PPO baseline.

Manages:
  - Single observation space (29D for 1 object)
  - Goal sampling and tracking
  - Dense reward computation after each push macro-action
  - Object placement on reset
  - Episode termination conditions

Does NOT manage push trajectory execution — that happens in the training loop
using cuRobo IK + action_push.py waypoints.
"""

import torch
import numpy as np
import gymnasium as gym


# ── Reward constants ───────────────────────────────────────────────────────────
PUSH_SUCCESS_THRESHOLD_POS = 0.05   # metres
PUSH_SUCCESS_THRESHOLD_ROT = 0.2    # radians (~11°), matches ASP wrapper goal_tolerance
PUSH_COMPLETION_BONUS = 5.0
PUSH_ROTATION_SUB_BONUS = 2.0   # layered on top of position bonus when rotation also matched (Fix P14)
PUSH_DENSE_ALPHA = 3.0        # fractional improvement gain (unitless) — same for pos and rot
# Normalised: pos_imp = α·(d_prev−d_now)/d_prev, rot_imp = α·(y_prev−y_now)/y_prev
# At d_prev=0.25m→0.13m:  3×0.12/0.25 = 1.44
# At y_prev=1.5rad→1.0rad: 3×0.50/1.50 = 1.00
# Both produce comparable reward magnitudes from a single coefficient.
# Replaces the old separate PUSH_DENSE_ALPHA=12 / PUSH_DENSE_ROT_ALPHA=1 (Fix P47).
PUSH_DENSE_BETA = 0.5        # distance penalty (urgency)
PUSH_DENSE_ROT_BETA = 0.25   # continuous rotation penalty — mirror of positional urgency (Fix P17)

TIP_OVER_THRESHOLD = 0.3     # rad — if abs(roll) or abs(pitch) > this, object is unrecoverable (Fix P16)

# Workspace bounds for goal sampling (local frame, relative to env origin)
_GOAL_X_RANGE = (-0.40, 0.40)
_GOAL_Y_RANGE = (0.30, 0.70)
_GOAL_Z = 0.02   # just above table surface so the ghost marker is visible

# Observation layout indices (see push_task_curobo.py)
# [ee_pose(6) | obj_state(14) | goal_pose(6) | goal_dist(2)] = 28D (no gripper)
#   — rel_obs adds [rel_dx(1) | rel_dy(1)] at the end → 30D
_OBS_ROBOT_DIM = 6
_OBS_OBJ_STATE_DIM = 14
_OBS_GOAL_DIM = 6
_OBS_DIST_DIM = 2
_OBS_BASE_DIM = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + _OBS_GOAL_DIM + _OBS_DIST_DIM  # 28
_OBS_REL_DIM = 2  # rel_dx, rel_dy appended when rel_obs=True


def _rot_distance_rad(euler_a: torch.Tensor, euler_b: torch.Tensor) -> torch.Tensor:
    """Maximum absolute Euler-angle difference with wraparound (range [0, π])."""
    # fmod to [0, 2π) first — goal is sampled in [0,2π], obs Euler is in [-π,π],
    # so the raw diff can exceed 2π and 2π−diff would go negative without this step.
    diff = (euler_a - euler_b) % (2.0 * torch.pi)
    diff = torch.where(diff > torch.pi, 2.0 * torch.pi - diff, diff)
    return diff.max(dim=-1)[0]


def _yaw_distance_rad(euler_a: torch.Tensor, euler_b: torch.Tensor) -> torch.Tensor:
    """Yaw-only Euler-angle difference with wraparound (range [0, π]).
    
    Isolates the Z-axis (yaw) component, ignoring roll/pitch wobble.
    This is the correct metric for planar pushing: a tipped block's roll/pitch
    error would otherwise contaminate the rotation reward with noise (Fix P15).
    """
    yaw_a = euler_a[..., 2]
    yaw_b = euler_b[..., 2]
    diff = (yaw_a - yaw_b) % (2.0 * torch.pi)
    diff = torch.where(diff > torch.pi, 2.0 * torch.pi - diff, diff)
    return diff


def _euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """ZYX Euler (roll, pitch, yaw) → unit quaternion (w, x, y, z)."""
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


class PushEnvWrapper:
    """
    Wraps a ManagerBasedRLEnv for single-agent push-PPO.

    The wrapper:
      - Provides a single observation space for the push agent (29D)
      - Computes dense reward after each push macro-action
      - Samples random goals on episode reset (and per-episode for done envs)
      - Moves the scene's goal_ghost marker to the sampled goal in world frame
      - Tracks object positions for improvement computation
    """

    def __init__(
        self,
        env,
        device: torch.device,
        num_objects: int = 1,
        max_pushes_per_episode: int = 20,
        headless: bool = False,
        rel_obs: bool = False,
    ):
        self.env = env
        self.device = device
        self.num_objects = num_objects
        self.max_pushes_per_episode = max_pushes_per_episode
        self.headless = headless
        self.rel_obs = rel_obs

        self.obs_dim = _OBS_BASE_DIM + (_OBS_REL_DIM if rel_obs else 0)
        self.robot_dim = _OBS_ROBOT_DIM
        self.obj_state_dim = _OBS_OBJ_STATE_DIM

        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"),
            shape=(self.obs_dim,), dtype=np.float32,
        )
        self.state_space = self.observation_space
        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"),
            shape=env.single_action_space.shape, dtype=np.float32,
        )

        self.push_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.prev_obj_pos = torch.zeros(self.num_envs, 3, device=device)
        self.prev_obj_euler = torch.zeros(self.num_envs, 3, device=device)
        self.at_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.goal_pos_euler = torch.zeros(self.num_envs, 6, device=device)
        self._gave_completion = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._gave_rot_bonus = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._last_pos_err = torch.zeros(self.num_envs, device=device)
        self._last_rot_err = torch.zeros(self.num_envs, device=device)
        self._last_pos_imp = torch.zeros(self.num_envs, device=device)
        self._last_rot_imp = torch.zeros(self.num_envs, device=device)
        self._last_penalty  = torch.zeros(self.num_envs, device=device)
        self._last_completion = torch.zeros(self.num_envs, device=device)

        self._ep_start_pos = torch.zeros(self.num_envs, 3, device=device)
        self._ep_start_euler = torch.zeros(self.num_envs, 3, device=device)
        self._ep_started = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self.episode_push_counts = []
        self.episode_successes = []
        self.episode_rew_ema = 0.0

    def reset(self) -> torch.Tensor:
        """Reset all environments and sample new goals."""
        self.push_count.zero_()
        self.at_goal.zero_()
        self._gave_completion.zero_()
        self._gave_rot_bonus.zero_()
        self._ep_started.zero_()

        self.env.reset()
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_object_spawn(all_ids)
        self._sample_goals_filtered(all_ids)
        self._update_goal_in_extras()
        self._move_goal_ghost(all_ids)
        obs = self._get_push_obs()
        self._capture_prev_obj(obs)
        return obs

    def step(self, action: torch.Tensor):
        """
        Single physics step.  Reward is always zero — push reward is computed
        by compute_push_reward() after the full trajectory completes.
        """
        obs_dict, reward, terminated, truncated, _ = self.env.step(action)
        obs = self._build_obs(obs_dict)
        return obs, torch.zeros_like(reward), terminated, truncated, {}

    def _build_obs(self, obs_dict: dict) -> torch.Tensor:
        obs = obs_dict["push_policy"]
        if self.rel_obs:
            obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
            goal_pos = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM:
                           _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
            rel_dx = (goal_pos[:, 0:1] - obj_pos[:, 0:1])
            rel_dy = (goal_pos[:, 1:2] - obj_pos[:, 1:2])
            obs = torch.cat([obs, rel_dx, rel_dy], dim=-1)
        return obs

    def _get_push_obs(self) -> torch.Tensor:
        """Recompute push observation from current sim state (after object moves).
        
        Calls observation_manager.compute() outside env.step() — safe because
        the only observation group (push_policy) has no side effects
        (no sensors, no cameras).  The low-volume print confirms call frequency."""
        self._update_goal_in_extras()
        obs_dict = self.env.observation_manager.compute()
        return self._build_obs(obs_dict)

    def _randomize_object_spawn(self, env_ids: torch.Tensor):
        """Teleport the target object to a random position within workspace.
        X∈[-0.4,0.4] Y∈[0.3,0.7] Z=0.05 yaw∈[0,2π] — same ranges as goals.
        Forces the policy to read the observation instead of memorizing (0, 0.5)."""
        if len(env_ids) == 0:
            return
        N = len(env_ids)
        ox = torch.empty(N, device=self.device).uniform_(*_GOAL_X_RANGE)
        oy = torch.empty(N, device=self.device).uniform_(*_GOAL_Y_RANGE)
        oz = torch.full((N,), 0.05, device=self.device)
        oyaw = torch.empty(N, device=self.device).uniform_(0, 2 * torch.pi)
        oeuler = torch.zeros(N, 3, device=self.device)
        oeuler[:, 2] = oyaw

        origins = self.env.scene.env_origins[env_ids]
        pos_local = torch.stack([ox, oy, oz], dim=-1)
        pos_world = pos_local + origins
        quat = _euler_to_quat(oeuler)
        pose_7d = torch.cat([pos_world, quat], dim=-1)
        self.env.scene["target_object"].write_root_pose_to_sim(pose_7d, env_ids=env_ids)

    def capture_pre_push(self, obs: torch.Tensor):
        """Snapshot object pose before a push for improvement computation."""
        self.prev_obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3].clone()
        self.prev_obj_euler = obs[:, _OBS_ROBOT_DIM + 3: _OBS_ROBOT_DIM + 6].clone()
        # Record episode start state (first push of a new episode)
        new_ep = ~self._ep_started
        if new_ep.any():
            self._ep_start_pos[new_ep] = self.prev_obj_pos[new_ep].clone()
            self._ep_start_euler[new_ep] = self.prev_obj_euler[new_ep].clone()
            self._ep_started[new_ep] = True

    def compute_push_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Dense reward after a complete push macro-action.

        R = α·(d_prev−d_now)/d_prev                      pos improvement (fractional)
          + α·(y_prev−y_now)/y_prev                      rot improvement (fractional)
          − β·d_now                                      distance penalty
          − β_rot·y_now                                  continuous yaw penalty
          + completion_bonus                              +5 for pos<0.05 (position gate)
          + rotation_sub_bonus                            +2 for pos<0.05 AND yaw<0.2
          − tip_penalty                                  −5 for tipped block

        Normalised deltas use the same α coefficient — a push that halves the
        remaining distance earns α×0.5 regardless of whether it's position or
        rotation.  Denominators clamped to min=0.01 to avoid division by zero.

        Off-center pushes induce torque (Akella & Mason 1998).
        """
        cur_obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
        cur_obj_euler = obs[:, _OBS_ROBOT_DIM + 3: _OBS_ROBOT_DIM + 6]
        goal_pos = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
        goal_euler = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 6]

        d_prev = (self.prev_obj_pos - goal_pos).norm(dim=-1)
        d_now = (cur_obj_pos - goal_pos).norm(dim=-1)

        # Yaw-only improvement — isolates planar rotation from roll/pitch wobble (Fix P15)
        y_prev = _yaw_distance_rad(self.prev_obj_euler, goal_euler)
        y_now  = _yaw_distance_rad(cur_obj_euler, goal_euler)

        # Full Euler diff for logging / tip-over detection only
        r_now = _rot_distance_rad(cur_obj_euler, goal_euler)

        pos_err = d_now
        rot_err = r_now
        self._last_pos_err = pos_err
        self._last_rot_err = rot_err

        at_goal = pos_err < PUSH_SUCCESS_THRESHOLD_POS
        rot_at_goal = rot_err < PUSH_SUCCESS_THRESHOLD_ROT

        pos_imp  = PUSH_DENSE_ALPHA * (d_prev - d_now) / d_prev.clamp(min=0.01)
        rot_imp  = PUSH_DENSE_ALPHA * (y_prev - y_now) / y_prev.clamp(min=0.01)
        penalty  = -PUSH_DENSE_BETA * d_now
        rot_penalty = -PUSH_DENSE_ROT_BETA * y_now                   # continuous yaw urgency (Fix P17)

        # Clamp reward components to defend against PhysX collision glitches
        # that launch objects thousands of metres away (0.26% of episodes).
        # Without clamping, d_now=6800m → penalty=−3400, pos_imp=−81600,
        # and the resulting return slaughters the critic (Val loss 356k+).
        pos_imp = torch.clamp(pos_imp, min=-5.0, max=5.0)           # Fix P30
        rot_imp = torch.clamp(rot_imp, min=-4.0, max=4.0)           # Fix P30
        penalty = torch.clamp(penalty, min=-2.0, max=0.0)           # Fix P30
        rot_penalty = torch.clamp(rot_penalty, min=-1.0, max=0.0)   # Fix P30
        reward = pos_imp + rot_imp + penalty + rot_penalty

        # Completion bonus (position gate — keeps 5.7% SR floor)
        new_completion = at_goal & ~self._gave_completion
        completion = torch.where(new_completion,
                                  torch.tensor(PUSH_COMPLETION_BONUS, device=self.device),
                                  torch.zeros_like(reward))
        reward = reward + completion

        # Rotation sub-bonus: +2 only when BOTH position AND rotation match (Fix P14)
        new_rot_bonus = at_goal & rot_at_goal & (~self._gave_completion | ~self._gave_rot_bonus)
        rot_bonus = torch.where(new_rot_bonus,
                                torch.tensor(PUSH_ROTATION_SUB_BONUS, device=self.device),
                                torch.zeros_like(reward))
        reward = reward + rot_bonus

        # Tip-over penalty: object is unrecoverable if tipped (Fix P16)
        tipped = (cur_obj_euler[:, 0].abs() > TIP_OVER_THRESHOLD) | \
                 (cur_obj_euler[:, 1].abs() > TIP_OVER_THRESHOLD)
        tip_penalty = torch.where(tipped,
                                  torch.tensor(-5.0, device=self.device),
                                  torch.zeros_like(reward))
        reward = reward + tip_penalty

        self._gave_completion[self._gave_completion | new_completion] = True
        self._gave_rot_bonus[self._gave_rot_bonus | new_rot_bonus] = True
        self.at_goal = at_goal

        self._last_pos_imp = pos_imp
        self._last_rot_imp = rot_imp
        self._last_penalty  = penalty
        self._last_completion = completion + rot_bonus + tip_penalty

        self.push_count += 1

        return reward

    def check_done(self, _obs: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """Episode ends: base termination, max pushes, position success, object launched, or tipped.
        Position success now terminates immediately — the object reached the goal,
        so the agent should reset and practice on a fresh goal instead of pushing
        it away with remaining push budget."""
        max_pushes = self.push_count >= self.max_pushes_per_episode
        obj_z = _obs[:, _OBS_ROBOT_DIM + 2]
        launched = obj_z > 0.05
        tipped = (_obs[:, _OBS_ROBOT_DIM + 3].abs() > TIP_OVER_THRESHOLD) | \
                 (_obs[:, _OBS_ROBOT_DIM + 4].abs() > TIP_OVER_THRESHOLD)
        obj_pos = _obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
        goal_pos = _obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
        out_of_bounds = (obj_pos - goal_pos).norm(dim=-1) > 0.5
        at_goal_pos = (obj_pos - goal_pos).norm(dim=-1) < PUSH_SUCCESS_THRESHOLD_POS
        return terminated | max_pushes | at_goal_pos | launched | tipped | out_of_bounds

    def reset_done_envs(self, dones: torch.Tensor):
        """Reset per-env bookkeeping for envs that finished an episode.
        Goal sampling and object randomisation happen in train_push.py's
        reset block AFTER the base env reset (so the object position is known
        before a goal is chosen — Fix P4 + P6)."""
        done_ids = torch.where(dones)[0]
        if len(done_ids) == 0:
            return
        self.episode_push_counts.extend(self.push_count[done_ids].cpu().tolist())
        self.episode_successes.extend(self.at_goal[done_ids].cpu().tolist())
        self.push_count[done_ids] = 0
        self.at_goal[done_ids] = False
        self._gave_completion[done_ids] = False
        self._gave_rot_bonus[done_ids] = False

    # ── Goal management ────────────────────────────────────────────────────────

    def _sample_goals(self, env_ids: torch.Tensor):
        """Sample random goal positions (local frame) for the specified envs.
        No distance filter — use _sample_goals_filtered if object position is known."""
        N = len(env_ids)
        gx = torch.empty(N, device=self.device).uniform_(*_GOAL_X_RANGE)
        gy = torch.empty(N, device=self.device).uniform_(*_GOAL_Y_RANGE)
        gz = torch.full((N,), _GOAL_Z, device=self.device)
        geuler = torch.zeros(N, 3, device=self.device)
        geuler[:, 2] = torch.empty(N, device=self.device).uniform_(0, 2 * torch.pi)
        self.goal_pos_euler[env_ids] = torch.cat([
            gx.unsqueeze(-1), gy.unsqueeze(-1), gz.unsqueeze(-1), geuler,
        ], dim=-1)

    def _sample_goals_filtered(self, env_ids: torch.Tensor):
        """Sample goals and reject those within success threshold of the object.
        
        Must be called AFTER object position is finalized (i.e. after
        _randomize_object_spawn).  Reads the current object scene position
        and resamples any goal that falls within 0.05 m of it — preventing
        episodes that would terminate instantly with zero pushes (P4)."""
        if len(env_ids) == 0:
            return
        N = len(env_ids)
        obj = self.env.scene["target_object"]
        obj_pos_w = obj.data.root_pos_w[env_ids]
        obj_pos_local = obj_pos_w - self.env.scene.env_origins[env_ids]

        gx = torch.empty(N, device=self.device).uniform_(*_GOAL_X_RANGE)
        gy = torch.empty(N, device=self.device).uniform_(*_GOAL_Y_RANGE)
        gz = torch.full((N,), _GOAL_Z, device=self.device)
        geuler = torch.zeros(N, 3, device=self.device)
        geuler[:, 2] = torch.empty(N, device=self.device).uniform_(0, 2 * torch.pi)

        goal_xy = torch.stack([gx, gy], dim=-1)
        obj_xy = obj_pos_local[:, :2]
        dist = (goal_xy - obj_xy).norm(dim=-1)
        too_close = dist < PUSH_SUCCESS_THRESHOLD_POS

        if too_close.any():
            tc_count = int(too_close.sum().item())
            # Resample goals for too-close envs
            gx[too_close] = torch.empty(tc_count, device=self.device).uniform_(*_GOAL_X_RANGE)
            gy[too_close] = torch.empty(tc_count, device=self.device).uniform_(*_GOAL_Y_RANGE)

        self.goal_pos_euler[env_ids] = torch.cat([
            gx.unsqueeze(-1), gy.unsqueeze(-1), gz.unsqueeze(-1), geuler,
        ], dim=-1)

    def _update_goal_in_extras(self):
        """Write the current goal tensor into env.extras so observation fns can read it."""
        if not hasattr(self.env, "extras"):
            self.env.extras = {}
        # Key matches what observations.goal_states() reads
        self.env.extras["goal_state"] = self.goal_pos_euler

    def _move_goal_ghost(self, env_ids: torch.Tensor):
        """Teleport the flat goal marker to the sampled goal on the table surface."""
        if "goal_marker" not in self.env.scene.rigid_objects:
            return
        origins = self.env.scene.env_origins[env_ids]          # (N, 3) world
        goal_pos_local = self.goal_pos_euler[env_ids, :3].clone()  # (N, 3) local
        goal_pos_local[:, 2] = 0.001  # flat on table surface (no collision — handled by collision_props)
        goal_euler = self.goal_pos_euler[env_ids, 3:6].clone()  # (N, 3) euler
        goal_euler[:, 0] = 0.0  # zero roll — keep marker flat
        goal_euler[:, 1] = 0.0  # zero pitch
        goal_pos_world = goal_pos_local + origins              # (N, 3) world
        goal_quat = _euler_to_quat(goal_euler)                 # (N, 4) wxyz
        pose_7d = torch.cat([goal_pos_world, goal_quat], dim=-1)
        self.env.scene["goal_marker"].write_root_pose_to_sim(pose_7d, env_ids=env_ids)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _capture_prev_obj(self, obs: torch.Tensor):
        self.prev_obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3].clone()
        self.prev_obj_euler = obs[:, _OBS_ROBOT_DIM + 3: _OBS_ROBOT_DIM + 6].clone()

    def _get_obj_pos(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]

    def _get_goal_pos(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def unwrapped(self):
        return self
