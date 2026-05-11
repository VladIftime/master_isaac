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
PUSH_SUCCESS_THRESHOLD_ROT = 0.035  # radians (~2°)
PUSH_COMPLETION_BONUS = 5.0
PUSH_DENSE_ALPHA = 10.0      # position improvement gain (metres → reward)
PUSH_DENSE_ROT_ALPHA = 2.0   # rotation improvement gain (radians → reward)

# Workspace bounds for goal sampling (local frame, relative to env origin)
_GOAL_X_RANGE = (-0.40, 0.40)
_GOAL_Y_RANGE = (0.30, 0.70)
_GOAL_Z = 0.02   # just above table surface so the ghost marker is visible

# Observation layout indices (see push_task_curobo.py)
# [ee_pose(6) | gripper(1) | obj_state(14) | goal_pose(6) | goal_dist(2)] = 29D
_OBS_ROBOT_DIM = 7
_OBS_OBJ_STATE_DIM = 14
_OBS_GOAL_DIM = 6
_OBS_DIST_DIM = 2
_OBS_DIM = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + _OBS_GOAL_DIM + _OBS_DIST_DIM  # 29


def _rot_distance_rad(euler_a: torch.Tensor, euler_b: torch.Tensor) -> torch.Tensor:
    """Maximum absolute Euler-angle difference with wraparound (range [0, π])."""
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * torch.pi - diff)
    return diff.max(dim=-1)[0]


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
    ):
        self.env = env
        self.device = device
        self.num_objects = num_objects
        self.max_pushes_per_episode = max_pushes_per_episode
        self.headless = headless

        self.obs_dim = _OBS_DIM
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
        self._last_pos_err = torch.zeros(self.num_envs, device=device)
        self._last_rot_err = torch.zeros(self.num_envs, device=device)

        self.episode_push_counts = []
        self.episode_successes = []
        self.episode_rew_ema = 0.0

    def reset(self) -> torch.Tensor:
        """Reset all environments and sample new goals."""
        self.push_count.zero_()
        self.at_goal.zero_()
        self._gave_completion.zero_()

        obs_dict = self.env.reset()[0]
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_goals(all_ids)
        self._update_goal_in_extras()
        self._move_goal_ghost(all_ids)
        obs = self._build_obs(obs_dict)
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
        return obs_dict["push_policy"]

    def capture_pre_push(self, obs: torch.Tensor):
        """Snapshot object pose before a push for improvement computation."""
        self.prev_obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3].clone()
        self.prev_obj_euler = obs[:, _OBS_ROBOT_DIM + 3: _OBS_ROBOT_DIM + 6].clone()

    def compute_push_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Dense reward after a complete push macro-action.

        R = α·max(0, d_prev−d_now) + γ·max(0, r_prev−r_now) + completion_bonus

        Off-center pushes induce torque (Akella & Mason 1998).  The rotation
        improvement term rewards the agent for chaining pushes that reorient
        the object toward the goal yaw.
        """
        cur_obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
        cur_obj_euler = obs[:, _OBS_ROBOT_DIM + 3: _OBS_ROBOT_DIM + 6]
        goal_pos = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
        goal_euler = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 6]

        d_prev = (self.prev_obj_pos - goal_pos).norm(dim=-1)
        d_now = (cur_obj_pos - goal_pos).norm(dim=-1)
        r_prev = _rot_distance_rad(self.prev_obj_euler, goal_euler)
        r_now  = _rot_distance_rad(cur_obj_euler, goal_euler)

        pos_err = d_now
        rot_err = r_now
        self._last_pos_err = pos_err  # exposed for training-log metrics
        self._last_rot_err = rot_err
        at_goal = (pos_err < PUSH_SUCCESS_THRESHOLD_POS) & (rot_err < PUSH_SUCCESS_THRESHOLD_ROT)

        pos_imp = (d_prev - d_now).clamp(min=0)
        rot_imp = (r_prev - r_now).clamp(min=0)
        reward = PUSH_DENSE_ALPHA * pos_imp + PUSH_DENSE_ROT_ALPHA * rot_imp

        new_completion = at_goal & ~self._gave_completion
        reward = torch.where(new_completion, reward + PUSH_COMPLETION_BONUS, reward)
        self._gave_completion[self._gave_completion | new_completion] = True
        self.at_goal = at_goal

        self.push_count += 1

        return reward

    def check_done(self, _obs: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """Episode ends: base termination, at-goal success, or max pushes reached."""
        max_pushes = self.push_count >= self.max_pushes_per_episode
        return terminated | self.at_goal | max_pushes

    def reset_done_envs(self, dones: torch.Tensor):
        """Reset per-env state and resample goals for envs that finished an episode."""
        done_ids = torch.where(dones)[0]
        if len(done_ids) == 0:
            return
        self.episode_push_counts.extend(self.push_count[done_ids].cpu().tolist())
        self.episode_successes.extend(self.at_goal[done_ids].cpu().tolist())
        self.push_count[done_ids] = 0
        self.at_goal[done_ids] = False
        self._gave_completion[done_ids] = False
        # Give each done env a fresh goal for the next episode
        self._sample_goals(done_ids)
        self._update_goal_in_extras()
        self._move_goal_ghost(done_ids)

    # ── Goal management ────────────────────────────────────────────────────────

    def _sample_goals(self, env_ids: torch.Tensor):
        """Sample random goal positions (local frame) for the specified envs."""
        N = len(env_ids)
        gx = torch.empty(N, device=self.device).uniform_(*_GOAL_X_RANGE)
        gy = torch.empty(N, device=self.device).uniform_(*_GOAL_Y_RANGE)
        gz = torch.full((N,), _GOAL_Z, device=self.device)
        geuler = torch.zeros(N, 3, device=self.device)
        geuler[:, 2] = torch.empty(N, device=self.device).uniform_(0, 2 * torch.pi)
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
        """Teleport the visual goal_ghost marker to the sampled goal in world frame."""
        if "goal_ghost" not in self.env.scene.rigid_objects:
            return
        origins = self.env.scene.env_origins[env_ids]          # (N, 3) world
        goal_pos_local = self.goal_pos_euler[env_ids, :3]      # (N, 3) local
        goal_euler = self.goal_pos_euler[env_ids, 3:6]         # (N, 3) euler
        goal_pos_world = goal_pos_local + origins              # (N, 3) world
        goal_quat = _euler_to_quat(goal_euler)                 # (N, 4) wxyz
        pose_7d = torch.cat([goal_pos_world, goal_quat], dim=-1)
        self.env.scene["goal_ghost"].write_root_pose_to_sim(pose_7d, env_ids=env_ids)

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
