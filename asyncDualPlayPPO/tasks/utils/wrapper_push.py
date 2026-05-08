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
PUSH_DENSE_ALPHA = 10.0   # improvement gain
PUSH_DENSE_BETA = 0.5     # absolute distance penalty

# Workspace bounds for goal sampling (local frame)
_GOAL_X_RANGE = (-0.40, 0.40)
_GOAL_Y_RANGE = (0.30, 0.70)

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


class PushEnvWrapper:
    """
    Wraps a ManagerBasedRLEnv for single-agent push-PPO.

    The wrapper:
      - Provides a single observation space for the push agent (29D)
      - Computes dense reward after each push macro-action
      - Samples random goals on episode reset
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
        self.num_envs = env.num_envs
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

        self.episode_push_counts = []
        self.episode_successes = []
        self.episode_rew_ema = 0.0

    def reset(self) -> torch.Tensor:
        """Reset environment and sample new goals."""
        self.push_count.zero_()
        self.at_goal.zero_()
        self._gave_completion.zero_()

        obs_dict = self.env.reset()[0]
        self._sample_goals()
        self._update_goal_in_extras()
        obs = self._build_obs(obs_dict)
        self._capture_prev_obj(obs)

        return obs

    def step(self, action: torch.Tensor):
        """
        Single physics step.  Reward is always zero — push reward is computed
        by compute_push_reward() after the full trajectory completes.
        """
        obs_dict, reward, terminated, truncated = self.env.step(action)
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

        reward = α · (d_prev − d_now) − β · d_now + completion_bonus
        """
        N = obs.shape[0]
        cur_obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
        cur_obj_euler = obs[:, _OBS_ROBOT_DIM + 3: _OBS_ROBOT_DIM + 6]
        goal_pos = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
        goal_euler = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3: _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 6]

        d_prev = (self.prev_obj_pos - goal_pos).norm(dim=-1)
        d_now = (cur_obj_pos - goal_pos).norm(dim=-1)

        pos_err = d_now
        rot_err = _rot_distance_rad(cur_obj_euler, goal_euler)
        at_goal = (pos_err < PUSH_SUCCESS_THRESHOLD_POS) & (rot_err < PUSH_SUCCESS_THRESHOLD_ROT)

        improvement = d_prev - d_now  # > 0 = got closer
        reward = PUSH_DENSE_ALPHA * improvement - PUSH_DENSE_BETA * d_now

        new_completion = at_goal & ~self._gave_completion
        reward = torch.where(new_completion, reward + PUSH_COMPLETION_BONUS, reward)
        self._gave_completion[self._gave_completion | new_completion] = True
        self.at_goal = at_goal

        self.push_count += 1

        return reward

    def check_done(self, obs: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """Episode ends: base termination, at-goal success, or max pushes reached."""
        max_pushes = self.push_count >= self.max_pushes_per_episode
        return terminated | self.at_goal | max_pushes

    def reset_done_envs(self, dones: torch.Tensor):
        """Reset per-env state for envs that finished this push."""
        done_ids = torch.where(dones)[0]
        if len(done_ids) == 0:
            return
        self.push_count[done_ids] = 0
        self.at_goal[done_ids] = False
        self._gave_completion[done_ids] = False

        done_cpu = dones.cpu().numpy()
        self.episode_push_counts.extend(self.push_count[done_ids].cpu().tolist())
        self.episode_successes.extend(self.at_goal[done_ids].cpu().tolist())

    def _sample_goals(self):
        N = self.num_envs
        gx = torch.empty(N, device=self.device).uniform_(*_GOAL_X_RANGE)
        gy = torch.empty(N, device=self.device).uniform_(*_GOAL_Y_RANGE)
        gz = torch.zeros(N, device=self.device)
        geuler = torch.zeros(N, 3, device=self.device)
        self.goal_pos_euler = torch.cat([
            gx.unsqueeze(-1), gy.unsqueeze(-1), gz.unsqueeze(-1), geuler,
        ], dim=-1)

    def _update_goal_in_extras(self):
        if not hasattr(self.env, "extras"):
            self.env.extras = {}
        self.env.extras["goal_states"] = self.goal_pos_euler

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
