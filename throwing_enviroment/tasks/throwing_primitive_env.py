"""Throwing Primitive Environment (gymnasium.Env wrapper).

One-shot episodic environment matching the Gazebo MDP:
  - Action space: Box(4) = [initial_joint_value, final_joint_value, releasing_time, duration]
  - Observation space: Box(8) = [robot_obs, basket_x, basket_y, obj_x, obj_y, dist, dist_x, dist_y]
  - Each step() executes the full throw primitive (IK grasping + joint-space throw)
  - Episode = 1 step, always terminates after the throw
  - Reward: Widened exponential + linear (0.9*exp(-d²/0.1) + 0.1*exp(-d²/0.5) + 0.5*max(0,1-d))

Internally creates a ThrowingEnv (ManagerBasedRLEnv) and uses IK-based grasping
via env.step(ik_action), matching the proven pickup flow from test_joint_throwing.py.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .throw_primitive import (
    execute_primitive_batched,
    map_action_to_params,
    DROP_PENALTY_DISTANCE,
    DRINK_WORLD_X,
    DRINK_WORLD_Y,
    DRINK_WORLD_Z,
)
from .events import _set_gripper_state

OBS_MAX_NORM = 3.0
_LOG_INTERVAL = 1


class ThrowingPrimitiveEnv(gym.Env):
    """One-shot throwing environment with Gazebo-style macro-action (4D).

    Wraps ThrowingEnv to use IK-based grasping for the pickup phases, then
    executes the joint-space throw primitive with the 4 learned parameters.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg=None, render_mode=None, **kwargs):
        super().__init__()

        if cfg is None:
            from .throwing_primitive_env_cfg import ThrowingPrimitiveEnvCfg
            cfg = ThrowingPrimitiveEnvCfg()

        self.cfg = cfg
        self._render_mode = render_mode

        from .throwing_env_cfg import ThrowingEnvCfg
        from .throwing_env import ThrowingEnv

        env_cfg = ThrowingEnvCfg()
        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.ik_solver = "diffik"
        env_cfg.playing_arm_side = cfg.playing_arm_side
        env_cfg.release_at_step = 0
        env_cfg.release_vel_threshold = float("inf")
        env_cfg.disable_attachment = True
        env_cfg.randomize_target = cfg.randomize_target
        env_cfg.target_x_range = cfg.target_x_range
        env_cfg.target_y_range = cfg.target_y_range
        env_cfg.__post_init__()
        env_cfg.episode_length_s = 60.0

        from isaaclab.managers import TerminationTermCfg as DoneTerm
        import isaaclab.envs.mdp as mdp
        from isaaclab.utils import configclass

        @configclass
        class NoTerminationsCfg:
            time_limit = DoneTerm(func=mdp.time_out, time_out="truncated")

        env_cfg.terminations = NoTerminationsCfg()

        if hasattr(env_cfg.actions.arm, "scale"):
            env_cfg.actions.arm.scale = 0.8
        if hasattr(env_cfg.actions.arm, "position_scale"):
            env_cfg.actions.arm.position_scale = 0.8
            env_cfg.actions.arm.orientation_scale = 0.8

        self._env = ThrowingEnv(cfg=env_cfg, render_mode=render_mode)
        self._env.reset()

        self._env._holding[:] = False
        self._env._released[:] = False

        self._side = cfg.playing_arm_side
        self._ee_body_name = f"{self._side}_wrist_3_link"

        if self._side == "right":
            arm_patterns = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]
        else:
            arm_patterns = ["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"]

        robot = self._env.scene["robot"]
        self._arm_ids, _ = robot.find_joints(arm_patterns)

        self._robot_obs_val = -1.0 if self._side == "left" else 1.0

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 0.05, 0.1], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones(8, dtype=np.float32),
            high=np.inf * np.ones(8, dtype=np.float32),
        )

        self.num_envs = cfg.num_envs
        self.device = self._env.device
        self._episode_count = 0
        self._tb_writer: SummaryWriter | None = None
        self._grasp_cache: dict = {}

    @property
    def state_space(self):
        return self.observation_space

    @property
    def unwrapped(self):
        return self

    @property
    def num_actions(self):
        return 4

    @property
    def num_observations(self):
        return 8

    def set_log_dir(self, log_dir: str):
        self._tb_writer = SummaryWriter(log_dir=log_dir)

    def _get_obs(self) -> torch.Tensor:
        """Compute 8D observation for all envs. Returns (N, 8) tensor."""
        milk = self._env.scene["milk"]
        target = self._env.scene["target"]
        origins = self._env.scene.env_origins.to(self.device)

        milk_pos = milk.data.root_pos_w[:, :3] - origins
        target_pos = target.data.root_pos_w[:, :3] - origins

        dist_vec = milk_pos - target_pos
        dist = torch.norm(dist_vec, dim=-1, keepdim=True)
        dist_x = torch.abs(dist_vec[:, 0:1])
        dist_y = torch.abs(dist_vec[:, 1:2])

        robot_indicator = torch.full(
            (self.num_envs, 1), self._robot_obs_val, device=self.device
        )

        obs = torch.cat([
            robot_indicator,
            target_pos[:, 0:1] / OBS_MAX_NORM,
            target_pos[:, 1:2] / OBS_MAX_NORM,
            milk_pos[:, 0:1] / OBS_MAX_NORM,
            milk_pos[:, 1:2] / OBS_MAX_NORM,
            dist / OBS_MAX_NORM,
            dist_x / OBS_MAX_NORM,
            dist_y / OBS_MAX_NORM,
        ], dim=-1)

        return obs

    def _compute_reward(self, distances: torch.Tensor) -> torch.Tensor:
        """Widened exponential + linear distance reward."""
        alpha = 0.9
        reward = (
            alpha * torch.exp(-(distances ** 2) / 0.1)
            + (1.0 - alpha) * torch.exp(-(distances ** 2) / 0.5)
            + 0.5 * torch.clamp(1.0 - distances, min=0.0)
        )
        success_mask = distances < 0.15
        reward[success_mask] = 2.0
        return reward

    def step(self, action):
        """Execute full throw primitive for all envs. Returns (obs, reward, terminated, truncated, info)."""
        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action).float().to(self.device)
        else:
            action_t = action.clone().to(self.device)

        if action_t.dim() == 1:
            action_t = action_t.unsqueeze(0)

        self._env.reset()
        self._env._holding[:] = False
        self._env._released[:] = False

        result = execute_primitive_batched(
            env=self._env,
            actions=action_t,
            arm_joint_ids=self._arm_ids,
            ee_body_name=self._ee_body_name,
            gripper_set_fn=_set_gripper_state,
            side=self._side,
            drink_x=self.cfg.drink_x,
            drink_y=self.cfg.drink_y,
            drink_z=self.cfg.drink_z,
            grasp_cache=self._grasp_cache,
        )

        distances = result["distances"]
        dropped = result["dropped"]
        milk_final_pos = result["milk_final_pos"]
        target_pos = result["target_pos"]

        reward = self._compute_reward(distances)
        obs = self._get_obs()

        terminated = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        info = {
            "distances": distances.cpu().numpy(),
            "dropped": dropped.cpu().numpy(),
        }

        self._episode_count += self.num_envs
        self._log_episodes(action_t, target_pos, milk_final_pos, distances, reward, dropped)

        return obs, reward, terminated, truncated, info

    def _log_episodes(self, actions, target_pos, milk_final_pos, distances, rewards, dropped):
        """Log episode summary for all envs (stdout + TensorBoard)."""
        params = map_action_to_params(actions, side=self._side)
        n_dropped = dropped.sum().item()
        n_success = (distances < 0.15).sum().item()
        valid_mask = ~dropped
        mean_dist = distances[valid_mask].mean().item() if valid_mask.any() else float("inf")
        min_dist = distances[valid_mask].min().item() if valid_mask.any() else float("inf")
        max_dist = distances[valid_mask].max().item() if valid_mask.any() else float("inf")
        mean_reward = rewards.mean().item()
        success_rate = n_success / self.num_envs
        drop_rate = n_dropped / self.num_envs

        print(
            f"[Ep {self._episode_count:>6}] "
            f"reward={mean_reward:.4f}  dist={mean_dist:.3f}m  "
            f"success={n_success}/{self.num_envs}  dropped={n_dropped}/{self.num_envs}"
        )

        if self.num_envs <= 50:
            for idx in range(self.num_envs):
                tgt = target_pos[idx]
                obj = milk_final_pos[idx]
                act = params[idx]
                d = distances[idx].item()
                r = rewards[idx].item()
                early = dropped[idx].item()
                print(
                    f"  env[{idx}]: target=({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}) "
                    f"action=[ijv={act[0]:.3f} fjv={act[1]:.3f} rel={act[2]:.3f} dur={act[3]:.3f}] "
                    f"obj=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f}) "
                    f"dist={d:.3f} rew={r:.4f}"
                    f"{' [EARLY_END]' if early else ''}"
                )
        else:
            log_idx = 0
            tgt = target_pos[log_idx]
            obj = milk_final_pos[log_idx]
            act = params[log_idx]
            d = distances[log_idx].item()
            r = rewards[log_idx].item()
            early = dropped[log_idx].item()
            print(
                f"  env[0]: target=({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}) "
                f"action=[ijv={act[0]:.3f} fjv={act[1]:.3f} rel={act[2]:.3f} dur={act[3]:.3f}] "
                f"obj=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f}) "
                f"dist={d:.3f} rew={r:.4f}"
                f"{' [EARLY_END]' if early else ''}"
            )

        if self._tb_writer is not None:
            step = self._episode_count
            self._tb_writer.add_scalar("Throw / Mean Distance", mean_dist, step)
            self._tb_writer.add_scalar("Throw / Min Distance", min_dist, step)
            self._tb_writer.add_scalar("Throw / Max Distance", max_dist, step)
            self._tb_writer.add_scalar("Throw / Success Rate", success_rate, step)
            self._tb_writer.add_scalar("Throw / Drop Rate", drop_rate, step)
            self._tb_writer.add_scalar("Throw / Mean Reward", mean_reward, step)
            mean_params = params.mean(dim=0)
            self._tb_writer.add_scalar("Action / Mean Initial JV", mean_params[0].item(), step)
            self._tb_writer.add_scalar("Action / Mean Final JV", mean_params[1].item(), step)
            self._tb_writer.add_scalar("Action / Mean Release Time", mean_params[2].item(), step)
            self._tb_writer.add_scalar("Action / Mean Duration", mean_params[3].item(), step)
            self._tb_writer.flush()

    def reset(self, seed=None, options=None):
        """Reset environment: randomize target, return fresh observation."""
        if seed is not None:
            torch.manual_seed(seed)

        self._env.reset()
        self._env._holding[:] = False
        self._env._released[:] = False

        obs = self._get_obs()
        return obs, {}

    def render(self):
        return self._env.render()

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
        self._env.close()

    @property
    def sim(self):
        return self._env.sim
