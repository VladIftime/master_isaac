"""
Push primitive SAC environment — encapsulates full push trajectory execution.

Wraps PushEnvWrapper + cuRobo IK into a 1-push-per-step gym-like interface
suitable for off-policy RL algorithms (SAC, TD3, etc).

Each call to step(action) executes a complete push trajectory:
  1. Decode continuous action → (Xs, Ys, length, theta)
  2. Compute push waypoints (approach, descend, push, retract, return)
  3. Execute waypoints via cuRobo IK → joint positions → physics steps
  4. Compute dense push reward
  5. Check done conditions
  6. Auto-reset done envs

Action space: Box(4) in [-1, 1]
  Absolute mode: maps to (Xs, Ys, length, theta) in world frame
  Relative mode: maps to (r, phi, length, theta) object-relative
"""

import torch
import numpy as np
import gymnasium as gym

from asyncDualPlayPPO.tasks.utils.action_push_continuous import (
    decode_push_action_continuous,
    decode_push_action_relative_continuous,
)
from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints


_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25, 0.70)
_WS_Z = (0.25, 0.55)

_QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)


class PushPrimitiveSACEnv:
    """
    Vectorized push environment for SAC training.

    Each step() executes one full push macro-action (72 physics substeps)
    and returns the resulting observation, reward, and done signal.
    """

    def __init__(
        self,
        push_env,
        ik_solver,
        device: torch.device,
        rel_obs: bool = False,
        rel_act: bool = False,
        max_pushes_per_episode: int = 5,
        ik_error: torch.Tensor = None,
    ):
        self.push_env = push_env
        self.ik_solver = ik_solver
        self.device = device
        self.rel_obs = rel_obs
        self.rel_act = rel_act
        self.max_pushes_per_episode = max_pushes_per_episode

        self._ik_error = ik_error if ik_error is not None else torch.zeros(1, 3, device=device)
        self._quat_tool_down = _QUAT_TOOL_DOWN.to(device)

        self._robot_scene = self.push_env.env.scene["robot"]
        self._arm_jids, _ = self._robot_scene.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
        lf_ids, _ = self._robot_scene.find_bodies("left_inner_finger")
        rf_ids, _ = self._robot_scene.find_bodies("right_inner_finger")
        self._lf_id = lf_ids[0]
        self._rf_id = rf_ids[0]

        self._obs_dim = push_env.obs_dim
        self._action_dim = 4

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32,
        )

        self._ee_pos_local = torch.zeros(self.num_envs, 3, device=device)
        self._ee_quat_w = self._quat_tool_down.expand(self.num_envs, 4).clone()
        self._prev_joint_cmd = torch.zeros(self.num_envs, 6, device=device)

        self._total_ik_fails = 0
        self._total_ik_steps = 0

    @property
    def num_envs(self) -> int:
        return self.push_env.num_envs

    @property
    def robot_dim(self) -> int:
        return self.push_env.robot_dim

    @property
    def obj_state_dim(self) -> int:
        return self.push_env.obj_state_dim

    def _tcp_pos_local(self) -> torch.Tensor:
        lf_w = self._robot_scene.data.body_pos_w[:, self._lf_id]
        rf_w = self._robot_scene.data.body_pos_w[:, self._rf_id]
        return ((lf_w + rf_w) / 2.0 - self.push_env.env.scene.env_origins).clone()

    def reset(self) -> torch.Tensor:
        obs = self.push_env.reset()
        self._sync_ee_state()
        self._close_gripper()
        self.push_env.capture_pre_push(obs)
        return obs

    def _close_gripper(self):
        act = torch.zeros(self.num_envs, self.push_env.action_space.shape[0], device=self.device)
        act[:, :6] = self._robot_scene.data.joint_pos[:, self._arm_jids]
        act[:, 6] = -1.0
        self.push_env.step(act)

    def _sync_ee_state(self):
        self._ee_pos_local = self._tcp_pos_local()
        self._ee_quat_w = self._quat_tool_down.expand(self.num_envs, 4).clone()
        self._prev_joint_cmd = self._robot_scene.data.joint_pos[:, self._arm_jids].clone()

    def step(self, action: torch.Tensor):
        """
        Execute one full push macro-action.

        Args:
            action: (num_envs, 4) continuous values in [-1, 1]

        Returns:
            obs: (num_envs, obs_dim)
            reward: (num_envs,)
            terminated: (num_envs,) bool
            truncated: (num_envs,) bool
            info: dict with diagnostics
        """
        from curobo.types.math import Pose as CuroboPose

        obs_pre = self.push_env._get_push_obs()
        self.push_env.capture_pre_push(obs_pre)

        if self.rel_act:
            obj_x = obs_pre[:, self.robot_dim]
            obj_y = obs_pre[:, self.robot_dim + 1]
            obj_yaw = obs_pre[:, self.robot_dim + 5]
            obj_xy = torch.stack([obj_x, obj_y], dim=-1)
            Xs, Ys, length, theta = decode_push_action_relative_continuous(
                action, obj_xy, obj_yaw,
            )
        else:
            Xs, Ys, length, theta = decode_push_action_continuous(action)

        Xf = Xs + length * torch.cos(theta)
        Yf = Ys + length * torch.sin(theta)

        waypoints = compute_push_waypoints(
            Xs=Xs, Ys=Ys, length=length, theta=theta,
            current_ee_pos=self._ee_pos_local,
            current_ee_quat=self._ee_quat_w,
            device=self.device,
        )

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for wp_pos, wp_quat, _wp_grip in waypoints:
            ik_target = wp_pos - self._ik_error
            ik_target[:, 0].clamp_(*_WS_X)
            ik_target[:, 1].clamp_(*_WS_Y)
            ik_target[:, 2].clamp_(*_WS_Z)

            result = self.ik_solver.solve_batch(
                CuroboPose(position=ik_target, quaternion=wp_quat),
                seed_config=self._prev_joint_cmd.unsqueeze(1),
                retract_config=self._prev_joint_cmd,
            )

            ik_ok = result.success.squeeze(-1)
            cur_joints = self._robot_scene.data.joint_pos[:, self._arm_jids]

            self._total_ik_steps += self.num_envs
            self._total_ik_fails += int((~ik_ok).sum().item())

            solved = result.solution.view(self.num_envs, 6)
            elbow_bad = solved[:, 2] < 0.0
            if elbow_bad.any():
                ik_ok[elbow_bad] = False

            raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, self._prev_joint_cmd)
            if terminated.any():
                raw_cmd[terminated] = cur_joints[terminated]
            self._prev_joint_cmd = raw_cmd.detach().clone()

            env_full = torch.zeros(self.num_envs, self.push_env.action_space.shape[0], device=self.device)
            env_full[:, :6] = raw_cmd
            env_full[:, 6] = -1.0

            obs, _, step_terminated, truncated, _ = self.push_env.step(env_full)
            terminated |= step_terminated

        reward = self.push_env.compute_push_reward(obs)
        reward[terminated] = -10.0

        done = self.push_env.check_done(obs, terminated)

        info = {
            "pos_err": self.push_env._last_pos_err.clone(),
            "rot_err": self.push_env._last_rot_err.clone(),
            "at_goal": self.push_env.at_goal.clone(),
            "Xs": Xs, "Ys": Ys, "Xf": Xf, "Yf": Yf,
            "length": length, "theta": theta,
        }

        if done.any():
            self._handle_resets(done, obs)

        self._ee_pos_local = self._tcp_pos_local()
        self._ee_quat_w = self._quat_tool_down.expand(self.num_envs, 4).clone()

        truncated_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, reward, done, truncated_out, info

    def _handle_resets(self, done: torch.Tensor, obs: torch.Tensor):
        done_ids = torch.where(done)[0]
        self.push_env.reset_done_envs(done)

        self.push_env.env.reset(env_ids=done_ids)
        self.push_env._randomize_object_spawn(done_ids)
        self.push_env._sample_goals_filtered(done_ids)
        self.push_env._update_goal_in_extras()
        self.push_env._move_goal_ghost(done_ids)
        self.push_env._ep_started[done_ids] = False

        obs_new = self.push_env._get_push_obs()
        obs[done] = obs_new[done]

        self._ee_pos_local[done] = self._tcp_pos_local()[done]
        self._ee_quat_w[done] = self._quat_tool_down.expand(done.sum().item(), 4).to(self.device)
        self._prev_joint_cmd[done] = self._robot_scene.data.joint_pos[:, self._arm_jids][done]

    def get_ik_fail_rate(self) -> float:
        if self._total_ik_steps == 0:
            return 0.0
        return self._total_ik_fails / self._total_ik_steps

    def reset_ik_counters(self):
        self._total_ik_fails = 0
        self._total_ik_steps = 0
