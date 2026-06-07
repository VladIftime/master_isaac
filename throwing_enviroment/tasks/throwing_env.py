"""Throwing environment — single-arm robot throws object toward a target.

Extends ManagerBasedRLEnv with:
  - Kinematic object attachment (Option A): object follows EE until release
  - Release based on EE velocity threshold
  - Gripper opens at release for visual feedback
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class ThrowingEnv(ManagerBasedRLEnv):
    """Throwing environment with single dual-arm robot, object, and target."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self._side = self.cfg.playing_arm_side
        self._ee_body = f"{self._side}_wrist_3_link"
        self._gripper_joint = f"{'l' if self._side == 'left' else 'r'}gripper_finger_joint"

        self._holding = torch.ones(self.num_envs, device=self.device).bool()
        self._released = torch.zeros(self.num_envs, device=self.device).bool()
        self._object_settled_count = torch.zeros(self.num_envs, device=self.device)
        self._steps_in_episode = torch.zeros(self.num_envs, device=self.device)

        self._object_landed = torch.zeros(self.num_envs, device=self.device).bool()

        self._dist_reward = torch.zeros(self.num_envs, device=self.device)
        self._success_bonus = torch.zeros(self.num_envs, device=self.device)
        self._ee_vel_reward = torch.zeros(self.num_envs, device=self.device)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        self._update_attachment()
        self._compute_rewards()

        return obs, reward, terminated, truncated, info

    def _update_attachment(self):
        """Attach object to EE while holding; release when velocity threshold met."""
        robot = self.scene["robot"]
        milk = self.scene["milk"]

        body_ids, _ = robot.find_bodies([self._ee_body])

        ee_pos = robot.data.body_pos_w[:, body_ids[0]]
        ee_quat = robot.data.body_quat_w[:, body_ids[0]]
        ee_vel = robot.data.body_lin_vel_w[:, body_ids[0]]

        self._steps_in_episode += 1

        vel_norm = torch.norm(ee_vel, dim=-1)
        release_mask = (
            self._holding
            & ~self._released
            & (self._steps_in_episode > self.cfg.release_min_steps)
            & (vel_norm > self.cfg.release_vel_threshold)
        )
        self._released[release_mask] = True
        self._holding[release_mask] = False

        still_holding = self._holding & ~self._released
        if still_holding.any():
            # Kinematic pose write: object follows EE between the finger pads
            bottle_root_offset = torch.tensor(
                [-0.129, -0.012, -0.176], device=ee_pos.device,
            )
            bottle_root = ee_pos[still_holding] + bottle_root_offset.unsqueeze(0)
            ee_pose = torch.cat([bottle_root, ee_quat[still_holding]], dim=-1)
            still_ids = still_holding.nonzero(as_tuple=True)[0]
            milk.write_root_pose_to_sim(ee_pose, env_ids=still_ids)

        if release_mask.any():
            env_ids = release_mask.nonzero(as_tuple=True)[0]
            gripper_ids, _ = robot.find_joints([self._gripper_joint])
            gripper_pos = robot.data.joint_pos[:, gripper_ids].clone()
            gripper_pos[env_ids] = 0.0
            robot.set_joint_position_target(gripper_pos, joint_ids=gripper_ids)

    def _compute_rewards(self):
        """Compute reward tensors from current physics state."""
        milk = self.scene["milk"]
        target = self.scene["target"]

        dev = milk.data.root_pos_w.device
        env_origins = self.scene.env_origins.to(dev)
        milk_pos = milk.data.root_pos_w - env_origins
        target_pos = target.data.root_pos_w - env_origins

        dist_vec = target_pos - milk_pos
        dist = torch.norm(dist_vec, dim=-1)

        robot = self.scene["robot"]
        body_ids, _ = robot.find_bodies([self._ee_body])
        ee_vel = robot.data.body_lin_vel_w[:, body_ids[0]]
        ee_vel_norm = torch.norm(ee_vel, dim=-1)

        self._dist_reward = torch.exp(-(dist**2) / 0.1)

        landed = (dist < 0.15) & self._released & ~self._object_landed
        self._success_bonus = landed.float()
        self._object_landed[landed] = True

        self._ee_vel_reward = ee_vel_norm * self._holding.float()

    def _reset_game_state(self, env_ids):
        """Reset game-state tracking tensors for specified environments."""
        self._holding[env_ids] = True
        self._released[env_ids] = False
        self._object_settled_count[env_ids] = 0
        self._steps_in_episode[env_ids] = 0
        self._object_landed[env_ids] = False
        self._dist_reward[env_ids] = 0.0
        self._success_bonus[env_ids] = 0.0
        self._ee_vel_reward[env_ids] = 0.0
