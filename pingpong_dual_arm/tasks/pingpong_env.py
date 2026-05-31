"""Ping pong environment — tracks rackets to end-effector poses each step."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class PingPongEnv(ManagerBasedRLEnv):
    """Environment that positions kinematic rackets at the playing-arm EE."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        for robot_name, racket_name in [("robot_A", "racket_A"), ("robot_B", "racket_B")]:
            try:
                robot = self.scene[robot_name]
                racket = self.scene[racket_name]
            except KeyError:
                continue

            body_ids, _ = robot.find_bodies(["right_wrist_3_link"])
            racket_pose = torch.cat([
                robot.data.body_pos_w[:, body_ids[0]],
                robot.data.body_quat_w[:, body_ids[0]],
            ], dim=-1)
            racket.write_root_pose_to_sim(racket_pose)

        return obs, reward, terminated, truncated, info
