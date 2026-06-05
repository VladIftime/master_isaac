"""Ping pong environment — full game logic ported from Isaaclab-TableTennisRobot.

Extends ManagerBasedRLEnv with:
  - Virtual paddle-ball contact detection (distance threshold)
  - Table zone scoring (opponent vs own halves)
  - Latched boolean game-state tracking
  - Rackets are fixed child links of each arm's end-effector
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply


class PingPongEnv(ManagerBasedRLEnv):
    """Ping pong environment with dual-arm robots and full game logic."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self._racket_offset = torch.tensor([0.0, 0.265, 0.0], device=self.device)

        self.has_touch_paddle_A = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_touch_paddle_B = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_touch_own_table_prev_A = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.has_touch_own_table_prev_B = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.has_first_bounce_A = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_first_bounce_B = torch.zeros(self.num_envs, device=self.device).bool()
        self.has_first_bounce_prev_A = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.has_first_bounce_prev_B = torch.zeros(
            self.num_envs, device=self.device
        ).bool()
        self.reward_vel_prev_A = torch.zeros(self.num_envs, device=self.device)
        self.reward_vel_prev_B = torch.zeros(self.num_envs, device=self.device)

        self._contact_A = torch.zeros(self.num_envs, device=self.device)
        self._contact_B = torch.zeros(self.num_envs, device=self.device)
        self._table_success_A = torch.zeros(self.num_envs, device=self.device)
        self._table_success_B = torch.zeros(self.num_envs, device=self.device)
        self._table_fail_A = torch.zeros(self.num_envs, device=self.device)
        self._table_fail_B = torch.zeros(self.num_envs, device=self.device)
        self._ball_floor = torch.zeros(self.num_envs, device=self.device).bool()
        self._velocity_A = torch.zeros(self.num_envs, device=self.device)
        self._velocity_B = torch.zeros(self.num_envs, device=self.device)
        self._ball_pos_rw_A = torch.zeros(self.num_envs, device=self.device)
        self._ball_pos_rw_B = torch.zeros(self.num_envs, device=self.device)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        self._compute_intermediate_values()

        return obs, reward, terminated, truncated, info

    def _compute_intermediate_values(self):
        """Compute all game-state tensors from current physics state.

        Ported from Isaaclab-TableTennisRobot's _compute_intermediate_values,
        adapted for two robots facing each other.
        """
        wrist = f"{self.cfg.playing_arm_side}_wrist_3_link"

        ball = self.scene["ball"]
        self.ball_global_pos = ball.data.root_pos_w
        self.ball_pos = ball.data.root_pos_w - self.scene.env_origins
        self.ball_linvel = ball.data.root_lin_vel_w

        self._paddle_contact_detection(
            "robot_A", wrist, "has_touch_paddle_A", "_contact_A"
        )
        self._paddle_contact_detection(
            "robot_B", wrist, "has_touch_paddle_B", "_contact_B"
        )

        self._table_zone_detection()

    def _paddle_contact_detection(
        self, robot_name, wrist_link, paddle_flag, contact_attr
    ):
        """Detect whether the ball is near the playing arm's paddle.

        Uses virtual distance threshold (no contact sensors).
        Updates the latched boolean flag and continuous contact score.
        """
        robot = self.scene[robot_name]
        body_ids, _ = robot.find_bodies([wrist_link])
        paddle_pos = robot.data.body_pos_w[:, body_ids[0], :]
        paddle_quat = robot.data.body_quat_w[:, body_ids[0], :]
        paddle_quat = paddle_quat / paddle_quat.norm(dim=1, keepdim=True)

        offset = self._racket_offset.unsqueeze(0).expand_as(paddle_pos)
        rotated_offset = quat_apply(paddle_quat, offset)
        touch_point = paddle_pos + rotated_offset

        distance = torch.norm(self.ball_global_pos - touch_point, dim=1)
        contact_score = (
            self.cfg.contact_threshold - distance
        ) / self.cfg.contact_threshold
        contact = torch.clamp(contact_score, min=0.0, max=1.0)

        paddle_tensor = getattr(self, paddle_flag)
        contact = contact * ~paddle_tensor
        new_hits = contact_score > 0
        still_false = ~paddle_tensor
        paddle_tensor[still_false] = new_hits[still_false]

        setattr(self, contact_attr, contact)

    def _table_zone_detection(self):
        """Check which table zones the ball occupies.

        Zone layout (env-local coords, Y+ = robot B direction):
          Negative Y zone [-1.35, -0.1]: Robot A's OWN zone, Robot B's OPPONENT zone
          Positive Y zone [0, 1.36]:    Robot B's OWN zone, Robot A's OPPONENT zone
        """
        bx, by, bz = self.ball_pos[:, 0], self.ball_pos[:, 1], self.ball_pos[:, 2]

        x_min, x_max = self.cfg.table_zone_x
        z_min, z_max = self.cfg.table_zone_z
        neg_y_min, neg_y_max = self.cfg.table_zone_neg_y
        pos_y_min, pos_y_max = self.cfg.table_zone_pos_y

        in_x = (bx >= x_min) & (bx <= x_max)
        in_z = (bz >= z_min) & (bz <= z_max)

        in_neg_y = (by >= neg_y_min) & (by <= neg_y_max)
        in_pos_y = (by >= pos_y_min) & (by <= pos_y_max)

        in_zone_neg = in_x & in_z & in_neg_y
        in_zone_pos = in_x & in_z & in_pos_y

        # Robot A: own = neg Y zone, opponent = pos Y zone
        touch_own_just_now_A = in_zone_neg & (~self.has_touch_own_table_prev_A)
        self._table_success_A = self.has_touch_paddle_A.float() * in_zone_pos.float()
        self.has_touch_own_table_prev_A = (
            self.has_touch_own_table_prev_A | touch_own_just_now_A
        )
        self.has_first_bounce_prev_A = self.has_first_bounce_A.clone()
        self.has_first_bounce_A[~self.has_first_bounce_A] = touch_own_just_now_A[
            ~self.has_first_bounce_A
        ]
        self._table_fail_A = (
            self.has_first_bounce_prev_A.float() * touch_own_just_now_A.float()
        )
        fail_mask_a = self._table_fail_A != 0
        self._table_fail_A[fail_mask_a] += by[fail_mask_a] + 0.1

        # Robot B: own = pos Y zone, opponent = neg Y zone
        touch_own_just_now_B = in_zone_pos & (~self.has_touch_own_table_prev_B)
        self._table_success_B = self.has_touch_paddle_B.float() * in_zone_neg.float()
        self.has_touch_own_table_prev_B = (
            self.has_touch_own_table_prev_B | touch_own_just_now_B
        )
        self.has_first_bounce_prev_B = self.has_first_bounce_B.clone()
        self.has_first_bounce_B[~self.has_first_bounce_B] = touch_own_just_now_B[
            ~self.has_first_bounce_B
        ]
        self._table_fail_B = (
            self.has_first_bounce_prev_B.float() * touch_own_just_now_B.float()
        )
        fail_mask_b = self._table_fail_B != 0
        self._table_fail_B[fail_mask_b] += by[fail_mask_b] + 0.1

        self._ball_floor = self.ball_pos[:, 2] < 0.65

        has_contact_a = self._contact_A > 0
        self._velocity_A = (
            -self.ball_linvel[:, 1]
            * self.has_touch_paddle_A.float()
            * (self._contact_A == 0).float()
            * torch.logical_not(self.reward_vel_prev_A > 0)
        )
        self._velocity_A[~has_contact_a] = 0.0
        still_zero_a = self.reward_vel_prev_A == 0
        self.reward_vel_prev_A[still_zero_a] = self._velocity_A[still_zero_a]

        has_contact_b = self._contact_B > 0
        self._velocity_B = (
            self.ball_linvel[:, 1]
            * self.has_touch_paddle_B.float()
            * (self._contact_B == 0).float()
            * torch.logical_not(self.reward_vel_prev_B > 0)
        )
        self._velocity_B[~has_contact_b] = 0.0
        still_zero_b = self.reward_vel_prev_B == 0
        self.reward_vel_prev_B[still_zero_b] = self._velocity_B[still_zero_b]

        self._ball_pos_rw_A = -self._table_success_A * self.ball_pos[:, 1]
        self._ball_pos_rw_B = self._table_success_B * self.ball_pos[:, 1]

    def _reset_game_state(self, env_ids):
        """Reset game-state tracking tensors for specified environments."""
        self.has_touch_paddle_A[env_ids] = False
        self.has_touch_paddle_B[env_ids] = False
        self.has_touch_own_table_prev_A[env_ids] = False
        self.has_touch_own_table_prev_B[env_ids] = False
        self.has_first_bounce_A[env_ids] = False
        self.has_first_bounce_B[env_ids] = False
        self.has_first_bounce_prev_A[env_ids] = False
        self.has_first_bounce_prev_B[env_ids] = False
        self.reward_vel_prev_A[env_ids] = 0.0
        self.reward_vel_prev_B[env_ids] = 0.0
        self._contact_A[env_ids] = 0.0
        self._contact_B[env_ids] = 0.0
        self._table_success_A[env_ids] = 0.0
        self._table_success_B[env_ids] = 0.0
        self._table_fail_A[env_ids] = 0.0
        self._table_fail_B[env_ids] = 0.0
        self._ball_floor[env_ids] = False
        self._velocity_A[env_ids] = 0.0
        self._velocity_B[env_ids] = 0.0
        self._ball_pos_rw_A[env_ids] = 0.0
        self._ball_pos_rw_B[env_ids] = 0.0
