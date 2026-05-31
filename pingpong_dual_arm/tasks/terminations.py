"""Termination conditions for ping pong environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_out_of_bounds(
    env: ManagerBasedRLEnv,
    z_min: float = 0.0,
    x_range: tuple[float, float] = (-1.2, 1.2),
    y_range: tuple[float, float] = (-1.4, 1.4),
) -> torch.Tensor:
    """Terminate when the ball leaves the play area."""
    ball = env.scene["ball"]
    pos_local = ball.data.root_pos_w - env.scene.env_origins.to(ball.data.root_pos_w.device)

    below_table = pos_local[:, 2] < z_min
    out_x = (pos_local[:, 0] < x_range[0]) | (pos_local[:, 0] > x_range[1])
    out_y = (pos_local[:, 1] < y_range[0]) | (pos_local[:, 1] > y_range[1])
    return below_table | out_x | out_y


def robot_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot_A", body_names=["right_wrist_3_link"]),
    table_z: float = 0.0,
    margin: float = -0.1,
    x_range: tuple[float, float] = (-1.2, 1.2),
    y_range: tuple[float, float] = (-1.4, 1.4),
) -> torch.Tensor:
    """Terminate when the end-effector leaves the safe workspace."""
    robot = env.scene[asset_cfg.name]
    body_names = asset_cfg.body_names or ["right_wrist_3_link"]
    body_ids, _ = robot.find_bodies(body_names)
    ee_pos = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins.to(robot.data.body_pos_w.device)

    z_violation = ee_pos[:, 2] < table_z + margin
    x_violation = (ee_pos[:, 0] < x_range[0]) | (ee_pos[:, 0] > x_range[1])
    y_violation = (ee_pos[:, 1] < y_range[0]) | (ee_pos[:, 1] > y_range[1])
    return z_violation | x_violation | y_violation


def point_scored(
    env: ManagerBasedRLEnv,
    y_left_range: tuple[float, float] = (-1.4, -0.5),
    y_right_range: tuple[float, float] = (0.5, 1.4),
    z_min: float = 0.0,
) -> torch.Tensor:
    """Detect when a point has been scored."""
    ball = env.scene["ball"]
    pos_local = ball.data.root_pos_w - env.scene.env_origins.to(ball.data.root_pos_w.device)
    vel_local = ball.data.root_lin_vel_w

    falling = vel_local[:, 2] < -0.1
    crossed_left = (pos_local[:, 1] < y_left_range[0]) & (pos_local[:, 2] < 0.1)
    crossed_right = (pos_local[:, 1] > y_right_range[0]) & (pos_local[:, 2] < 0.1)
    below_table = pos_local[:, 2] < z_min

    point_scored_left = crossed_left & (falling | below_table)
    point_scored_right = crossed_right & (falling | below_table)
    return point_scored_left | point_scored_right
