"""Termination conditions for ping pong environment.

Ported from Isaaclab-TableTennisRobot, adapted for two robots:
  - ball_to_floor: ball drops below z=0.65
  - ball_out_of_bounds: ball leaves play area (x/y/z limits)
  - round_end_success: ball reaches opponent's table half after paddle contact
  - round_end_fail: ball hits own table half after bouncing from opponent

Episode terminates on: success, fail, floor, or out-of-bounds.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .pingpong_env import PingPongEnv


def ball_out_of_bounds(
    env: "PingPongEnv",
    z_min: float = -1.0,
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (-3.5, 3.5),
) -> torch.Tensor:
    """Terminate when the ball leaves the play area.

    Wide bounds accommodate the full trajectory between robots at y=±2.7.
    """
    ball = env.scene["ball"]
    pos_local = ball.data.root_pos_w - env.scene.env_origins.to(ball.data.root_pos_w.device)

    below_floor = pos_local[:, 2] < z_min
    out_x = (pos_local[:, 0] < x_range[0]) | (pos_local[:, 0] > x_range[1])
    out_y = (pos_local[:, 1] < y_range[0]) | (pos_local[:, 1] > y_range[1])
    return below_floor | out_x | out_y


def ball_to_floor(env: "PingPongEnv") -> torch.Tensor:
    """Terminate when ball drops below table height (z < 0.65)."""
    return env._ball_floor


def round_end_success_A(env: "PingPongEnv") -> torch.Tensor:
    """Episode ends when robot A successfully hits ball to opponent's table half."""
    return env._table_success_A != 0


def round_end_success_B(env: "PingPongEnv") -> torch.Tensor:
    """Episode ends when robot B successfully hits ball to opponent's table half."""
    return env._table_success_B != 0


def round_end_fail_A(env: "PingPongEnv") -> torch.Tensor:
    """Episode ends when ball lands on robot A's own table half."""
    return env._table_fail_A != 0


def round_end_fail_B(env: "PingPongEnv") -> torch.Tensor:
    """Episode ends when ball lands on robot B's own table half."""
    return env._table_fail_B != 0
