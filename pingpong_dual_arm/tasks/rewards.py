"""Reward functions for competitive ping pong — ported from Isaaclab-TableTennisRobot.

All heavy computation is done in PingPongEnv._compute_intermediate_values().
These functions are lightweight accessors returning precomputed reward tensors.

Reward structure (per robot):
  - paddle_contact: continuous 0-1 proximity bonus
  - velocity: one-time bonus for fast ball return speed
  - table_success: +5 when ball reaches opponent's table half after paddle contact
  - table_fail: -2 (augmented by ball Y) when ball hits own table half after bounce
  - ball_floor: -3.5 when ball drops below table height
  - ball_pos: position shaping proportional to forward progress on success
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pingpong_env import PingPongEnv


def paddle_contact_reward_A(env: "PingPongEnv") -> torch.Tensor:
    """Continuous 0-1 proximity reward for robot A's paddle nearing the ball."""
    return env._contact_A


def paddle_contact_reward_B(env: "PingPongEnv") -> torch.Tensor:
    """Continuous 0-1 proximity reward for robot B's paddle nearing the ball."""
    return env._contact_B


def table_success_reward_A(env: "PingPongEnv") -> torch.Tensor:
    """+5 when ball reaches robot B's table half after robot A's paddle contact."""
    return env._table_success_A


def table_success_reward_B(env: "PingPongEnv") -> torch.Tensor:
    """+5 when ball reaches robot A's table half after robot B's paddle contact."""
    return env._table_success_B


def table_fail_reward_A(env: "PingPongEnv") -> torch.Tensor:
    """Penalty when ball hits robot A's own table half after bouncing from B's side."""
    return env._table_fail_A


def table_fail_reward_B(env: "PingPongEnv") -> torch.Tensor:
    """Penalty when ball hits robot B's own table half after bouncing from A's side."""
    return env._table_fail_B


def ball_floor_penalty(env: "PingPongEnv") -> torch.Tensor:
    """Penalty when ball drops below table height (z < 0.65)."""
    return env._ball_floor.float()


def velocity_reward_A(env: "PingPongEnv") -> torch.Tensor:
    """One-time bonus for fast ball velocity toward opponent after A's contact."""
    return env._velocity_A


def velocity_reward_B(env: "PingPongEnv") -> torch.Tensor:
    """One-time bonus for fast ball velocity toward opponent after B's contact."""
    return env._velocity_B


def ball_pos_reward_A(env: "PingPongEnv") -> torch.Tensor:
    """Position shaping: reward proportional to forward ball progress on success."""
    return env._ball_pos_rw_A


def ball_pos_reward_B(env: "PingPongEnv") -> torch.Tensor:
    """Position shaping: reward proportional to forward ball progress on success."""
    return env._ball_pos_rw_B
