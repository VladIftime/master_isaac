"""Termination conditions for throwing environment.

Episode terminates on:
  - Time limit (5 seconds)
  - Object flies out of bounds
  - Object settles (low velocity for N consecutive steps after release)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .throwing_env import ThrowingEnv


def object_out_of_bounds(
    env: "ThrowingEnv",
    x_range: tuple = (-3.0, 3.0),
    y_range: tuple = (-3.0, 3.0),
    z_min: float = -0.2,
) -> torch.Tensor:
    """Terminate when the thrown object leaves the play area."""
    obj = env.scene["milk"]
    dev = obj.data.root_pos_w.device
    pos_local = obj.data.root_pos_w - env.scene.env_origins.to(dev)

    below_floor = pos_local[:, 2] < z_min
    out_x = (pos_local[:, 0] < x_range[0]) | (pos_local[:, 0] > x_range[1])
    out_y = (pos_local[:, 1] < y_range[0]) | (pos_local[:, 1] > y_range[1])
    return below_floor | out_x | out_y


def object_settled(
    env: "ThrowingEnv",
    vel_threshold: float = 0.05,
    settle_steps: int = 30,
) -> torch.Tensor:
    """Terminate when object velocity stays below threshold for N steps after release."""
    obj = env.scene["milk"]
    vel_norm = torch.norm(obj.data.root_lin_vel_w, dim=-1)

    is_settling = (vel_norm < vel_threshold) & env._released
    env._object_settled_count = torch.where(
        is_settling, env._object_settled_count + 1, torch.zeros_like(env._object_settled_count)
    )
    return env._object_settled_count >= settle_steps
