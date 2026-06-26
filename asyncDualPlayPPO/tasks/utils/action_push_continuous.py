"""
Continuous push primitive action decode for SAC.

Maps continuous action values in [-1, 1] to push macro-parameters, matching
the same workspace ranges as the discrete (MultiCategorical) decode functions
in action_push.py and action_push_relative.py.

Two decode modes:
  - Absolute: action → (Xs, Ys, length, theta) in world frame
  - Relative: action → (r, phi, length, theta) → object-relative approach

The waypoint generator (compute_push_waypoints in action_push.py) is unchanged.
"""

import math
from typing import Tuple

import torch


def decode_push_action_continuous(
    action: torch.Tensor,
    max_xs: float = 0.50,
    max_ys: float = 0.225,
    ys_center: float = 0.475,
    max_len: float = 0.20,
    max_theta: float = math.pi,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode continuous 4D action (tanh-squashed, in [-1, 1]) to push parameters.

    Absolute world-frame parameterization matching decode_push_action() ranges.

    Args:
        action: (N, 4) continuous values in [-1, 1] from SAC policy.

    Returns:
        Xs, Ys, length, theta — same interface as decode_push_action().
    """
    Xs = action[:, 0] * max_xs
    Ys = action[:, 1] * max_ys + ys_center
    length = ((action[:, 2] + 1.0) / 2.0 * max_len).clamp(0.0, max_len)
    theta = action[:, 3] * max_theta

    return Xs, Ys, length, theta


def decode_push_action_relative_continuous(
    action: torch.Tensor,
    obj_xy: torch.Tensor,
    obj_yaw: torch.Tensor,
    min_r: float = 0.02,
    max_r: float = 0.08,
    max_len: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode continuous 4D action to push parameters using object-relative approach.

    Matches decode_push_action_relative() ranges.

    Args:
        action: (N, 4) continuous values in [-1, 1] from SAC policy.
        obj_xy: (N, 2) object XY position in local frame.
        obj_yaw: (N,) object yaw angle in radians.

    Returns:
        Xs, Ys, length, theta — world-frame push parameters.
    """
    r = (action[:, 0] + 1.0) / 2.0 * (max_r - min_r) + min_r
    phi = action[:, 1] * math.pi
    length = ((action[:, 2] + 1.0) / 2.0 * max_len).clamp(0.0, max_len)
    theta = action[:, 3] * math.pi

    world_angle = obj_yaw + phi
    Xs = obj_xy[:, 0] + r * torch.cos(world_angle)
    Ys = obj_xy[:, 1] + r * torch.sin(world_angle)

    return Xs, Ys, length, theta
