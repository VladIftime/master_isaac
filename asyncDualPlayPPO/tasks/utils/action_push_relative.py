"""
Object-relative push primitive action decode for Push-ASP.

Guarantees contact by parameterizing approach offset relative to the object's
current position.  Push direction is world-frame (decoupled from approach angle)
for easy translation learning.

The waypoint generator (compute_push_waypoints in action_push.py) is unchanged —
this module only replaces the DECODE step that converts bin indices to (Xs, Ys,
length, theta) in world coordinates.

Action space (single object, 4D × 21 bins):
  dim 0: r     — radial offset from object center  [min_r, max_r] m
  dim 1: φ     — approach angle in object's frame   [-π, π] rad
  dim 2: length — push distance                      [0.0, MAX_LEN] m
  dim 3: θ     — push direction in world frame      [-π, π] rad

Conversion to world coordinates:
  world_angle = obj_yaw + φ
  Xs = obj_x + r × cos(world_angle)
  Ys = obj_y + r × sin(world_angle)
  Xf = Xs + length × cos(θ)
  Yf = Ys + length × sin(θ)
"""

import math
from typing import Tuple

import torch

# ── Centralised approach-radius constants — single source of truth ──────────
# Every caller of decode_push_action_relative() MUST use these (or the Disc
# variants below), not bare literals.  The function-signature defaults mirror
# TBLOCK_* so that callers without explicit arguments still get the right range.
TBLOCK_MIN_R = 0.04   # Isaac T-block  reaches ~0.102 m from near-centre origin
TBLOCK_MAX_R = 0.12
DISC_MIN_R   = 0.06   # disc  radius 0.05 m; min_r keeps gripper 1 cm outside
DISC_MAX_R   = 0.12
MAX_LEN      = 0.20   # maximum push length in metres


def decode_push_action_relative(
    bin_indices: torch.Tensor,
    obj_xy: torch.Tensor,
    obj_yaw: torch.Tensor,
    num_bins: int = 21,
    min_r: float = TBLOCK_MIN_R,
    max_r: float = TBLOCK_MAX_R,
    max_len: float = MAX_LEN,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode 4D bins → world-frame push parameters using object pose.

    Args:
        bin_indices: (N, 4) integer bin indices from MultiCategorical policy.
        obj_xy: (N, 2) object XY position in local frame (from observation).
        obj_yaw: (N,) object yaw angle in radians (from observation).
        num_bins: number of bins per dimension.
        min_r: minimum approach offset from object center (metres).
        max_r: maximum approach offset from object center (metres).
        max_len: maximum push length (metres).

    Returns:
        Xs, Ys, length, theta — same interface as decode_push_action(),
        ready to pass directly to compute_push_waypoints().
    """
    center = (num_bins - 1) / 2.0

    r_norm = bin_indices[:, 0].float() / (num_bins - 1)
    r = min_r + r_norm * (max_r - min_r)

    phi_norm = (bin_indices[:, 1].float() - center) / center
    phi = phi_norm * math.pi

    len_norm = bin_indices[:, 2].float() / (num_bins - 1)
    length = (len_norm * max_len).clamp(0.0, max_len)

    theta_norm = (bin_indices[:, 3].float() - center) / center
    theta = theta_norm * math.pi

    world_angle = obj_yaw + phi
    Xs = obj_xy[:, 0] + r * torch.cos(world_angle)
    Ys = obj_xy[:, 1] + r * torch.sin(world_angle)

    return Xs, Ys, length, theta


def build_bob_relative_goal(
    obj_xy: torch.Tensor,
    obj_yaw: torch.Tensor,
    goal_xy: torch.Tensor,
    goal_yaw: torch.Tensor,
) -> torch.Tensor:
    """
    Compute relative goal features for Bob's observation.

    Args:
        obj_xy: (N, 2) current object XY position.
        obj_yaw: (N,) current object yaw.
        goal_xy: (N, 2) goal XY position.
        goal_yaw: (N,) goal yaw.

    Returns:
        (N, 5) tensor: [delta_x, delta_y, rel_yaw, pos_dist, rot_dist]
          - delta_x/y: world-frame displacement to goal (aligns with world-frame θ)
          - rel_yaw: wrapped to [-π, π], sign = shortest rotation direction
          - pos_dist: L2 XY distance to goal
          - rot_dist: |rel_yaw|
    """
    delta_xy = goal_xy - obj_xy

    rel_yaw = goal_yaw - obj_yaw
    rel_yaw = torch.atan2(torch.sin(rel_yaw), torch.cos(rel_yaw))

    pos_dist = delta_xy.norm(dim=-1, keepdim=True)
    rot_dist = rel_yaw.abs().unsqueeze(-1)

    return torch.cat([delta_xy, rel_yaw.unsqueeze(-1), pos_dist, rot_dist], dim=-1)
