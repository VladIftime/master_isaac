"""Swing primitive — waypoint generation for table-tennis strokes.

Generates a multi-phase arc trajectory (backswing, forward, contact, follow, return)
as a list of waypoints. Each waypoint carries a target EE position and orientation
(quaternion). Orientation interpolates via SLERP across phases so the racket face
angle changes naturally (neutral at backswing → angled at contact → neutral at follow).

Pattern:
    decode_swing_action  →  macro parameters (contact point, depth, apex, etc.)
    compute_swing_waypoints → [(position, quaternion), ...]
    convert to 6D relative deltas → env.step(action) via configured IK solver
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

SWING_NSTEPS_BACKSWING = 12
SWING_NSTEPS_FORWARD = 16
SWING_NSTEPS_CONTACT = 8
SWING_NSTEPS_FOLLOW = 12
SWING_NSTEPS_RETURN = 12


@dataclass
class SwingConfig:
    max_xs: float = 0.50
    max_ys: float = 0.225
    max_zs: float = 0.30
    max_backswing_depth: float = 0.25
    max_apex_height: float = 0.15
    max_racket_angle: float = 0.7853981633974483
    max_swing_heading: float = 3.141592653589793
    table_height: float = 0.00
    n_backswing: int = SWING_NSTEPS_BACKSWING
    n_forward: int = SWING_NSTEPS_FORWARD
    n_contact: int = SWING_NSTEPS_CONTACT
    n_follow: int = SWING_NSTEPS_FOLLOW
    n_return: int = SWING_NSTEPS_RETURN
    num_bins: int = 21


def total_swing_substeps(cfg: "SwingConfig | None" = None) -> int:
    if cfg is None:
        cfg = SwingConfig()
    return cfg.n_backswing + cfg.n_forward + cfg.n_contact + cfg.n_follow + cfg.n_return


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched spherical linear interpolation between quaternions (wxyz).

    Args:
        q0: (N, 4) start quaternion (w, x, y, z)
        q1: (N, 4) end quaternion (w, x, y, z)
        t: float or (N,) interpolation fraction [0, 1]

    Returns:
        (N, 4) interpolated quaternion
    """
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    flip = dot < 0
    q1 = q1.clone()
    q1[flip.squeeze(-1)] = -q1[flip.squeeze(-1)]
    dot = dot.abs().clamp(max=1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    t = t.unsqueeze(-1) if t.ndim == 1 else t
    mask = so > 1e-6
    result = torch.where(
        mask,
        (torch.sin((1 - t) * omega) / so) * q0 + (torch.sin(t * omega) / so) * q1,
        q0,
    )
    return result


def _racket_quat_for_face_angle(
    N: int, pitch: torch.Tensor, heading: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Compute racket quaternion for a given pitch angle and swing heading.

    The racket starts with its normal facing approximately along the swing direction.
    This is a simplified model:
      - pitch: racket face tilt (0 = vertical, + = open)
      - heading: swing direction in the XY plane
    """
    half_pitch = pitch / 2
    half_heading = heading / 2

    qw = torch.cos(half_pitch) * torch.cos(half_heading)
    qx = torch.sin(half_pitch) * torch.cos(half_heading)
    qy = torch.cos(half_pitch) * torch.sin(half_heading)
    qz = torch.sin(half_pitch) * torch.sin(half_heading)

    return torch.stack([qw, qx, qy, qz], dim=-1)


def compute_swing_waypoints(
    Xs: torch.Tensor,
    Ys: torch.Tensor,
    Zs: torch.Tensor,
    backswing_depth: torch.Tensor,
    apex_height: torch.Tensor,
    racket_pitch: torch.Tensor,
    swing_heading: torch.Tensor,
    current_ee_pos: torch.Tensor,
    current_ee_quat: torch.Tensor,
    device: torch.device,
    cfg: "SwingConfig | None" = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate swing trajectory waypoints for N environments.

    Returns a list of (position, quaternion) tuples.
      - position:  (N, 3) EE target position in local frame
      - quaternion: (N, 4) EE target orientation (wxyz)

    Phases:
      1. Backswing: EE → retracted behind contact point (racket neutral)
      2. Forward:   backswing → apex (above, racket starts closing)
      3. Contact:   apex → contact point (racket full angle)
      4. Follow:    contact → extended forward past contact
      5. Return:    follow → home ready pose
    """
    if cfg is None:
        cfg = SwingConfig()

    N = Xs.shape[0]
    table_z = torch.full((N,), cfg.table_height, device=device)

    contact_pos = torch.stack([Xs, Ys, Zs + table_z], dim=-1)

    backswing_offset_x = -backswing_depth * torch.cos(swing_heading)
    backswing_offset_y = -backswing_depth * torch.sin(swing_heading)
    backswing_pos = torch.stack(
        [
            Xs + backswing_offset_x,
            Ys + backswing_offset_y,
            Zs + table_z,
        ],
        dim=-1,
    )

    apex_pos = torch.stack(
        [
            (contact_pos[:, 0] + backswing_pos[:, 0]) / 2,
            (contact_pos[:, 1] + backswing_pos[:, 1]) / 2,
            Zs + table_z + apex_height,
        ],
        dim=-1,
    )

    follow_offset_x = backswing_depth * 0.4 * torch.cos(swing_heading)
    follow_offset_y = backswing_depth * 0.4 * torch.sin(swing_heading)
    follow_pos = torch.stack(
        [
            Xs + follow_offset_x,
            Ys + follow_offset_y,
            Zs + table_z,
        ],
        dim=-1,
    )

    home_pos = torch.stack(
        [
            current_ee_pos[:, 0],
            current_ee_pos[:, 1],
            Zs + table_z + 0.15,
        ],
        dim=-1,
    )

    q_neutral = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(N, 4).contiguous()
    )
    q_contact = _racket_quat_for_face_angle(N, racket_pitch, swing_heading, device)

    waypoints: List[Tuple[torch.Tensor, torch.Tensor]] = []
    start_pos = current_ee_pos.clone()
    start_quat = current_ee_quat.clone()

    n_back = cfg.n_backswing
    n_fwd = cfg.n_forward
    n_con = cfg.n_contact
    n_fol = cfg.n_follow
    n_ret = cfg.n_return

    for i in range(1, n_back + 1):
        alpha = i / n_back
        pos = start_pos * (1 - alpha) + backswing_pos * alpha
        quat = _slerp(start_quat, q_neutral, torch.tensor(alpha, device=device))
        waypoints.append((pos, quat))

    for i in range(1, n_fwd + 1):
        alpha = i / n_fwd
        pos = backswing_pos * (1 - alpha) + apex_pos * alpha
        quat = _slerp(q_neutral, q_contact, torch.tensor(alpha, device=device))
        waypoints.append((pos, quat))

    for i in range(1, n_con + 1):
        alpha = i / n_con
        pos = apex_pos * (1 - alpha) + contact_pos * alpha
        waypoints.append((pos, q_contact.clone()))

    for i in range(1, n_fol + 1):
        alpha = i / n_fol
        pos = contact_pos * (1 - alpha) + follow_pos * alpha
        quat = _slerp(q_contact, q_neutral, torch.tensor(alpha, device=device))
        waypoints.append((pos, quat))

    for i in range(1, n_ret + 1):
        alpha = i / n_ret
        pos = follow_pos * (1 - alpha) + home_pos * alpha
        waypoints.append((pos, q_neutral.clone()))

    return waypoints


def decode_swing_action(
    bin_indices: torch.Tensor,
    num_bins: int = 21,
    max_xs: float = 0.50,
    max_ys: float = 0.225,
    max_zs: float = 0.30,
    max_backswing_depth: float = 0.25,
    max_apex_height: float = 0.15,
    max_racket_angle: float = 0.7853981633974483,
    max_swing_heading: float = 3.141592653589793,
):
    """Decode 6D MultiCategorical bin indices → swing macro-parameters.

    Returns (N,) tensors: Xs, Ys, Zs, backswing_depth, apex_height,
                           racket_pitch, swing_heading.

    Dim layout:
      0: Xs               → contact X (lateral position across table)
      1: Ys               → contact Y (depth into table)
      2: Zs               → contact Z (height above table)
      3: backswing_depth  → retraction distance behind contact
      4: racket_pitch     → racket face angle at contact
      5: swing_heading    → swing direction in XY plane
    """
    center = (num_bins - 1) / 2.0
    norm = (bin_indices.float() - center) / center

    Xs = norm[:, 0] * max_xs
    Ys = norm[:, 1] * max_ys
    Zs = norm[:, 2].abs() * max_zs + 0.05
    backswing_depth = norm[:, 3].abs() * max_backswing_depth + 0.03
    racket_pitch = norm[:, 4] * max_racket_angle
    swing_heading = norm[:, 5] * max_swing_heading

    return Xs, Ys, Zs, backswing_depth, racket_pitch, swing_heading
