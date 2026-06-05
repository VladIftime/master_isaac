"""
Push primitive — waypoint generation for tabletop pushing with cuRobo IK.

Generates a multi-phase trajectory (approach, descend, push, retract, return)
as a list of waypoints.  Gripper always closed.  Each waypoint is executed
with cuRobo IK → joint positions → env.step().
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch


# ── Default steps-per-phase ────────────────────────────────────────────────────
PUSH_NSTEPS_APPROACH = 12
PUSH_NSTEPS_DESCEND  = 16
PUSH_NSTEPS_PUSH     = 20
PUSH_NSTEPS_RETRACT  = 16
PUSH_NSTEPS_RETURN   = 8
# Total: 12 + 16 + 20 + 16 + 8 = 72 substeps per push
# Gripper always closed — no engage/release phases

# ── Fixed heights (relative to env origin, local frame) ────────────────────────
PUSH_APPROACH_HEIGHT = 0.50  # Z height for approach / retract (above table)
PUSH_TABLE_SURFACE = 0.00     # table surface Z in local frame (object sits on this)


# Tool-down base quaternion (wxyz): 180° about X-axis = pointing vertically downward
_QUAT_TOOL_DOWN = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def compute_push_waypoints(
    Xs: torch.Tensor,           # (N,) push start X world coords
    Ys: torch.Tensor,           # (N,) push start Y world coords
    length: torch.Tensor,       # (N,) push length
    theta: torch.Tensor,        # (N,) push orientation angle
    current_ee_pos: torch.Tensor,  # (N,3) current EE position (local frame, TCP)
    current_ee_quat: torch.Tensor, # (N,4) current EE orientation (wxyz)
    device: torch.device,
    approach_height: float = PUSH_APPROACH_HEIGHT,
    table_z: float = PUSH_TABLE_SURFACE,
    n_approach: int = PUSH_NSTEPS_APPROACH,
    n_descend: int = PUSH_NSTEPS_DESCEND,
    n_push: int = PUSH_NSTEPS_PUSH,
    n_retract: int = PUSH_NSTEPS_RETRACT,
    n_return: int = PUSH_NSTEPS_RETURN,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate push trajectory waypoints for N environments.

    Returns a list of (position, quaternion, gripper_cmd) tuples.
      - position:  (N, 3) TCP target position in local frame
      - quaternion: (N, 4) TCP target orientation (wxyz) — always tool-down
      - gripper_cmd: (N,)  -1.0 = closed (gripper ALWAYS closed)

    Phases (gripper closed throughout):
      1. Approach: EE → (Xs, Ys, approach_height)
      2. Descend:  (Xs, Ys, approach_height) → contact
      3. Push:     contact → (Xf, Yf, contact_Z)  where Xf=Xs+len·cosθ, Yf=Ys+len·sinθ
      4. Retract:  (Xf, Yf, contact_Z) → (Xf, Yf, approach_height)
      5. Return:   (Xf, Yf, approach_height) → above pre-push start position
    """
    N = Xs.shape[0]
    q_tool_down = _QUAT_TOOL_DOWN.to(device).expand(N, 4).contiguous()

    approach_pos = torch.stack([
        Xs, Ys, torch.full((N,), approach_height, device=device),
    ], dim=-1)

    contact_pos = torch.stack([
        Xs, Ys, torch.full((N,), table_z + 0.110, device=device),
    ], dim=-1)

    Xf = Xs + length * torch.cos(theta)
    Yf = Ys + length * torch.sin(theta)
    push_target = torch.stack([Xf, Yf, contact_pos[:, 2]], dim=-1)

    retract_pos = torch.stack([
        Xf, Yf, torch.full((N,), approach_height, device=device),
    ], dim=-1)

    closed = -torch.ones(N, device=device)
    waypoints: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    start_pos = current_ee_pos.clone()

    # Phase 1: Approach (current → above push start)
    for i in range(1, n_approach + 1):
        alpha = i / n_approach
        pos = start_pos * (1.0 - alpha) + approach_pos * alpha
        waypoints.append((pos, q_tool_down, closed.clone()))

    # Phase 2: Descend (approach → contact)
    for i in range(1, n_descend + 1):
        alpha = i / n_descend
        pos = approach_pos * (1.0 - alpha) + contact_pos * alpha
        waypoints.append((pos, q_tool_down, closed.clone()))

    # Phase 3: Push (contact → push target, tool-down, no yaw)
    for i in range(1, n_push + 1):
        alpha = i / n_push
        pos = contact_pos * (1.0 - alpha) + push_target * alpha
        waypoints.append((pos, q_tool_down, closed.clone()))

    # Phase 4: Retract (push target → up)
    for i in range(1, n_retract + 1):
        alpha = i / n_retract
        pos = push_target * (1.0 - alpha) + retract_pos * alpha
        waypoints.append((pos, q_tool_down, closed.clone()))

    # Phase 5: Return (above push target → above pre-push start)
    home_pos = torch.stack([
        start_pos[:, 0],
        start_pos[:, 1],
        torch.full((N,), approach_height, device=device),
    ], dim=-1)
    for i in range(1, n_return + 1):
        alpha = i / n_return
        pos = retract_pos * (1.0 - alpha) + home_pos * alpha
        waypoints.append((pos, q_tool_down, closed.clone()))

    return waypoints


import math


def decode_push_action(
    bin_indices: torch.Tensor,   # (N, 4) integer bin indices
    num_bins: int = 21,
    max_xs: float = 0.50,
    max_ys: float = 0.225,
    max_len: float = 0.20,
    max_theta: float = math.pi,
):
    """
    Decode 4D MultiCategorical bin indices → push macro-parameters.

    Returns (N,) tensors: Xs, Ys, length, theta.

    Dim layout:
      0: Xs     → push start X world coords [−0.50, 0.50] m
      1: Ys     → push start Y world coords [0.25, 0.70] m
       2: length → push length [0.0, 0.20] m (clamped)
      3: theta  → push orientation [−π, π] rad
    """
    center = (num_bins - 1) / 2.0
    norm = (bin_indices.float() - center) / center

    Xs = norm[:, 0] * max_xs
    Ys = norm[:, 1] * max_ys + 0.475
    length = (norm[:, 2] * max_len).clamp(min=0.0, max=max_len)
    theta = norm[:, 3] * max_theta

    return Xs, Ys, length, theta


@dataclass
class PushConfig:
    """Configuration for push trajectory generation."""
    max_xs: float = 0.50
    max_ys: float = 0.225
    max_len: float = 0.30
    max_theta: float = 3.141592653589793
    approach_height: float = PUSH_APPROACH_HEIGHT
    table_z: float = PUSH_TABLE_SURFACE
    n_approach: int = PUSH_NSTEPS_APPROACH
    n_descend: int = PUSH_NSTEPS_DESCEND
    n_push: int = PUSH_NSTEPS_PUSH
    n_retract: int = PUSH_NSTEPS_RETRACT
    n_return: int = PUSH_NSTEPS_RETURN
    num_bins: int = 21


def total_push_substeps(cfg: "PushConfig | None" = None) -> int:
    """Return total number of substeps per push macro-action."""
    if cfg is None:
        cfg = PushConfig()
    return (cfg.n_approach + cfg.n_descend + cfg.n_push
            + cfg.n_retract + cfg.n_return)
