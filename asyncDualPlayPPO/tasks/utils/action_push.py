"""
Push primitive — waypoint generation for tabletop pushing with cuRobo IK.

Generates a multi-phase trajectory (approach, orient, descend, engage, push,
release, retract) as a list of waypoints.  The training loop iterates over
waypoints and calls cuRobo IK per substep.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch


# ── Default steps-per-phase ────────────────────────────────────────────────────
PUSH_NSTEPS_APPROACH = 18
PUSH_NSTEPS_ENGAGE   = 5
PUSH_NSTEPS_DESCEND  = 27
PUSH_NSTEPS_PUSH     = 30
PUSH_NSTEPS_RETRACT  = 10
PUSH_NSTEPS_RELEASE  = 2
PUSH_NSTEPS_RETURN   = 12
# Total: 18 + 5 + 20 + 30 + 10 + 5 + 12 = 100 substeps per push

# ── Fixed heights (relative to env origin, local frame) ────────────────────────
PUSH_APPROACH_HEIGHT = 0.15   # Z height for approach / retract (above table)
PUSH_TABLE_SURFACE = 0.00     # table surface Z in local frame (object sits on this)


# Tool-down base quaternion (wxyz): 180° about X-axis = pointing vertically downward
_QUAT_TOOL_DOWN = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def compute_push_waypoints(
    offset_x: torch.Tensor,   # (N,)  approach offset X from object
    offset_y: torch.Tensor,   # (N,)  approach offset Y from object
    push_dx: torch.Tensor,    # (N,)  push delta X
    push_dy: torch.Tensor,    # (N,)  push delta Y
    push_dz: torch.Tensor,    # (N,)  push delta Z
    obj_pos: torch.Tensor,    # (N,3) object position (local frame)
    current_ee_pos: torch.Tensor,  # (N,3) current EE position (local frame, TCP)
    current_ee_quat: torch.Tensor, # (N,4) current EE orientation (wxyz)
    device: torch.device,
    approach_height: float = PUSH_APPROACH_HEIGHT,
    table_z: float = PUSH_TABLE_SURFACE,
    n_approach: int = PUSH_NSTEPS_APPROACH,
    n_engage: int = PUSH_NSTEPS_ENGAGE,
    n_descend: int = PUSH_NSTEPS_DESCEND,
    n_push: int = PUSH_NSTEPS_PUSH,
    n_retract: int = PUSH_NSTEPS_RETRACT,
    n_release: int = PUSH_NSTEPS_RELEASE,
    n_return: int = PUSH_NSTEPS_RETURN,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate push trajectory waypoints for N environments.

    Returns a list of (position, quaternion, gripper_cmd) tuples.
      - position:  (N, 3) TCP target position in local frame
      - quaternion: (N, 4) TCP target orientation (wxyz) — always tool-down
      - gripper_cmd: (N,)  -1.0 = close, +1.0 = open

    Phases:
      1. Approach: EE→above object (tool-down, gripper open)
      2. Engage:   close gripper at approach height
      3. Descend:  move down to surface contact (gripper closed)
      4. Push:     move in push direction (gripper closed)
      5. Retract:  move up above push target (gripper closed)
      6. Release:  open gripper at approach height
      7. Return:   move back above pre-push start position
    """
    N = offset_x.shape[0]

    q_tool_down = _QUAT_TOOL_DOWN.to(device).expand(N, 4).contiguous()
    target_quat = q_tool_down.clone()

    # ── Key positions ─────────────────────────────────────────────────────
    approach_pos = torch.stack([
        obj_pos[:, 0] + offset_x,
        obj_pos[:, 1] + offset_y,
        torch.full((N,), approach_height, device=device),
    ], dim=-1)

    contact_pos = torch.stack([
        obj_pos[:, 0] + offset_x,
        obj_pos[:, 1] + offset_y,
        torch.full((N,), table_z + 0.04, device=device),  # contact height above table
    ], dim=-1)

    push_target = torch.stack([
        contact_pos[:, 0] + push_dx,
        contact_pos[:, 1] + push_dy,
        contact_pos[:, 2] + push_dz,
    ], dim=-1)

    retract_pos = torch.stack([
        push_target[:, 0],
        push_target[:, 1],
        torch.full((N,), approach_height, device=device),
    ], dim=-1)

    open_g = torch.ones(N, device=device)
    close_g = -torch.ones(N, device=device)

    waypoints: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # ── Phase 1: Approach (current → above object, tool-down) ─────────────
    start_pos = current_ee_pos.clone()
    for i in range(1, n_approach + 1):
        alpha = i / n_approach
        pos = start_pos * (1.0 - alpha) + approach_pos * alpha
        waypoints.append((pos, q_tool_down, open_g.clone()))

    # ── Phase 2: Engage (close gripper at approach height) ────────────────
    for i in range(1, n_engage + 1):
        waypoints.append((approach_pos.clone(), target_quat.clone(), close_g.clone()))

    # ── Phase 3: Descend (approach → contact) ─────────────────────────────
    for i in range(1, n_descend + 1):
        alpha = i / n_descend
        pos = approach_pos * (1.0 - alpha) + contact_pos * alpha
        waypoints.append((pos, target_quat.clone(), close_g.clone()))

    # ── Phase 4: Push (contact → push target) ─────────────────────────────
    for i in range(1, n_push + 1):
        alpha = i / n_push
        pos = contact_pos * (1.0 - alpha) + push_target * alpha
        waypoints.append((pos, target_quat.clone(), close_g.clone()))

    # ── Phase 5: Retract (push target → up) ──────────────────────────────
    for i in range(1, n_retract + 1):
        alpha = i / n_retract
        pos = push_target * (1.0 - alpha) + retract_pos * alpha
        waypoints.append((pos, target_quat.clone(), close_g.clone()))

    # ── Phase 6: Release (open gripper at approach height) ────────────────
    for i in range(1, n_release + 1):
        waypoints.append((retract_pos.clone(), target_quat.clone(), open_g.clone()))

    # ── Phase 7: Return (above push target → above pre-push start) ────────
    home_pos = torch.stack([
        start_pos[:, 0],
        start_pos[:, 1],
        torch.full((N,), approach_height, device=device),
    ], dim=-1)
    for i in range(1, n_return + 1):
        alpha = i / n_return
        pos = retract_pos * (1.0 - alpha) + home_pos * alpha
        waypoints.append((pos, q_tool_down, open_g.clone()))

    return waypoints


def decode_push_action(
    bin_indices: torch.Tensor,   # (N, 6) integer bin indices
    num_bins: int = 11,
    max_offset_xy: float = 0.15,  # max approach offset in XY (meters)
    max_push_delta: float = 0.30, # max push delta in XY (meters)
    max_yaw: float = 3.14159,     # max yaw (radians)
    max_push_dz: float = 0.03,    # max push delta in Z (meters)
) -> torch.Tensor:
    """
    Decode 6D MultiCategorical bin indices → push parameters.

    Returns (N, 6) tensor:
      [offset_x, offset_y, push_dx, push_dy, yaw, push_dz]

    Dim layout:
      0: approach_offset_x  → (bin − center)/center × max_offset_xy
      1: approach_offset_y  → (bin − center)/center × max_offset_xy
      2: push_dx            → (bin − center)/center × max_push_delta
      3: push_dy            → (bin − center)/center × max_push_delta
      4: yaw                → (bin − center)/center × max_yaw
      5: push_dz            → (bin − center)/center × max_push_dz
    """
    center = (num_bins - 1) / 2.0
    normalized = (bin_indices.float() - center) / center  # (N, 6), range [-1, 1]

    params = torch.empty_like(normalized)
    params[:, 0] = normalized[:, 0] * max_offset_xy
    params[:, 1] = normalized[:, 1] * max_offset_xy
    params[:, 2] = normalized[:, 2] * max_push_delta
    params[:, 3] = normalized[:, 3] * max_push_delta
    params[:, 4] = normalized[:, 4] * max_yaw
    params[:, 5] = normalized[:, 5] * max_push_dz

    return params


@dataclass
class PushConfig:
    """Configuration for push trajectory generation."""
    max_offset_xy: float = 0.15
    max_push_delta: float = 0.30
    max_yaw: float = 3.14159
    max_push_dz: float = 0.03
    approach_height: float = PUSH_APPROACH_HEIGHT
    table_z: float = PUSH_TABLE_SURFACE
    n_approach: int = PUSH_NSTEPS_APPROACH
    n_engage: int = PUSH_NSTEPS_ENGAGE
    n_descend: int = PUSH_NSTEPS_DESCEND
    n_push: int = PUSH_NSTEPS_PUSH
    n_retract: int = PUSH_NSTEPS_RETRACT
    n_release: int = PUSH_NSTEPS_RELEASE
    n_return: int = PUSH_NSTEPS_RETURN
    num_bins: int = 11
    num_bins: int = 11


def total_push_substeps(cfg: PushConfig = None) -> int:
    """Return total number of substeps per push macro-action."""
    if cfg is None:
        cfg = PushConfig()
    return (cfg.n_approach + cfg.n_engage + cfg.n_descend
            + cfg.n_push + cfg.n_retract + cfg.n_release + cfg.n_return)
