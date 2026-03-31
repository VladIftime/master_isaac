"""
Goal validation utilities for asymmetric self-play.

Validates whether Alice's proposed goals are valid according to paper criteria:
1. At least one object moved
2. All objects still on table
3. Penalty flag if objects outside placement area
"""

import math
import torch


def validate_goal(
    initial_state: torch.Tensor,
    goal_state: torch.Tensor,
    table_bounds: dict,
    placement_bounds: dict,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Validate goals set by Alice.

    Current Configuration:
    - Alice must move at least one object > pos_threshold (0.04m) OR > rot_threshold (0.2rad)
    - All objects must remain on the table.

    Args:
        initial_state: Object states at episode start (batch, num_objects * state_dim)
        goal_state: Object states at end of Alice's phase (batch, num_objects * state_dim)
        table_bounds: Dictionary with x_range, y_range, z_min for table bounds
        placement_bounds: Dictionary with x_range, y_range for valid placement area
        pos_threshold: Minimum position distance for considering an object "moved"
        rot_threshold: Minimum rotation distance for considering an object "moved"

    Returns:
        valid: Boolean tensor (batch,) - True if goal is valid
        rewards: Float tensor (batch,) - Alice's validation reward (+1, -3, or 0)
        reasons: List of strings (batch,) - Human-readable reason for the result
    """
    batch_size = initial_state.shape[0]
    total_dims = initial_state.shape[1]

    # State coming from _extract_object_states is [pos(3)+euler(3)] per object = 6D (Euler)
    # or legacy [pos(3)+quat(4)] = 7D.  Support both plus extended states.
    for state_dim in (6, 7, 13, 15, 17):
        if total_dims % state_dim == 0:
            break
    else:
        state_dim = 6  # fallback: Euler is the new default

    num_objects = total_dims // state_dim

    # Reshape to (batch, num_objects, state_dim)
    initial = initial_state.view(batch_size, num_objects, state_dim)
    goal = goal_state.view(batch_size, num_objects, state_dim)

    # Extract positions
    initial_pos = initial[:, :, :3]
    goal_pos = goal[:, :, :3]

    # --- Rotation distance ---
    if state_dim >= 7:  # quaternion format (legacy)
        initial_quat = initial[:, :, 3:7]
        goal_quat = goal[:, :, 3:7]
        quat_dot = torch.sum(initial_quat * goal_quat, dim=-1)
        rot_movements = 1.0 - torch.abs(quat_dot)
    else:  # 6D Euler format: max absolute angle diff with wrap-around
        initial_euler = initial[:, :, 3:6]
        goal_euler = goal[:, :, 3:6]
        diff = ((goal_euler - initial_euler + math.pi) % (2 * math.pi)) - math.pi
        rot_movements = torch.max(torch.abs(diff), dim=-1)[0]  # (batch, num_objects)

    # --- STEP 1: Check if objects fell off the table (CRITICAL PENALTY) ---
    x_on_table = (goal_pos[:, :, 0] >= table_bounds["x_range"][0]) & (
        goal_pos[:, :, 0] <= table_bounds["x_range"][1]
    )
    y_on_table = (goal_pos[:, :, 1] >= table_bounds["y_range"][0]) & (
        goal_pos[:, :, 1] <= table_bounds["y_range"][1]
    )
    z_on_table = goal_pos[:, :, 2] >= table_bounds["z_min"]
    all_on_table = torch.all(x_on_table & y_on_table & z_on_table, dim=1)  # (batch,)

    # --- STEP 2: Check if objects moved ---
    pos_movements = torch.norm(goal_pos - initial_pos, dim=-1)
    # rot_movements already computed above in the Euler/quat if/else block

    obj_moved = (pos_movements > pos_threshold) | (rot_movements > rot_threshold)
    any_moved = torch.any(obj_moved, dim=1)  # (batch,)

    # --- STEP 3: Check stability ---
    if state_dim >= 13:
        max_lin = torch.max(torch.abs(goal[:, :, 7:10]), dim=-1)[0]
        max_ang = torch.max(torch.abs(goal[:, :, 10:13]), dim=-1)[0]
        obj_stable = (max_lin < 0.5) & (max_ang < 0.5)
        all_stable = torch.all(obj_stable, dim=1)
    else:
        all_stable = torch.ones(
            batch_size, dtype=torch.bool, device=initial_state.device
        )

    # --- STEP 4: Check if outside placement area ---
    x_in_placement = (goal_pos[:, :, 0] >= placement_bounds["x_range"][0]) & (
        goal_pos[:, :, 0] <= placement_bounds["x_range"][1]
    )
    y_in_placement = (goal_pos[:, :, 1] >= placement_bounds["y_range"][0]) & (
        goal_pos[:, :, 1] <= placement_bounds["y_range"][1]
    )
    any_outside_placement = torch.any(
        ~(x_in_placement & y_in_placement), dim=1
    )  # (batch,)

    # --- ASSIGN REWARDS AND REASONS (vectorised) ---
    # Priority: Off Table > Not Moved > Unstable > Out of Zone > Valid
    off_table = ~all_on_table
    not_moved = all_on_table & ~any_moved
    unstable = all_on_table & any_moved & ~all_stable
    out_of_zone = all_on_table & any_moved & all_stable & any_outside_placement
    is_valid = all_on_table & any_moved & all_stable & ~any_outside_placement

    rewards = torch.where(
        off_table,
        torch.full((batch_size,), -3.0, device=initial_state.device),
        torch.where(
            not_moved,
            torch.zeros(batch_size, device=initial_state.device),
            torch.where(
                unstable,
                torch.zeros(batch_size, device=initial_state.device),
                torch.where(
                    out_of_zone,
                    torch.full((batch_size,), -3.0, device=initial_state.device),
                    torch.ones(batch_size, device=initial_state.device),
                ),
            ),
        ),
    )

    valid = is_valid

    reasons = ["Unknown"] * batch_size
    for i in range(batch_size):
        if off_table[i]:
            reasons[i] = "Off Table (-3.0)"
        elif not_moved[i]:
            reasons[i] = "Not Moved (0.0)"
        elif unstable[i]:
            reasons[i] = "Unstable (0.0)"
        elif out_of_zone[i]:
            reasons[i] = "Out of Zone (-3.0)"
        else:
            reasons[i] = "Valid Goal (+1.0)"

    return valid, rewards, reasons
