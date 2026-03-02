"""
Goal validation utilities for asymmetric self-play.

Validates whether Alice's proposed goals are valid according to paper criteria:
1. At least one object moved
2. All objects still on table
3. Penalty flag if objects outside placement area
"""

import torch


def validate_goal(
    initial_state: torch.Tensor,
    goal_state: torch.Tensor,
    table_bounds: dict,
    placement_bounds: dict,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        out_of_bounds_penalty: Boolean tensor (batch,) - True if any object is outside placement area
    """
    batch_size = initial_state.shape[0]
    total_dims = initial_state.shape[1]
    
    # Infer state dim and number of objects
    # Infer state dim and number of objects
    # Supported dims: 7 (pos+rot), 13 (pos+rot+vel), or 17 (extended)
    if total_dims % 17 == 0:
        state_dim = 17
    elif total_dims % 13 == 0:
        state_dim = 13
    elif total_dims % 7 == 0:
        state_dim = 7
    else:
        # Fallback to 7 as default if ambiguous
        state_dim = 7 
        
    num_objects = total_dims // state_dim
    
    # Reshape to (batch, num_objects, state_dim)
    initial = initial_state.view(batch_size, num_objects, state_dim)
    goal = goal_state.view(batch_size, num_objects, state_dim)
    
    # Extract positions and rotations
    initial_pos = initial[:, :, :3]  # (batch, num_objects, 3)
    goal_pos = goal[:, :, :3]
    
    initial_quat = initial[:, :, 3:7]  # (batch, num_objects, 4)
    goal_quat = goal[:, :, 3:7]
    
    # 1. Check if at least one object moved
    # Position distance
    pos_movements = torch.norm(goal_pos - initial_pos, dim=-1)  # (batch, num_objects)
    
    # Rotation distance (1 - |dot(q1, q2)|)
    quat_dot = torch.sum(initial_quat * goal_quat, dim=-1)
    rot_movements = 1.0 - torch.abs(quat_dot)  # (batch, num_objects)
    
    # An object "moved" if either pos or rot exceeds threshold
    obj_moved = (pos_movements > pos_threshold) | (rot_movements > rot_threshold)
    any_moved = torch.any(obj_moved, dim=1)  # (batch,)
    
    # 2. Check if all objects are still on table
    # If even ONE object falls off, the entire goal is invalid.
    x_on_table = (goal_pos[:, :, 0] >= table_bounds["x_range"][0]) & (goal_pos[:, :, 0] <= table_bounds["x_range"][1])
    y_on_table = (goal_pos[:, :, 1] >= table_bounds["y_range"][0]) & (goal_pos[:, :, 1] <= table_bounds["y_range"][1])
    z_on_table = goal_pos[:, :, 2] >= table_bounds["z_min"]
    
    all_on_table = torch.all(x_on_table & y_on_table & z_on_table, dim=1)  # (batch,)
    
    # 3. Check if objects are outside placement area (penalty but still valid)
    x_in_placement = (goal_pos[:, :, 0] >= placement_bounds["x_range"][0]) & (goal_pos[:, :, 0] <= placement_bounds["x_range"][1])
    y_in_placement = (goal_pos[:, :, 1] >= placement_bounds["y_range"][0]) & (goal_pos[:, :, 1] <= placement_bounds["y_range"][1])
    
    any_outside_placement = torch.any(~(x_in_placement & y_in_placement), dim=1)  # (batch,)
    
    # Goal is valid if: at least one object moved AND all objects on table
    valid = any_moved & all_on_table
    
    return valid, any_outside_placement
