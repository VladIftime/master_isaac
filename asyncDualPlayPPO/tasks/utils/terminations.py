"""Termination functions for dual-arm environment - table boundary checks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg_left: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_3_link"),
    asset_cfg_right: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_wrist_3_link"),
    table_z: float = 0.0,
    margin: float = -0.05,  # Negative allows touching table (terminate if Z < -0.05)
    x_range: tuple[float, float] = (-0.8, 0.8),  # Wider than IK bounds [-0.6, 0.6]
    y_range: tuple[float, float] = (-0.8, 0.8),  # Wider than IK bounds
) -> torch.Tensor:
    """
    Terminate episode if robot leaves the safe workspace (Table + Buffer Zone).
    
    Args:
        env: The RL environment
        asset_cfg_left: Left robot configuration
        asset_cfg_right: Right robot configuration
        table_z: Table surface Z height
        margin: Z-axis safety margin. Negative value allows penetration/touching.
                Example: -0.05 means terminate if Z < -0.05m (allows touching table at Z=0).
        x_range: Safe X range (local env coordinates). Should be wider than IK action bounds.
        y_range: Safe Y range (local env coordinates). Should be wider than IK action bounds.
    
    Returns:
        Boolean tensor indicating which environments should terminate
    """
    # Get robots
    robot_left = env.scene[asset_cfg_left.name]
    robot_right = env.scene[asset_cfg_right.name]
    
    # Resolve body indices (safely handle unified robot)
    # Default to wrist links if body_names not provided in config
    left_body = asset_cfg_left.body_names if asset_cfg_left.body_names is not None else "left_wrist_3_link"
    right_body = asset_cfg_right.body_names if asset_cfg_right.body_names is not None else "right_wrist_3_link"
    
    left_ids, _ = robot_left.find_bodies(left_body)
    right_ids, _ = robot_right.find_bodies(right_body)
    
    # Get end-effector Global positions
    left_ee_pos_w = robot_left.data.body_pos_w[:, left_ids[0], :]
    right_ee_pos_w = robot_right.data.body_pos_w[:, right_ids[0], :]
    
    # Convert to LOCAL positions (Critical for parallel envs)
    left_ee_pos = left_ee_pos_w - env.scene.env_origins
    right_ee_pos = right_ee_pos_w - env.scene.env_origins

    # 1. Check Z-Height (Table Safety)
    # With margin=-0.05: terminate if z < 0.0 + (-0.05) = -0.05
    # This allows touching table surface (Z=0.0) without terminating
    left_z_out = left_ee_pos[:, 2] < (table_z + margin)
    right_z_out = right_ee_pos[:, 2] < (table_z + margin)
    
    # 2. Check X/Y Bounds (Workspace Safety)
    # These bounds should be WIDER than IK action bounds to allow for momentum
    left_x_out = (left_ee_pos[:, 0] < x_range[0]) | (left_ee_pos[:, 0] > x_range[1])
    left_y_out = (left_ee_pos[:, 1] < y_range[0]) | (left_ee_pos[:, 1] > y_range[1])
    
    right_x_out = (right_ee_pos[:, 0] < x_range[0]) | (right_ee_pos[:, 0] > x_range[1])
    right_y_out = (right_ee_pos[:, 1] < y_range[0]) | (right_ee_pos[:, 1] > y_range[1])

    # Combine all checks
    out_of_bounds = (
        left_z_out | right_z_out | 
        left_x_out | left_y_out | 
        right_x_out | right_y_out
    )

    if out_of_bounds.any():
        print(f"[Termination] Robot out of bounds: {out_of_bounds.sum().item()} envs")
    
    return out_of_bounds


def objects_out_of_bounds(
    env: ManagerBasedRLEnv,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-0.5, 1.5),
    z_min: float = -0.2,
) -> torch.Tensor:
    """
    Terminate episode if any object leaves the table workspace.
    
    Args:
        env: The RL environment
        x_range: Valid X range for objects
        y_range: Valid Y range for objects
        z_min: Minimum Z height (objects falling off table)
    
    Returns:
        Boolean tensor indicating which environments should terminate
    """
    object_names = ["target_object", "cube", "cylinder", "rect", "triangle"]
    
    out_of_bounds = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    for obj_name in object_names:
        try:
            obj = env.scene[obj_name]
            pos_w = obj.data.root_pos_w
            pos_local = pos_w - env.scene.env_origins
            
            x_out = (pos_local[:, 0] < x_range[0]) | (pos_local[:, 0] > x_range[1])
            y_out = (pos_local[:, 1] < y_range[0]) | (pos_local[:, 1] > y_range[1])
            z_out = pos_w[:, 2] < z_min 
            
            out_of_bounds = out_of_bounds | x_out | y_out | z_out
            
        except KeyError:
            pass
    
    if out_of_bounds.any():
        print(f"[Termination] Objects out of bounds: {out_of_bounds.sum().item()} envs")

    return out_of_bounds
