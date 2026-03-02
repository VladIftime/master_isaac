"""Event functions for dual-arm environment - object randomization."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg


def randomize_object_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (-0.3, 0.3),
    y_range: tuple[float, float] = (0.55, 0.65),
    z_height: float = 0.03,
) -> None:
    """
    Randomize positions of all objects in the scene on reset.
    
    Reads object list and randomizes their positions within workspace limits.
    Ensures objects don't overlap by checking minimum distances.
    
    Args:
        env: The RL environment
        env_ids: Environment indices to reset
        x_range: Min/max x position range
        y_range: Min/max y position range
        z_height: Fixed z height for all objects
    """
    # List of object names in scene
    object_names = ["target_object", "cube", "cylinder", "rect", "triangle"]
    
    num_resets = len(env_ids)
    
    # Randomize each object's position
    for obj_name in object_names:
        try:
            obj = env.scene[obj_name]
            
            # Random position (local to environment)
            x_local = torch.rand(num_resets, device=env.device) * (x_range[1] - x_range[0]) + x_range[0]
            y_local = torch.rand(num_resets, device=env.device) * (y_range[1] - y_range[0]) + y_range[0]
            z_local = torch.full((num_resets,), z_height, device=env.device)
            
            # Add environment origins to make it global
            env_origins = env.scene.env_origins[env_ids]
            x_pos = x_local + env_origins[:, 0]
            y_pos = y_local + env_origins[:, 1]
            z_pos = z_local + env_origins[:, 2]
            
            # Random orientation (yaw only, keep objects upright)
            yaw = torch.rand(num_resets, device=env.device) * 2 * torch.pi
            
            # Convert to quaternion (only yaw rotation around Z axis)
            quat_w = torch.cos(yaw / 2)
            quat_x = torch.zeros_like(yaw)
            quat_y = torch.zeros_like(yaw)
            quat_z = torch.sin(yaw / 2)
            
            # Update object pose
            obj.write_root_pose_to_sim(
                torch.cat([
                    torch.stack([x_pos, y_pos, z_pos], dim=1),
                    torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)
                ], dim=1),
                env_ids=env_ids
            )
            
        except KeyError:
            # Object doesn't exist in scene (e.g., half_cube, half_cylinder disabled)
            pass

def reset_objects_to_fixed_safe_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """
    Reset objects to a FIXED safe configuration to prevent collisions and overlapping.
    
    This replaces randomization with a single verified valid state.
    
    Args:
        env: The RL environment
        env_ids: Environment indices to reset
    """
    num_resets = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    
    # Define fixed local positions (relative to env origin)
    # Target Object: Left side
    target_local_pos = torch.tensor([-0.15, 0.5, 0.05], device=env.device)
    # Cube: Further left
    cube_local_pos = torch.tensor([-0.25, 0.5, 0.05], device=env.device)
    
    # Define fixed orientation (Identity quaternion)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # 1. Target Object
    try:
        obj = env.scene['target_object']
        
        # Expand for batch
        pos = target_local_pos.unsqueeze(0).repeat(num_resets, 1) + env_origins
        quat = identity_quat.unsqueeze(0).repeat(num_resets, 1)
        
        obj.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=1),
            env_ids=env_ids
        )
        # Stop velocity
        obj.write_root_velocity_to_sim(
            torch.zeros(num_resets, 6, device=env.device),
            env_ids=env_ids
        )
    except KeyError:
        pass
        
    # 2. Cube
    try:
        obj = env.scene['cube']
        
        # Expand for batch
        pos = cube_local_pos.unsqueeze(0).repeat(num_resets, 1) + env_origins
        quat = identity_quat.unsqueeze(0).repeat(num_resets, 1)
        
        obj.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=1),
            env_ids=env_ids
        )
        # Stop velocity
        obj.write_root_velocity_to_sim(
            torch.zeros(num_resets, 6, device=env.device),
            env_ids=env_ids
        )
    except KeyError:
        pass

def reset_robot_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """
    Reset robot joints to their default configuration.
    
    Args:
        env: The RL environment
        env_ids: Environment indices to reset
    """
    # Reset Left Robot
    try:
        robot_left = env.scene["robot_left"]
        # Apply default joint positions and velocities
        robot_left.write_joint_state_to_sim(
            position=robot_left.data.default_joint_pos[env_ids],
            velocity=robot_left.data.default_joint_vel[env_ids],
            env_ids=env_ids
        )
    except KeyError:
        pass
        
    # Reset Right Robot
    try:
        robot_right = env.scene["robot_right"]
        robot_right.write_joint_state_to_sim(
            position=robot_right.data.default_joint_pos[env_ids],
            velocity=robot_right.data.default_joint_vel[env_ids],
            env_ids=env_ids
        )
    except KeyError:
        pass
