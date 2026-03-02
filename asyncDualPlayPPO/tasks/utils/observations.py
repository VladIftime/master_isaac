"""
Observation functions for asymmetric dual-play.

Alice observes: robot state + object state
Bob observes: robot state + object state + goal state
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp


def robot_joint_positions(
    env: ManagerBasedRLEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Get joint positions for both arms.
    
    Returns:
        Joint positions concatenated (num_envs, num_joints_left + num_joints_right)
    """
    # Both are actually pointing to the same robot object now
    left_robot = env.scene[left_arm_cfg.name]
    right_robot = env.scene[right_arm_cfg.name]

    # Create indices using find_joints
    if left_arm_cfg.joint_names is not None:
        left_indices, _ = left_robot.find_joints(left_arm_cfg.joint_names)
    else:
        raise ValueError("robot_joint_positions: joint_names must be provided in left_arm_cfg")
        
    if right_arm_cfg.joint_names is not None:
        right_indices, _ = right_robot.find_joints(right_arm_cfg.joint_names)
    else:
        raise ValueError("robot_joint_positions: joint_names must be provided in right_arm_cfg")
    
    # Get joint positions
    left_joints = left_robot.data.joint_pos[:, left_indices]
    right_joints = right_robot.data.joint_pos[:, right_indices]
    
    return torch.cat([left_joints, right_joints], dim=-1)


def gripper_positions(
    env: ManagerBasedRLEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Get gripper positions for both arms.
    
    Returns:
        Gripper positions (num_envs, 2) - one value per gripper
    """
    # Both are actually pointing to the same RobotUnified object now
    left_robot = env.scene[left_arm_cfg.name]
    right_robot = env.scene[right_arm_cfg.name]
    
    # Resolve indices (joint_names MUST be provided in config)
    if left_arm_cfg.joint_names is not None:
        left_indices, _ = left_robot.find_joints(left_arm_cfg.joint_names)
    else:
         raise ValueError("gripper_positions: joint_names must be provided in left_arm_cfg")

    if right_arm_cfg.joint_names is not None:
        right_indices, _ = right_robot.find_joints(right_arm_cfg.joint_names)
    else:
        raise ValueError("gripper_positions: joint_names must be provided in right_arm_cfg")

    # Get gripper joint position
    left_gripper = left_robot.data.joint_pos[:, left_indices]
    right_gripper = right_robot.data.joint_pos[:, right_indices]
    
    return torch.cat([left_gripper, right_gripper], dim=-1)


def object_states(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    left_gripper_cfg: SceneEntityCfg = None,
    right_gripper_cfg: SceneEntityCfg = None,
    left_contact_cfg: SceneEntityCfg = None,
    right_contact_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """
    Get expanded object states (17 dims).
    
    Structure per object:
    - Position (3)
    - Rotation Quaternion (4)
    - Linear Velocity (3)
    - Angular Velocity (3)
    - Dist to Left Gripper (1)
    - Dist to Right Gripper (1)
    - Contact Left (1)
    - Contact Right (1)
    
    Total: 17 dimensions per object.
    
    Args:
        env: The environment.
        object_cfg: Configuration for the object(s).
        left_gripper_cfg: Configuration for left gripper body (for distance).
        right_gripper_cfg: Configuration for right gripper body (for distance).
        left_contact_cfg: Configuration for left contact sensor.
        right_contact_cfg: Configuration for right contact sensor.
                   
    Returns:
        Flattened object states tensor: (num_envs, num_instances * 17)
    """
    obj = env.scene[object_cfg.name]
    
    # 1. Pose
    # Root position: (num_envs, num_instances, 3)
    pos = obj.data.root_pos_w
    # Root quaternion: (num_envs, num_instances, 4)
    quat = obj.data.root_quat_w
    
    # 2. Velocity
    # Linear velocity: (num_envs, num_instances, 3)
    lin_vel = torch.clamp(obj.data.root_lin_vel_w, -5.0, 5.0)
    # Angular velocity: (num_envs, num_instances, 3)
    ang_vel = torch.clamp(obj.data.root_ang_vel_w, -5.0, 5.0)
    
    # Handle single instance case
    if pos.dim() == 2:
        pos = pos.unsqueeze(1)
        quat = quat.unsqueeze(1)
        lin_vel = lin_vel.unsqueeze(1)
        ang_vel = ang_vel.unsqueeze(1)

    # Convert to local coordinates
    pos = pos - env.scene.env_origins.unsqueeze(1)

    batch_size, num_instances = pos.shape[:2]

    # 3. Gripper Distances
    # Initialize with zeros if configs not provided
    dist_left = torch.zeros(batch_size, num_instances, 1, device=env.device)
    dist_right = torch.zeros(batch_size, num_instances, 1, device=env.device)
    
    if left_gripper_cfg is not None:
        # Get gripper body position (world frame)
        # body_pos_w: (num_envs, num_bodies, 3) -> slice using cfg body indices
        # We assume cfg points to a specific link (e.g. wrist_3)
        left_robot = env.scene[left_gripper_cfg.name]
        # Get indices for the requested bodies
        # We specified body_names=".*wrist_3_link" in the config.
        # Resolve body indices for the configured body name.
        # The articulation data `body_pos_w` is (num_envs, num_bodies, 3).
        
        # Helper to get body pos
        def get_body_pos(robot, body_pattern):
            # Find body index
            # `robot.find_bodies(body_pattern)` returns indices.
            if body_pattern is None:
                return torch.zeros(batch_size, 3, device=env.device)
                
            indices, _ = robot.find_bodies(body_pattern)
            if len(indices) > 0:
                # Take first match
                idx = indices[0]
                return robot.data.body_pos_w[:, idx, :] # (num_envs, 3)
            return torch.zeros(batch_size, 3, device=env.device)

        left_grip_pos = get_body_pos(left_robot, left_gripper_cfg.body_names) # (num_envs, 3)
        
        # Compute Distance
        # Obj pos: (N, I, 3)
        # Grip pos: (N, 3) -> (N, 1, 3)
        d = torch.norm(pos - (left_grip_pos.unsqueeze(1) - env.scene.env_origins.unsqueeze(1)), dim=-1, keepdim=True)
        dist_left = d

    if right_gripper_cfg is not None:
        right_robot = env.scene[right_gripper_cfg.name]
        right_grip_pos = get_body_pos(right_robot, right_gripper_cfg.body_names)
        d = torch.norm(pos - (right_grip_pos.unsqueeze(1) - env.scene.env_origins.unsqueeze(1)), dim=-1, keepdim=True)
        dist_right = d

    # 4. Contact
    # Initialize with zeros
    contact_left = torch.zeros(batch_size, num_instances, 1, device=env.device)
    contact_right = torch.zeros(batch_size, num_instances, 1, device=env.device)
    
    if left_contact_cfg is not None and hasattr(env.scene, "sensors"):
        # Check sensor existence in env.scene.sensors

        
        if left_contact_cfg.name in env.scene.sensors:
            sensor = env.scene.sensors[left_contact_cfg.name]
            
            # Gripper Force Magnitude
            # Compute max force magnitude across links.
            forces = torch.norm(sensor.data.net_forces_w, dim=-1) # (num_envs, num_links)
            has_contact = (torch.max(forces, dim=1)[0] > 0.1).float().unsqueeze(1).unsqueeze(2) # (N, 1, 1)
            
            # Combine with distance check (from step 3)
            # Threshold: 0.25m distance check.
            is_close = (dist_left < 0.25).float()
            
            contact_left = has_contact * is_close

    if right_contact_cfg is not None and right_contact_cfg.name in env.scene.sensors:
        sensor = env.scene.sensors[right_contact_cfg.name]
        forces = torch.norm(sensor.data.net_forces_w, dim=-1)
        has_contact = (torch.max(forces, dim=1)[0] > 0.1).float().unsqueeze(1).unsqueeze(2)
        is_close = (dist_right < 0.25).float()
        contact_right = has_contact * is_close

    # Concatenate state: (num_envs, num_instances, 17)
    # [pos(3), quat(4), lin_vel(3), ang_vel(3), dist_L(1), dist_R(1), cont_L(1), cont_R(1)]
    obj_state = torch.cat([pos, quat, lin_vel, ang_vel, dist_left, dist_right, contact_left, contact_right], dim=-1)
    
    # Flatten: (num_envs, num_instances * 17)
    return obj_state.view(batch_size, -1)


def goal_states(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Get goal states for Bob. Returns goal state for the specified object.
    
    The goal states are stored in episode_manager.goal_states as concatenated states
    for all tracked objects: [target_object(7), cube(7)] = (num_envs, 14)
    
    Returns:
        Goal states for the specified object (num_envs, 7).
        If not present (e.g. Alice phase), returns zeros.
    """
    # Try to retrieve from episode manager (preferred source of truth)
    if hasattr(env, 'episode_manager') and env.episode_manager.goal_states is not None:
        full_goals = env.episode_manager.goal_states
        
        # Map object name to slice index
        # Goal states are stored as: [target_object(0:7), cube(7:14)]
        object_slices = {
            "target_object": (0, 7),
            "cube": (7, 14),
        }
        
        if object_cfg.name in object_slices:
            start, end = object_slices[object_cfg.name]
            if full_goals.shape[1] >= end:
                return full_goals[:, start:end]
        
        # Fallback: return first 7 values if object not found in mapping
        return full_goals[:, :7]
    
    # Try to retrieve from extras
    if hasattr(env, 'extras') and "goal_state" in env.extras:
        return env.extras["goal_state"]
    
    # Fallback/Default: Zero tensor matching object dimensions
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w
    
    if pos.dim() == 2: # (num_envs, 3)
        num_instances = 1
    else: # (num_envs, num_instances, 3)
        num_instances = pos.shape[1]
        
    batch_size = pos.shape[0]
    state_dim = 7 # pos + quat
    
    return torch.zeros(batch_size, num_instances * state_dim, device=env.device)


def goal_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Compute distance from current object state to goal state.
    
    Returns:
        Flattened distance metrics (num_envs, num_objects * 2):
        - position distance (L2 norm)
        - rotation distance (1 - |dot(q1, q2)|)
    """
    # These are already flattened: (num_envs, num_instances * 17)
    current_flat = object_states(env, object_cfg)
    # Goal states are flattened: (num_envs, num_instances * 7)
    goal_flat = goal_states(env, object_cfg)
    
    batch_size = current_flat.shape[0]
    # Infer num_instances
    # current has 17 dims per object (pos, rot, vel, etc)
    num_instances = current_flat.shape[1] // 17
    
    # Reshape
    current = current_flat.view(batch_size, num_instances, 17)
    # Extract pos+rot from current (first 7 dims)
    current = current[..., :7]
    
    goal = goal_flat.view(batch_size, num_instances, 7)
    
    # Split pos and quat
    pos_current = current[..., :3]
    pos_goal = goal[..., :3]
    
    quat_current = current[..., 3:7]
    quat_goal = goal[..., 3:7]
    
    # Position distance: (num_envs, num_instances, 1)
    pos_dist = torch.norm(pos_current - pos_goal, dim=-1, keepdim=True)
    
    # Rotation distance
    # q1 . q2
    quat_dot = torch.sum(quat_current * quat_goal, dim=-1, keepdim=True)
    # 1 - |dot|
    quat_dist = 1.0 - torch.abs(quat_dot)
    
    # Combine: (num_envs, num_instances, 2)
    dists = torch.cat([pos_dist, quat_dist], dim=-1)
    
    # Flatten: (num_envs, num_instances * 2)
    return dists.view(batch_size, -1)
