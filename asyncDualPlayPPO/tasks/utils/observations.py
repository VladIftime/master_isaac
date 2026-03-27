"""
Observation functions for asymmetric dual-play.

Alice observes: robot joint positions + gripper positions + object states.
Bob observes: everything Alice observes + goal states + per-object distances to goal.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def robot_joint_positions(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Joint positions for the robot arm.

    The config must specify joint_names to select the correct subset
    of joints from the robot asset.

    Returns:
        (num_envs, num_joints)
    """
    robot = env.scene[arm_cfg.name]

    if arm_cfg.joint_names is None:
        raise ValueError("robot_joint_positions: arm_cfg must specify joint_names")

    joint_ids, _ = robot.find_joints(arm_cfg.joint_names)

    return robot.data.joint_pos[:, joint_ids]


def ee_poses(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist_3_link"),
) -> torch.Tensor:
    """
    End-effector pose (position + quaternion) for the single arm.

    Positions are returned in environment-local coordinates (relative to
    env_origins) so they are comparable across parallel environments.

    Returns:
        (num_envs, 7) — [pos(3), quat(4)]
    """
    robot = env.scene[ee_cfg.name]
    body_ids, _ = robot.find_bodies(ee_cfg.body_names)

    pos = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, body_ids[0]]

    return torch.cat([pos, quat], dim=-1)


def other_arm_ee_pos(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist_3_link"),
) -> torch.Tensor:
    """
    Other arm's end-effector position in environment-local coordinates (3 dims).

    Phase 1.5: gives each worker minimal spatial awareness of the opposing arm.
    This is NOT for collision avoidance (RMPflow handles that) but for informed
    timing decisions when the shared workspace is contested.

    Returns:
        (num_envs, 3) — [x, y, z] in local frame
    """
    robot = env.scene[ee_cfg.name]
    body_ids, _ = robot.find_bodies(ee_cfg.body_names)
    pos = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    return pos


def gripper_positions(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Gripper finger joint position for the arm.

    The config must specify joint_names pointing to the gripper finger joint.

    Returns:
        (num_envs, 1) — finger joint position
    """
    robot = env.scene[arm_cfg.name]

    if arm_cfg.joint_names is None:
        raise ValueError("gripper_positions: arm_cfg must specify joint_names")

    joint_ids, _ = robot.find_joints(arm_cfg.joint_names)

    return robot.data.joint_pos[:, joint_ids]


def object_states(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    gripper_cfg: SceneEntityCfg = None,
    contact_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """
    Full per-object state vector (15 dims per object instance).

    Layout per instance:
        pos(3) | quat(4) | lin_vel(3) | ang_vel(3) | dist(1) | contact(1)

    Velocities are clamped to [-5, 5] m/s. All positions are returned in
    environment-local coordinates (i.e., relative to env_origins).

    Args:
        env: The environment.
        object_cfg: Which object to query.
        gripper_cfg: Body config for the end-effector (for distance).
        contact_cfg: Contact sensor for the gripper.

    Returns:
        (num_envs, num_instances * 15)
    """
    obj = env.scene[object_cfg.name]

    pos = obj.data.root_pos_w
    quat = obj.data.root_quat_w
    lin_vel = torch.clamp(obj.data.root_lin_vel_w, -5.0, 5.0)
    ang_vel = torch.clamp(obj.data.root_ang_vel_w, -5.0, 5.0)

    if pos.dim() == 2:
        pos = pos.unsqueeze(1)
        quat = quat.unsqueeze(1)
        lin_vel = lin_vel.unsqueeze(1)
        ang_vel = ang_vel.unsqueeze(1)

    pos_local = pos - env.scene.env_origins.unsqueeze(1)
    batch_size, num_instances = pos.shape[:2]

    # Initialize distance and contact tensors
    dist = torch.zeros(batch_size, num_instances, 1, device=env.device)
    contact = torch.zeros(batch_size, num_instances, 1, device=env.device)

    # Calculate distance to gripper if provided
    if gripper_cfg is not None:
        robot = env.scene[gripper_cfg.name]
        body_ids, _ = robot.find_bodies(gripper_cfg.body_names)
        if body_ids:
            grip_pos_local = (
                robot.data.body_pos_w[:, body_ids[0], :] - env.scene.env_origins
            )
            dist = torch.norm(
                pos_local - grip_pos_local.unsqueeze(1), dim=-1, keepdim=True
            )

    # Calculate contact if sensor provided
    if contact_cfg is not None and hasattr(env.scene, "sensors"):
        if contact_cfg.name in env.scene.sensors:
            sensor = env.scene.sensors[contact_cfg.name]
            forces = torch.norm(sensor.data.net_forces_w, dim=-1)
            # Contact if net force > 0.1N and object is close (< 0.25m)
            has_contact = (forces.max(dim=1)[0] > 0.1).float().unsqueeze(1).unsqueeze(2)
            contact = has_contact * (dist < 0.25).float()

    state = torch.cat([pos_local, quat, lin_vel, ang_vel, dist, contact], dim=-1)
    return state.view(batch_size, -1)


def goal_states(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Goal state for a specific object, as recorded by the episode manager.

    Goal states are stored as [target_object(0:7), cube(7:14)].
    Returns zeros during Alice's phase.

    Returns:
        (num_envs, 7) — [pos(3), quat(4)] for the requested object
    """
    if hasattr(env, "episode_manager") and env.episode_manager.goal_states is not None:
        object_slices = {
            "target_object": (0, 7),
            "cube": (7, 14),
        }
        full_goals = env.episode_manager.goal_states
        if object_cfg.name in object_slices:
            start, end = object_slices[object_cfg.name]
            if full_goals.shape[1] >= end:
                return full_goals[:, start:end]
        return full_goals[:, :7]

    if hasattr(env, "extras") and "goal_state" in env.extras:
        return env.extras["goal_state"]

    obj = env.scene[object_cfg.name]
    batch_size = obj.data.root_pos_w.shape[0]
    num_instances = (
        1 if obj.data.root_pos_w.dim() == 2 else obj.data.root_pos_w.shape[1]
    )
    return torch.zeros(batch_size, num_instances * 7, device=env.device)


def goal_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Per-object distance from the current state to the goal state in the LOCAL frame.
    """
    # 1. Get world-space states
    current_flat = object_states(env, object_cfg)
    # 2. Get stored goal states (assumed to be stored in LOCAL frame via updated wrapper)
    goal_flat = goal_states(env, object_cfg)

    batch_size = current_flat.shape[0]
    num_instances = current_flat.shape[1] // 15

    current = current_flat.view(batch_size, num_instances, 15)

    # 3. Positions from object_states() are ALREADY in LOCAL frame (env_origins
    #    subtracted inside object_states).  Do NOT subtract again here.
    current_pos_local = current[..., :3]
    current_quat = current[..., 3:7]

    goal = goal_flat.view(batch_size, num_instances, 7)
    goal_pos = goal[..., :3]
    goal_quat = goal[..., 3:7]

    # 4. Compute distances in the same frame (both Local)
    pos_dist = torch.norm(current_pos_local - goal_pos, dim=-1, keepdim=True)
    quat_dot = torch.sum(current_quat * goal_quat, dim=-1, keepdim=True)
    rot_dist = 1.0 - torch.abs(quat_dot)

    return torch.cat([pos_dist, rot_dist], dim=-1).view(batch_size, -1)
