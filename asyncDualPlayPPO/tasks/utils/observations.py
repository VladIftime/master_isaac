"""
Observation functions for asymmetric dual-play.

Alice observes: EE pose + gripper state + object states (Euler).
Bob observes: everything Alice observes + goal states (Euler) + per-object distances to goal.

Rotation representation: ZYX Euler angles (roll, pitch, yaw) matching the OpenAI ASP paper.
Quaternions are converted at observation time; the policy never sees raw quaternions.
"""

import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion (w, x, y, z) → ZYX Euler angles (roll, pitch, yaw).

    Matches the paper's rotation representation (Appendix A.2: 'three Euler angles
    on three dimensions, roll, pitch, and yaw').

    Safe for all orientations in table manipulation tasks (pitch < ±80°).
    Gimbal lock at ±90° pitch will not occur for a UR5e doing reach/grasp.

    Args:
        q: (..., 4) tensor [w, x, y, z] — unit quaternion.
    Returns:
        (..., 3) tensor [roll, pitch, yaw] in radians.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Roll (rotation about X-axis)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr, cosr)
    # Pitch (rotation about Y-axis) — clamped to prevent NaN at ±90°
    sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    # Yaw (rotation about Z-axis)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny, cosy)
    return torch.stack([roll, pitch, yaw], dim=-1)


def _euler_xyz_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert ZYX Euler angles (roll, pitch, yaw) → unit quaternion (w, x, y, z).

    Inverse of _quat_to_euler_xyz. Used when a Euler-format state must be
    converted back to quat for Isaac Lab's write_root_pose_to_sim (which
    expects 7D pos+quat, not pos+euler).

    Args:
        euler: (..., 3) tensor [roll, pitch, yaw] in radians.
    Returns:
        (..., 4) tensor [w, x, y, z] — unit quaternion.
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = torch.cos(roll * 0.5),  torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5),   torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


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
    End-effector pose (position + Euler angles) for the single arm.

    Positions are returned in environment-local coordinates (relative to
    env_origins). Rotation returned as ZYX Euler angles (roll, pitch, yaw)
    matching the paper's gripper_position representation.

    Returns:
        (num_envs, 6) — [pos(3), roll, pitch, yaw]
    """
    robot = env.scene[ee_cfg.name]
    body_ids, _ = robot.find_bodies(ee_cfg.body_names)

    pos  = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, body_ids[0]]
    euler = _quat_to_euler_xyz(quat)   # (num_envs, 3)

    return torch.cat([pos, euler], dim=-1)   # (num_envs, 6)


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
    Full per-object state vector (14 dims per object instance).

    Layout per instance:
        pos(3) | euler(3) | lin_vel(3) | ang_vel(3) | dist(1) | contact(1)

    Rotation is ZYX Euler angles (roll, pitch, yaw) matching the paper.
    Velocities are clamped to [-5, 5] m/s. All positions are returned in
    environment-local coordinates (i.e., relative to env_origins).

    Args:
        env: The environment.
        object_cfg: Which object to query.
        gripper_cfg: Body config for the end-effector (for distance).
        contact_cfg: Contact sensor for the gripper.

    Returns:
        (num_envs, num_instances * 14)
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

    # Convert quaternion → Euler angles (paper representation)
    euler = _quat_to_euler_xyz(quat.squeeze(1) if quat.dim() == 3 and quat.shape[1] == 1 else
                               quat.view(-1, 4)).view(*quat.shape[:-1], 3)

    state = torch.cat([pos_local, euler, lin_vel, ang_vel, dist, contact], dim=-1)
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
        # episode_manager stores goals as [pos(3)+euler(3)] × num_objects = 6D per object
        object_slices = {
            "target_object": (0, 6),
            "cube": (6, 12),
        }
        full_goals = env.episode_manager.goal_states
        if object_cfg.name in object_slices:
            start, end = object_slices[object_cfg.name]
            if full_goals.shape[1] >= end:
                return full_goals[:, start:end]
        return full_goals[:, :6]

    if hasattr(env, "extras") and "goal_state" in env.extras:
        return env.extras["goal_state"]

    obj = env.scene[object_cfg.name]
    batch_size = obj.data.root_pos_w.shape[0]
    num_instances = (
        1 if obj.data.root_pos_w.dim() == 2 else obj.data.root_pos_w.shape[1]
    )
    return torch.zeros(batch_size, num_instances * 6, device=env.device)


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
    num_instances = current_flat.shape[1] // 14

    current = current_flat.view(batch_size, num_instances, 14)

    # 3. Positions from object_states() are ALREADY in LOCAL frame (env_origins
    #    subtracted inside object_states).  Do NOT subtract again here.
    current_pos_local = current[..., :3]
    current_euler = current[..., 3:6]   # roll, pitch, yaw

    goal = goal_flat.view(batch_size, num_instances, 6)
    goal_pos = goal[..., :3]
    goal_euler = goal[..., 3:6]         # roll, pitch, yaw

    # 4. Compute distances in the same frame (both Local)
    pos_dist = torch.norm(current_pos_local - goal_pos, dim=-1, keepdim=True)

    # Angular distance: max absolute Euler difference with wraparound correction.
    # Matches the paper (Appendix A.2): "take the maximum among them".
    euler_diff = current_euler - goal_euler
    # Map to [-π, π] to handle wraparound (e.g. 170° vs -170° = 20° apart)
    euler_diff = (euler_diff + math.pi) % (2 * math.pi) - math.pi
    rot_dist = euler_diff.abs().max(dim=-1, keepdim=True)[0]

    return torch.cat([pos_dist, rot_dist], dim=-1).view(batch_size, -1)
