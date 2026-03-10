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
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Joint positions for both arms, concatenated.

    Both configs must specify joint_names to select the correct subset
    of joints from the unified robot asset.

    Returns:
        (num_envs, num_left_joints + num_right_joints)
    """
    robot = env.scene[left_arm_cfg.name]

    if left_arm_cfg.joint_names is None:
        raise ValueError("robot_joint_positions: left_arm_cfg must specify joint_names")
    if right_arm_cfg.joint_names is None:
        raise ValueError("robot_joint_positions: right_arm_cfg must specify joint_names")

    left_ids, _ = robot.find_joints(left_arm_cfg.joint_names)
    right_ids, _ = env.scene[right_arm_cfg.name].find_joints(right_arm_cfg.joint_names)

    return torch.cat([
        robot.data.joint_pos[:, left_ids],
        env.scene[right_arm_cfg.name].data.joint_pos[:, right_ids],
    ], dim=-1)


def ee_poses(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_wrist_3_link"),
    right_ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_wrist_3_link"),
) -> torch.Tensor:
    """
    End-effector poses (position + quaternion) for both arms.

    Positions are returned in environment-local coordinates (relative to
    env_origins) so they are comparable across parallel environments.

    Returns:
        (num_envs, 14) — [left_pos(3), left_quat(4), right_pos(3), right_quat(4)]
    """
    robot = env.scene[left_ee_cfg.name]

    left_ids, _ = robot.find_bodies(left_ee_cfg.body_names)
    right_ids, _ = env.scene[right_ee_cfg.name].find_bodies(right_ee_cfg.body_names)

    left_pos = robot.data.body_pos_w[:, left_ids[0]] - env.scene.env_origins
    left_quat = robot.data.body_quat_w[:, left_ids[0]]

    right_pos = env.scene[right_ee_cfg.name].data.body_pos_w[:, right_ids[0]] - env.scene.env_origins
    right_quat = env.scene[right_ee_cfg.name].data.body_quat_w[:, right_ids[0]]

    return torch.cat([left_pos, left_quat, right_pos, right_quat], dim=-1)


def gripper_positions(
    env: ManagerBasedRLEnv,
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Gripper finger joint positions for both arms, concatenated.

    Both configs must specify joint_names pointing to the gripper finger joints.

    Returns:
        (num_envs, 2) — one value per gripper
    """
    robot = env.scene[left_arm_cfg.name]

    if left_arm_cfg.joint_names is None:
        raise ValueError("gripper_positions: left_arm_cfg must specify joint_names")
    if right_arm_cfg.joint_names is None:
        raise ValueError("gripper_positions: right_arm_cfg must specify joint_names")

    left_ids, _ = robot.find_joints(left_arm_cfg.joint_names)
    right_ids, _ = env.scene[right_arm_cfg.name].find_joints(right_arm_cfg.joint_names)

    return torch.cat([
        robot.data.joint_pos[:, left_ids],
        env.scene[right_arm_cfg.name].data.joint_pos[:, right_ids],
    ], dim=-1)


def object_states(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    left_gripper_cfg: SceneEntityCfg = None,
    right_gripper_cfg: SceneEntityCfg = None,
    left_contact_cfg: SceneEntityCfg = None,
    right_contact_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """
    Full per-object state vector (17 dims per object instance).

    Layout per instance:
        pos(3) | quat(4) | lin_vel(3) | ang_vel(3) | dist_left(1) | dist_right(1) | contact_left(1) | contact_right(1)

    Velocities are clamped to [-5, 5] m/s to prevent outlier values from
    destabilising the network.  All positions are returned in environment-local
    coordinates (i.e., relative to env_origins) so they are comparable across
    parallel environments.

    Args:
        env: The environment.
        object_cfg: Which object to query.
        left_gripper_cfg: Body config for the left end-effector (for distance).
        right_gripper_cfg: Body config for the right end-effector (for distance).
        left_contact_cfg: Contact sensor for the left gripper.
        right_contact_cfg: Contact sensor for the right gripper.

    Returns:
        (num_envs, num_instances * 17)
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

    pos = pos - env.scene.env_origins.unsqueeze(1)
    batch_size, num_instances = pos.shape[:2]

    dist_left  = torch.zeros(batch_size, num_instances, 1, device=env.device)
    dist_right = torch.zeros(batch_size, num_instances, 1, device=env.device)

    def gripper_distance(gripper_cfg) -> torch.Tensor:
        """Return per-instance distance from object to a gripper end-effector."""
        robot = env.scene[gripper_cfg.name]
        ids, _ = robot.find_bodies(gripper_cfg.body_names)
        if not ids:
            return torch.zeros(batch_size, num_instances, 1, device=env.device)
        grip_pos_local = robot.data.body_pos_w[:, ids[0], :] - env.scene.env_origins
        return torch.norm(pos - grip_pos_local.unsqueeze(1), dim=-1, keepdim=True)

    if left_gripper_cfg is not None:
        dist_left = gripper_distance(left_gripper_cfg)
    if right_gripper_cfg is not None:
        dist_right = gripper_distance(right_gripper_cfg)

    contact_left  = torch.zeros(batch_size, num_instances, 1, device=env.device)
    contact_right = torch.zeros(batch_size, num_instances, 1, device=env.device)

    def gripper_contact(contact_cfg, dist) -> torch.Tensor:
        """Return 1.0 where the gripper is both touching (force > 0.1 N) and close (< 0.25 m)."""
        if contact_cfg.name not in env.scene.sensors:
            return torch.zeros(batch_size, num_instances, 1, device=env.device)
        sensor = env.scene.sensors[contact_cfg.name]
        forces = torch.norm(sensor.data.net_forces_w, dim=-1)
        has_contact = (forces.max(dim=1)[0] > 0.1).float().unsqueeze(1).unsqueeze(2)
        return has_contact * (dist < 0.25).float()

    if left_contact_cfg is not None and hasattr(env.scene, "sensors"):
        contact_left = gripper_contact(left_contact_cfg, dist_left)
    if right_contact_cfg is not None and hasattr(env.scene, "sensors"):
        contact_right = gripper_contact(right_contact_cfg, dist_right)

    state = torch.cat([pos, quat, lin_vel, ang_vel, dist_left, dist_right, contact_left, contact_right], dim=-1)
    return state.view(batch_size, -1)


def goal_states(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Goal state for a specific object, as recorded by the episode manager.

    Goal states are stored by the wrapper as [target_object(0:7), cube(7:14)].
    Returns zeros during Alice's phase, when no goal has been set yet.

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
    pos = obj.data.root_pos_w
    num_instances = 1 if pos.dim() == 2 else pos.shape[1]
    return torch.zeros(pos.shape[0], num_instances * 7, device=env.device)


def goal_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """
    Per-object distance from the current state to the goal state.

    Rotation distance uses the geodesic approximation 1 − |q₁ · q₂|,
    which is 0 when the orientations match and 1 when they are maximally
    opposed.

    Returns:
        (num_envs, num_instances * 2) — [pos_dist, rot_dist] per instance
    """
    current_flat = object_states(env, object_cfg)
    goal_flat = goal_states(env, object_cfg)

    batch_size = current_flat.shape[0]
    num_instances = current_flat.shape[1] // 17

    current = current_flat.view(batch_size, num_instances, 17)[..., :7]
    goal = goal_flat.view(batch_size, num_instances, 7)

    pos_dist = torch.norm(current[..., :3] - goal[..., :3], dim=-1, keepdim=True)
    quat_dot = torch.sum(current[..., 3:7] * goal[..., 3:7], dim=-1, keepdim=True)
    rot_dist = 1.0 - torch.abs(quat_dot)

    return torch.cat([pos_dist, rot_dist], dim=-1).view(batch_size, -1)
