# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operational Space Control (OSC) for Dual-Arm Setup using Custom URDF

This script demonstrates operational space control for a dual-arm robot setup
using a single URDF file that contains both arms integrated into one robot model.

The script allows easy editing of desired end-effector poses through goal arrays.

Usage:
    ./isaaclab.sh -p dual_arm_Isaacgym/asyncDualPlayPPO/run_dual_arm_osc.py --num_envs 4
    ./isaaclab.sh -p dual_arm_Isaacgym/asyncDualPlayPPO/run_dual_arm_osc.py --num_envs 4 --goal 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Dual-arm operational space controller demonstration."
)
parser.add_argument(
    "--num_envs", type=int, default=4, help="Number of environments to spawn."
)
parser.add_argument(
    "--goal", type=int, default=0, help="Goal index to start with (0, 1, 2, etc.)"
)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import (
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

# Get URDF path
URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf/dual_arm_robot.urdf")


@configclass
class DualArmSceneCfg(InteractiveSceneCfg):
    """Configuration for dual-arm scene using single URDF."""

    # Dual-arm robot (both arms in one URDF)
    robot = ArticulationCfg(
        spawn=UrdfFileCfg(
            asset_path=URDF_PATH,
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Left arm crane pose (extending upward and to the left)
                "left_shoulder_pan_joint": 0.785,  # 45° outward to the left
                "left_shoulder_lift_joint": -2.0,  # Raised up
                "left_elbow_joint": -1.0,  # Bent elbow
                "left_wrist_1_joint": -1.0,  # Wrist bend
                "left_wrist_2_joint": 1.57,  # Wrist rotation
                "left_wrist_3_joint": 0.0,  # Neutral
                # Right arm crane pose (extending upward and to the right - MIRRORED)
                "right_shoulder_pan_joint": -0.785,  # 45° outward to the right (mirrored)
                "right_shoulder_lift_joint": -2.0,  # Raised up (same as left)
                "right_elbow_joint": 1.0,  # Bent elbow (mirrored sign)
                "right_wrist_1_joint": -1.0,  # Wrist bend (same as left)
                "right_wrist_2_joint": -1.57,  # Wrist rotation (mirrored)
                "right_wrist_3_joint": 0.0,  # Neutral
                # Grippers (open position)
                "lgripper_finger_joint": 0.0,
                "rgripper_finger_joint": 0.0,
            },
        ),
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
                stiffness=0.0,
                damping=0.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_.*",
                    "right_elbow_.*",
                    "right_wrist_.*",
                ],
                stiffness=0.0,
                damping=0.0,
            ),
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["lgripper_.*"],
                stiffness=1e5,
                damping=80.0,
            ),
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=["rgripper_.*"],
                stiffness=1e5,
                damping=80.0,
            ),
        },
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop with dual-arm OSC control.

    ===============================================================================
    EDITABLE GOAL POSES SECTION - Modify these to change target positions
    ===============================================================================

    Goals are in body frame relative to robot base: [x, y, z, qx, qy, qz, qw]
    - Position (x, y, z) in meters
    - Orientation as quaternion (qx, qy, qz, qw)

    Example orientations:
    - [0.707, 0, 0.707, 0] = pointing down
    - [0.0, 1.0, 0.0, 0.0] = pointing forward
    - [0.707, 0.707, 0.0, 0.0] = angled
    """

    # LEFT ARM GOAL POSES (edit these!)
    left_goals = [
        # Goal 0: Reach forward and left
        [-0.3, 0.5, 0.3, 0.707, 0, 0.707, 0],
        # Goal 1: Higher position
        [-0.2, 0.6, 0.4, 0.707, 0.707, 0.0, 0.0],
        # Goal 2: Lower position
        [-0.3, 0.4, 0.25, 0.0, 1.0, 0.0, 0.0],
    ]

    # RIGHT ARM GOAL POSES (edit these!)
    right_goals = [
        # Goal 0: Reach forward and right (mirrored from left)
        [0.3, 0.5, 0.3, 0.707, 0, 0.707, 0],
        # Goal 1: Higher position
        [0.2, 0.6, 0.4, 0.707, 0.707, 0.0, 0.0],
        # Goal 2: Lower position
        [0.3, 0.4, 0.25, 0.0, 1.0, 0.0, 0.0],
    ]

    """
    ===============================================================================
    END OF EDITABLE SECTION
    ===============================================================================
    """

    # Extract robot (single articulation containing both arms)
    robot = scene["robot"]

    # Configure robot entity settings for both arms
    left_arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
        body_names=["left_wrist_3_link"],
    )
    right_arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
        body_names=["right_wrist_3_link"],
    )

    # Resolve scene entities
    left_arm_cfg.resolve(scene)
    right_arm_cfg.resolve(scene)

    # Get end-effector Jacobian indices
    if robot.is_fixed_base:
        ee_jacobi_idx_left = left_arm_cfg.body_ids[0] - 1
        ee_jacobi_idx_right = right_arm_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx_left = left_arm_cfg.body_ids[0]
        ee_jacobi_idx_right = right_arm_cfg.body_ids[0]

    # Create OSC controllers for both arms
    # Note: UR5e has 6 DOF, controlling 6 DOF task space -> no redundancy, no nullspace
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=1.0,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],  # Control all 6 DOF
        nullspace_control="none",  # No nullspace for 6-DOF arm with 6-DOF control
    )
    osc_left = OperationalSpaceController(
        osc_cfg, num_envs=scene.num_envs, device=sim.device
    )
    osc_right = OperationalSpaceController(
        osc_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # Create visualization markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Markers for left arm
    ee_marker_left = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_left_current")
    )
    goal_marker_left = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_left_goal")
    )

    # Markers for right arm
    ee_marker_right = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_right_current")
    )
    goal_marker_right = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_right_goal")
    )

    # Convert goals to tensors
    left_goals = torch.tensor(left_goals, device=sim.device)
    right_goals = torch.tensor(right_goals, device=sim.device)

    # Stiffness values for variable impedance mode (optional: can be edited per goal)
    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
        ],
        device=sim.device,
    )

    # Combine pose and stiffness into command tensors
    ee_target_set_left = torch.cat([left_goals, kp_set_task[: len(left_goals)]], dim=-1)
    ee_target_set_right = torch.cat(
        [right_goals, kp_set_task[: len(right_goals)]], dim=-1
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Update robot buffers
    robot.update(dt=sim_dt)

    # Track current goal index
    current_goal_idx = min(args_cli.goal, len(left_goals) - 1)

    # Command buffers
    command_left = torch.zeros(scene.num_envs, osc_left.action_dim, device=sim.device)
    command_right = torch.zeros(scene.num_envs, osc_right.action_dim, device=sim.device)
    ee_target_pose_b_left = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b_right = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_w_left = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_w_right = torch.zeros(scene.num_envs, 7, device=sim.device)

    # Set joint efforts to zero initially
    zero_joint_efforts = torch.zeros(
        scene.num_envs, robot.num_joints, device=sim.device
    )

    # Print info
    print("\n" + "=" * 60)
    print("Dual-Arm Operational Space Control")
    print(f"  Starting at goal index: {current_goal_idx}")
    print(f"  Total goals available: {len(left_goals)}")
    print("  Robot will cycle through goals every 500 steps")
    print("  Press Ctrl+C to quit")
    print("=" * 60 + "\n")

    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            # Reset joint state to default
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)
            robot.write_data_to_sim()
            robot.reset()

            # Update robot buffers
            robot.update(sim_dt)

            # Get current states
            (
                jacobian_b_left,
                mass_matrix_left,
                gravity_left,
                ee_pose_b_left,
                ee_vel_b_left,
                root_pose_w,
                ee_pose_w_left,
            ) = update_states_left(sim, scene, robot, ee_jacobi_idx_left, left_arm_cfg)

            (
                jacobian_b_right,
                mass_matrix_right,
                gravity_right,
                ee_pose_b_right,
                ee_vel_b_right,
                _,
                ee_pose_w_right,
            ) = update_states_right(
                sim, scene, robot, ee_jacobi_idx_right, right_arm_cfg
            )

            # Update target pose
            (
                command_left,
                ee_target_pose_b_left,
                ee_target_pose_w_left,
                current_goal_idx,
            ) = update_target(
                sim, scene, osc_left, root_pose_w, ee_target_set_left, current_goal_idx
            )
            command_right, ee_target_pose_b_right, ee_target_pose_w_right, _ = (
                update_target(
                    sim,
                    scene,
                    osc_right,
                    root_pose_w,
                    ee_target_set_right,
                    current_goal_idx,
                )
            )

            # Set OSC commands
            osc_left.reset()
            osc_right.reset()
            command_left, task_frame_pose_b_left = convert_to_task_frame(
                osc_left, command_left, ee_target_pose_b_left
            )
            command_right, task_frame_pose_b_right = convert_to_task_frame(
                osc_right, command_right, ee_target_pose_b_right
            )
            osc_left.set_command(
                command=command_left,
                current_ee_pose_b=ee_pose_b_left,
                current_task_frame_pose_b=task_frame_pose_b_left,
            )
            osc_right.set_command(
                command=command_right,
                current_ee_pose_b=ee_pose_b_right,
                current_task_frame_pose_b=task_frame_pose_b_right,
            )

            print(
                f"[GOAL {current_goal_idx}] Left: {left_goals[current_goal_idx][:3].tolist()}, Right: {right_goals[current_goal_idx][:3].tolist()}"
            )
        else:
            # Get updated states
            (
                jacobian_b_left,
                mass_matrix_left,
                gravity_left,
                ee_pose_b_left,
                ee_vel_b_left,
                root_pose_w,
                ee_pose_w_left,
            ) = update_states_left(sim, scene, robot, ee_jacobi_idx_left, left_arm_cfg)

            (
                jacobian_b_right,
                mass_matrix_right,
                gravity_right,
                ee_pose_b_right,
                ee_vel_b_right,
                _,
                ee_pose_w_right,
            ) = update_states_right(
                sim, scene, robot, ee_jacobi_idx_right, right_arm_cfg
            )

            # Compute joint commands for left arm
            joint_efforts_left = osc_left.compute(
                jacobian_b=jacobian_b_left,
                current_ee_pose_b=ee_pose_b_left,
                current_ee_vel_b=ee_vel_b_left,
                mass_matrix=mass_matrix_left,
                gravity=gravity_left,
                current_joint_pos=robot.data.joint_pos[:, left_arm_cfg.joint_ids],
                current_joint_vel=robot.data.joint_vel[:, left_arm_cfg.joint_ids],
            )

            # Compute joint commands for right arm
            joint_efforts_right = osc_right.compute(
                jacobian_b=jacobian_b_right,
                current_ee_pose_b=ee_pose_b_right,
                current_ee_vel_b=ee_vel_b_right,
                mass_matrix=mass_matrix_right,
                gravity=gravity_right,
                current_joint_pos=robot.data.joint_pos[:, right_arm_cfg.joint_ids],
                current_joint_vel=robot.data.joint_vel[:, right_arm_cfg.joint_ids],
            )

            # Apply actions
            robot.set_joint_effort_target(
                joint_efforts_left, joint_ids=left_arm_cfg.joint_ids
            )
            robot.set_joint_effort_target(
                joint_efforts_right, joint_ids=right_arm_cfg.joint_ids
            )
            robot.write_data_to_sim()

        # Update marker positions
        ee_marker_left.visualize(ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7])
        goal_marker_left.visualize(
            ee_target_pose_w_left[:, 0:3], ee_target_pose_w_left[:, 3:7]
        )

        ee_marker_right.visualize(ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7])
        goal_marker_right.visualize(
            ee_target_pose_w_right[:, 0:3], ee_target_pose_w_right[:, 3:7]
        )

        # Perform step
        sim.step(render=True)
        # Update robot buffers
        robot.update(sim_dt)
        # Update scene buffers
        scene.update(sim_dt)
        # Update sim-time
        count += 1


def update_states_left(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot,
    ee_jacobi_idx: int,
    arm_cfg: SceneEntityCfg,
):
    """Update the left arm states."""
    # Obtain dynamics related quantities from simulation
    jacobian_w = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, arm_cfg.joint_ids
    ]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[
        :, arm_cfg.joint_ids, :
    ][:, :, arm_cfg.joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[
        :, arm_cfg.joint_ids
    ]

    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, arm_cfg.body_ids[0]]
    ee_quat_w = robot.data.body_quat_w[:, arm_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, arm_cfg.body_ids[0], :]
    root_vel_w = robot.data.root_vel_w
    relative_vel_w = ee_vel_w - root_vel_w
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w


def update_states_right(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot,
    ee_jacobi_idx: int,
    arm_cfg: SceneEntityCfg,
):
    """Update the right arm states."""
    # Obtain dynamics related quantities from simulation
    jacobian_w = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, arm_cfg.joint_ids
    ]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[
        :, arm_cfg.joint_ids, :
    ][:, :, arm_cfg.joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[
        :, arm_cfg.joint_ids
    ]

    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, arm_cfg.body_ids[0]]
    ee_quat_w = robot.data.body_quat_w[:, arm_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, arm_cfg.body_ids[0], :]
    root_vel_w = robot.data.root_vel_w
    relative_vel_w = ee_vel_w - root_vel_w
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w


def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.Tensor,
    ee_target_set: torch.Tensor,
    current_goal_idx: int,
):
    """Update the targets for the operational space controller."""
    # Update the ee desired command
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    # Update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:] = command[:, :7]

    # Update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_target_pose_b[:, 0:3],
        ee_target_pose_b[:, 3:7],
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


def convert_to_task_frame(
    osc: OperationalSpaceController,
    command: torch.Tensor,
    ee_target_pose_b: torch.Tensor,
):
    """Converts the target commands to the task frame."""
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3],
                task_frame_pose_b[:, 3:],
                command[:, :3],
                command[:, 3:7],
            )
            cmd_idx += 7
        else:
            raise ValueError(f"Undefined target_type: {target_type}")

    return command, task_frame_pose_b


def main():
    """Main function."""

    # Verify URDF exists
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found at: {URDF_PATH}")

    print(f"[INFO] Using URDF: {URDF_PATH}")

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    # Design scene - use custom dual-arm URDF configuration
    scene_cfg = DualArmSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO] Setup complete!")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print("[INFO] Press Ctrl+C to exit")

    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
