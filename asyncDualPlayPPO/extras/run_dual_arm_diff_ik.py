# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Differential IK Controller for Dual-Arm Setup using Custom URDF

This script demonstrates differential inverse kinematics control for a dual-arm robot setup
using a single URDF file that contains both arms integrated into one robot model.

Usage:
    ./isaaclab.sh -p dual_arm_Isaacgym/asyncDualPlayPPO/run_dual_arm_diff_ik.py --num_envs 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Dual-arm differential IK controller demonstration.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--sync_mode", action="store_true", help="If set, both arms move to the same goal. Otherwise, independent goals.")
parser.add_argument("--goal", type=int, default=0, help="Goal index to move to (0, 1, 2, etc.)")
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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# Get URDF path
URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "urdf/dual_arm_robot.urdf"
)


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
                    stiffness=1000.0,
                    damping=50.0,
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
                "left_shoulder_pan_joint": 0.785,     # 45° outward to the left
                "left_shoulder_lift_joint": -2.0,     # Raised up
                "left_elbow_joint": -1.0,             # Bent elbow
                "left_wrist_1_joint": -1.0,           # Wrist bend
                "left_wrist_2_joint": 1.57,           # Wrist rotation
                "left_wrist_3_joint": 0.0,            # Neutral
                # Right arm crane pose (extending upward and to the right - MIRRORED)
                "right_shoulder_pan_joint": -0.785,   # 45° outward to the right (mirrored)
                "right_shoulder_lift_joint": -2.0,    # Raised up (same as left)
                "right_elbow_joint": 1.0,             # Bent elbow (mirrored sign)
                "right_wrist_1_joint": -1.0,          # Wrist bend (same as left)
                "right_wrist_2_joint": -1.57,         # Wrist rotation (mirrored)
                "right_wrist_3_joint": 0.0,           # Neutral
                # Grippers (open position)
                "lgripper_finger_joint": 0.0,
                "rgripper_finger_joint": 0.0,
            },
        ),
        actuators={
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
                stiffness=5000.0,
                damping=200.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
                stiffness=5000.0,
                damping=200.0,
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
    """Runs the simulation loop with dual-arm differential IK control."""
    
    # Extract robot (single articulation containing both arms)
    robot = scene["robot"]
    
    # Create differential IK controllers for both arms
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls"
    )
    diff_ik_left = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    diff_ik_right = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    # Create visualization markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    # Markers for left arm (green)
    ee_marker_left = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_left_current"))
    goal_marker_left = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_left_goal"))
    
    # Markers for right arm (blue)
    ee_marker_right = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_right_current"))
    goal_marker_right = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_right_goal"))
    
    # Define goal sequences for both arms
    # Goals are in world frame: [x, y, z, qx, qy, qz, qw]
    
    if args_cli.sync_mode:
        # Both arms move to mirrored positions
        left_goals = [
            [-0.3, 0.5, 0.3, 0.707, 0, 0.707, 0],
            [-0.2, 0.6, 0.4, 0.707, 0.707, 0.0, 0.0],
            [-0.3, 0.4, 0.25, 0.0, 1.0, 0.0, 0.0],
        ]
        right_goals = [
            [0.3, 0.5, 0.3, 0.707, 0, 0.707, 0],
            [0.2, 0.6, 0.4, 0.707, 0.707, 0.0, 0.0],
            [0.3, 0.4, 0.25, 0.0, 1.0, 0.0, 0.0],
        ]
    else:
        # Independent movements for coordinated tasks
        left_goals = [
            [-0.3, 0.5, 0.3, 0.707, 0, 0.707, 0],
            [-0.2, 0.7, 0.35, 0.707, 0.707, 0.0, 0.0],
            [-0.4, 0.5, 0.25, 0.0, 1.0, 0.0, 0.0],
            [-0.2, 0.4, 0.4, 0.707, 0, 0.707, 0],
        ]
        right_goals = [
            [0.3, 0.5, 0.3, 0.707, 0, 0.707, 0],
            [0.4, 0.5, 0.25, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.7, 0.35, 0.707, 0.707, 0.0, 0.0],
            [0.2, 0.4, 0.4, 0.707, 0, 0.707, 0],
        ]
    
    left_goals = torch.tensor(left_goals, device=sim.device)
    right_goals = torch.tensor(right_goals, device=sim.device)
    
    # Create buffers for IK commands (will be set later from command-line args)
    ik_commands_left = torch.zeros(scene.num_envs, diff_ik_left.action_dim, device=robot.device)
    ik_commands_right = torch.zeros(scene.num_envs, diff_ik_right.action_dim, device=robot.device)
    
    # Configure robot entity settings for both arms
    # Both arms are part of the same robot, so we use different joint patterns
    left_arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"],
        body_names=["left_wrist_3_link"]
    )
    right_arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
        body_names=["right_wrist_3_link"]
    )
    
    # Resolve scene entities
    left_arm_cfg.resolve(scene)
    right_arm_cfg.resolve(scene)
    
    # Get end-effector Jacobian indices
    # For fixed base robots, frame index is one less than body index
    if robot.is_fixed_base:
        ee_jacobi_idx_left = left_arm_cfg.body_ids[0] - 1
        ee_jacobi_idx_right = right_arm_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx_left = left_arm_cfg.body_ids[0]
        ee_jacobi_idx_right = right_arm_cfg.body_ids[0]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    
    # Print info
    print("\n" + "="*60)
    print("Dual-Arm Differential IK Control")
    print("  Arms will move to goal immediately")
    print("  Change goals by restarting with --goal <index>")
    print("  Press Ctrl+C to quit")
    print("="*60 + "\n")
    
    # Helper function to set new goal
    def set_goal(goal_idx):
        ik_commands_left[:] = left_goals[goal_idx]
        ik_commands_right[:] = right_goals[goal_idx]
        diff_ik_left.set_command(ik_commands_left)
        diff_ik_right.set_command(ik_commands_right)
        print(f"[GOAL {goal_idx}/{num_goals-1}] Left: {left_goals[goal_idx][:3].tolist()}, Right: {right_goals[goal_idx][:3].tolist()}")
    
    # Helper function to reset robot
    def reset_robot():
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()
        diff_ik_left.reset()
        diff_ik_right.reset()
        print("[INFO] Robot reset to home position")
    
    # Track current goal index from command line
    current_goal_idx = min(args_cli.goal, len(left_goals) - 1)
    num_goals = len(left_goals)
    
    # Set the goal commands
    ik_commands_left[:] = left_goals[current_goal_idx]
    ik_commands_right[:] = right_goals[current_goal_idx]
    diff_ik_left.set_command(ik_commands_left)
    diff_ik_right.set_command(ik_commands_right)
    
    # Print goal information
    print(f"[GOAL {current_goal_idx}/{num_goals-1}]")
    print(f"  Left arm target:  {left_goals[current_goal_idx][:3].tolist()}")
    print(f"  Right arm target: {right_goals[current_goal_idx][:3].tolist()}")
    print("\n[INFO] Starting movement to goal...\n")
    
    # Simulation loop
    while simulation_app.is_running():
            # ===== LEFT ARM IK =====
            # Get Jacobian and poses
            jacobian_left = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx_left, :, left_arm_cfg.joint_ids]
            ee_pose_w_left = robot.data.body_pose_w[:, left_arm_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos_left = robot.data.joint_pos[:, left_arm_cfg.joint_ids]
            
            # Compute end-effector pose in root frame
            ee_pos_b_left, ee_quat_b_left = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7]
            )
            
            # Compute joint commands
            joint_pos_des_left = diff_ik_left.compute(ee_pos_b_left, ee_quat_b_left, jacobian_left, joint_pos_left)
            
            # ===== RIGHT ARM IK =====
            jacobian_right = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx_right, :, right_arm_cfg.joint_ids]
            ee_pose_w_right = robot.data.body_pose_w[:, right_arm_cfg.body_ids[0]]
            joint_pos_right = robot.data.joint_pos[:, right_arm_cfg.joint_ids]
            
            ee_pos_b_right, ee_quat_b_right = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7]
            )
            
            joint_pos_des_right = diff_ik_right.compute(ee_pos_b_right, ee_quat_b_right, jacobian_right, joint_pos_right)
            
            # ===== APPLY COMMANDS =====
            # Combine both arms for joint commands
            robot.set_joint_position_target(joint_pos_des_left, left_arm_cfg.joint_ids)
            robot.set_joint_position_target(joint_pos_des_right, right_arm_cfg.joint_ids)
            robot.write_data_to_sim()
            
            # Perform step
            sim.step()
            
            # Update buffers
            scene.update(sim_dt)
            
            # Update visualization markers
            ee_pose_w_left = robot.data.body_state_w[:, left_arm_cfg.body_ids[0], 0:7]
            ee_pose_w_right = robot.data.body_state_w[:, right_arm_cfg.body_ids[0], 0:7]
            
            ee_marker_left.visualize(ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7])
            goal_marker_left.visualize(ik_commands_left[:, 0:3] + scene.env_origins, ik_commands_left[:, 3:7])
            
            ee_marker_right.visualize(ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7])
            goal_marker_right.visualize(ik_commands_right[:, 0:3] + scene.env_origins, ik_commands_right[:, 3:7])


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
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.5, 0.0])
    
    # Design scene - use custom dual-arm URDF configuration
    scene_cfg = DualArmSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    print("[INFO] Setup complete!")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    print(f"[INFO] Sync mode: {args_cli.sync_mode}")
    print("[INFO] Use --sync_mode flag for synchronized arm movements")
    print("[INFO] Press Ctrl+C to exit")
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
