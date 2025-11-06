#!/usr/bin/env python3
"""
UR5e Push and Grasp Simulation

This script demonstrates a UR5e robot performing push manipulation followed by grasping.
It creates a cube at the center of the platform, pushes it, then grasps it.
Height maps are saved at three stages: before pushing, after pushing, and after grasping.

Command-line arguments:
  --display-images    Enable visualization of intermediate camera images and detections
                     (default: disabled for better performance)

Example usage:
  python main_push_grasp.py                    # Run without visualizations
  python main_push_grasp.py --display-images   # Run with visualizations enabled
"""

import argparse
from isaacsim import SimulationApp

# Parse command line arguments
parser = argparse.ArgumentParser(description="UR5e Push and Grasp Simulation")
parser.add_argument(
    "--display-images",
    action="store_true",
    default=False,
    help="Display intermediate camera images and visualizations",
)
args, unknown = parser.parse_known_args()

# Flag to control display of intermediate pictures
DISPLAY_INTERMEDIATE_IMAGES = args.display_images

# Initialize the simulation application with a GUI
simulation_app = SimulationApp({"headless": False})

# Disable Materials and Lights: To get simpler/faster rendering
import carb

carb.settings.get_settings().set_bool("/rtx/post/ambientOcclusion/enabled", False)
carb.settings.get_settings().set_bool("/rtx/post/bloom/enabled", False)
carb.settings.get_settings().set_bool("/rtx/post/depthOfField/enabled", False)
carb.settings.get_settings().set_bool("/rtx/reflections/enabled", False)
carb.settings.get_settings().set_bool("/rtx/shadows/enabled", False)

import sys

import isaacsim
from isaacsim.core.api import World
from isaacsim.core.utils.rotations import euler_angles_to_quat
import isaacsim.core.utils.stage as stage_utils
from omni.kit.viewport.utility import get_active_viewport
from dev_utils.point_cloud_utils import display_heightmap

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from isaacsim.core.api.materials import DeformableMaterial, DeformableMaterialView
from isaacsim.core.api.objects import DynamicCuboid

# Add the path to the custom utility scripts
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils_robots.controllers.pick_place_controller_robotiq import PickPlaceController
from utils_robots.controllers.pick_place_controller_ext import CustomPickPlaceController
from utils_robots.controllers.push_manipulation_controller import (
    PushManipulationController,
)

from utils_robots.controllers.basic_manipulation_controller import (
    BasicManipulationController,
)
from utils_robots.controllers.RMPFflow_pickplace import RMPFlowController
from utils_robots.tasks.pick_place_task import UR5ePickPlace


# Create directory for saving height maps
heightmap_dir = os.path.join(os.getcwd(), "heightmaps")
os.makedirs(heightmap_dir, exist_ok=True)

# Save paths for height maps
save_heightmap_before_push = os.path.join(heightmap_dir, "heightmap_before_push.npy")
save_heightmap_after_push = os.path.join(heightmap_dir, "heightmap_after_push.npy")
save_heightmap_after_grasp = os.path.join(heightmap_dir, "heightmap_after_grasp.npy")

# Save paths for images
save_root_depth = os.path.join(os.getcwd(), "camera_image/depth.png")
save_root_rgb = os.path.join(os.getcwd(), "camera_image/rgb.png")
save_root_semantic = os.path.join(os.getcwd(), "camera_image/semantic.png")


# Define the path to the objects directory
PATH_to_objects = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "objects"
)
print(PATH_to_objects)

# Define the number of objects (0, we'll create our own cube)
number_of_objects = 0

# Try to load the stage
stage_usd_path = os.path.realpath("ure5_stage_basket.usd")

# Initialize the simulation world
my_world = World(physics_dt=0.01, stage_units_in_meters=1.0)

my_task = UR5ePickPlace(number_of_objects=number_of_objects)
# Add the task to the simulation world and reset the world
my_world.add_task(my_task)
my_world.reset()

# Obtain ur5e and camera from the task
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
camera_ortho = my_task.get_camera_ortho()
camera_pointcloud = my_task.get_camera_pointcloud()

# Log robot base position for debugging coordinate issues
robot_base_pos, robot_base_ori = my_ur5e.get_world_pose()
print(f"\n{'='*70}")
print(f"ROBOT CONFIGURATION")
print(f"{'='*70}")
print(f"Robot base position (world frame): [{robot_base_pos[0]:.3f}, {robot_base_pos[1]:.3f}, {robot_base_pos[2]:.3f}]")
print(f"Robot base orientation (world):    [{robot_base_ori[0]:.3f}, {robot_base_ori[1]:.3f}, {robot_base_ori[2]:.3f}, {robot_base_ori[3]:.3f}]")
print(f"{'='*70}\n")

# Create PickPlace controller with RMPFlowController
# Ensure robot has USD path for RMPFlow configuration
if not hasattr(my_ur5e, '_usd_path'):
    # Get USD path from task
    working_dir = os.path.dirname(os.path.realpath(__file__))
    ur5e_usd_path = os.path.join(working_dir, "utils_robots", "tasks", "ur5e_handeye_gripper.usd")
    if os.path.isfile(ur5e_usd_path):
        my_ur5e._usd_path = ur5e_usd_path
        print(f"Set robot USD path: {ur5e_usd_path}")

pick_and_place_controller = CustomPickPlaceController(
    name="pick_place_controller", 
    gripper=my_ur5e.gripper, 
    robot_articulation=my_ur5e
)
print("PickPlace controller created with RMPFlowController")

# Declare instance for robot control (PD control)
articulation_controller = my_ur5e.get_articulation_controller()


# Create EndEffector Controller for push manipulation
push_controller = PushManipulationController(
    name="push_manipulation_controller",
    gripper=my_ur5e.gripper,
    robot_articulation=my_ur5e,
    end_effector_offset=np.array([0, 0, 0.11]),
    original_position=np.array([0.5, 0, 0.3]),
)


# Specify the view point in the GUI (when converting from Depth camera view to Perspective view, it is generally easier to see)
viewport = get_active_viewport()
viewport.set_active_camera("/OmniverseKit_Persp")

# State variables
cube_created = False
cube_position = None
push_started = False
grasp_started = False
heightmap_before_push_saved = False
heightmap_after_push_saved = False
heightmap_after_grasp_saved = False

# Observation storage
observation_before_push = None
observation_after_push = None
observation_after_grasp = None

# Push and grasp planning variables
start_point = None
end_point = None
picking_position = None
placing_position = None

def world_to_robot_frame(world_pos: np.ndarray, robot_base_pos: np.ndarray, robot_base_ori: np.ndarray) -> np.ndarray:
    """Convert world frame position to robot local frame.
    
    Args:
        world_pos: Position in world frame [x, y, z]
        robot_base_pos: Robot base position in world frame [x, y, z]
        robot_base_ori: Robot base orientation quaternion in world frame [w, x, y, z]
    
    Returns:
        Position in robot local frame [x, y, z]
    """
    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat([robot_base_ori[1], robot_base_ori[2], robot_base_ori[3], robot_base_ori[0]]).as_matrix()
    
    # Translate to robot base origin
    pos_relative = world_pos - robot_base_pos
    
    # Rotate to robot frame (inverse rotation)
    rot_matrix_inv = rot_matrix.T
    robot_frame_pos = rot_matrix_inv @ pos_relative
    
    return robot_frame_pos

print("-" * 60)
print("Starting Push and Grasp Simulation")
print(f"Display images: {'ENABLED' if DISPLAY_INTERMEDIATE_IMAGES else 'DISABLED'}")
print("-" * 60)

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        actions = None
        rgb_image, depth_image, distance_image = None, None, None

        if my_world.current_time_step_index == 0:
            my_world.reset()
            pick_and_place_controller.reset()
            push_controller.reset()
            
            push_started = False
            grasp_started = False
            cube_created = False
            heightmap_before_push_saved = False
            heightmap_after_push_saved = False
            heightmap_after_grasp_saved = False
            cube_position = None
            start_point = None
            end_point = None
            picking_position = None
            placing_position = None

        # Create cube at center of platform after initial settling
        if not cube_created and my_world.current_time_step_index > 100:
            # Platform center is at [0.5, 0, 0.025] based on pick_place_task.py
            platform_center = np.array([0.5, 0, 0.025])
            cube_size = 0.05  # 5cm cube
            cube_height = cube_size / 2
            cube_position = platform_center.copy()
            cube_position[2] = platform_center[2] + cube_height  # Place on top of platform
            
            try:
                target_cube = DynamicCuboid(
                    prim_path="/World/TargetCube",
                    name="target_cube",
                    position=cube_position,
                    scale=np.array([cube_size, cube_size, cube_size]),
                    color=np.array([0.0, 1.0, 0.0]),  # Green cube
                    mass=0.01,
                )
                my_world.scene.add(target_cube)
                cube_created = True
                print(f"\n{'='*60}")
                print(f"CUBE CREATED")
                print(f"{'='*60}")
                print(f"Position: [{cube_position[0]:.3f}, {cube_position[1]:.3f}, {cube_position[2]:.3f}]")
                print(f"Size: {cube_size*100:.1f}cm")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Could not spawn cube: {e}")

        # Wait for cube to settle, then capture initial observation
        if cube_created and not heightmap_before_push_saved and my_world.current_time_step_index > 200:
            observation_before_push = my_task.get_observations()
            heightmap_before_push = observation_before_push["height_map"]
            
            # Save height map
            np.save(save_heightmap_before_push, heightmap_before_push)
            
            # Save RGB and depth images
            cv2.imwrite(save_root_rgb.replace(".png", "_before_push.png"), observation_before_push["rgb_image"])
            cv2.imwrite(save_root_depth.replace(".png", "_before_push.png"), observation_before_push["depth_image"])
            
            print(f"\n{'='*60}")
            print(f"HEIGHT MAP SAVED: BEFORE PUSH")
            print(f"{'='*60}")
            print(f"Saved to: {save_heightmap_before_push}")
            print(f"Height map shape: {heightmap_before_push.shape}")
            print(f"Number of points: {len(heightmap_before_push)}")
            print(f"RGB image saved: {save_root_rgb.replace('.png', '_before_push.png')}")
            print(f"Depth image saved: {save_root_depth.replace('.png', '_before_push.png')}")
            print(f"{'='*60}\n")
            
            heightmap_before_push_saved = True
            
            # Get cube center for push planning
            rgb_image = observation_before_push["rgb_image"]
            center_u = rgb_image.shape[1] // 2
            center_v = rgb_image.shape[0] // 2
            
            # Get box center by converting center pixel to world coordinates
            box_center = my_task.pixel_to_robot_space(
                u=center_u,
                v=center_v,
                camera=camera_ortho,
                display=DISPLAY_INTERMEDIATE_IMAGES,
                rgb_image=observation_before_push["rgb_image"],
                depth_image=observation_before_push["depth_image"],
                heightmap=observation_before_push["height_map"],
            )
            
            if DISPLAY_INTERMEDIATE_IMAGES:
                display_heightmap(observation_before_push["height_map"], name="Height Map - Before Push")
            
            # Calculate push start and end positions
            start_point = box_center.copy()
            start_point[0] -= 0.05  # 5cm to left of box
            start_point[1] = 0.0
            start_point[2] = max(box_center[2], 0.21)  # Min 21cm base height
            
            end_point = start_point.copy()
            end_point[0] += 0.10  # Push 10cm in +X direction
            
            push_vector = end_point - start_point
            print(f"\n{'='*60}")
            print(f"PUSH PLANNING")
            print(f"{'='*60}")
            print(f"Box center: [{box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}]")
            print(f"Push start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"Push end:   [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"Vector: [{push_vector[0]:.3f}, {push_vector[1]:.3f}, {push_vector[2]:.3f}] ({np.linalg.norm(push_vector)*100:.1f}cm)")
            print(f"{'='*60}\n")
            
            push_started = True

        # Execute push manipulation
        if push_started and not push_controller.is_done() and start_point is not None and end_point is not None:
            actions = push_controller.forward(push_start_position=start_point, push_end_position=end_point)

        # After push completes, save height map
        if push_controller.is_done() and push_started and not heightmap_after_push_saved:
            # Wait a few steps for physics to settle before capturing
            if my_world.current_time_step_index > 0:  # Ensure we're past initial steps
                observation_after_push = my_task.get_observations()
                heightmap_after_push = observation_after_push["height_map"]
                
                # Save height map
                np.save(save_heightmap_after_push, heightmap_after_push)
                
                # Save RGB and depth images
                cv2.imwrite(save_root_rgb.replace(".png", "_after_push.png"), observation_after_push["rgb_image"])
                cv2.imwrite(save_root_depth.replace(".png", "_after_push.png"), observation_after_push["depth_image"])
                
                print(f"\n{'='*60}")
                print(f"PUSH SEQUENCE COMPLETED")
                print(f"{'='*60}")
                print(f"HEIGHT MAP SAVED: AFTER PUSH")
                print(f"Saved to: {save_heightmap_after_push}")
                print(f"Height map shape: {heightmap_after_push.shape}")
                print(f"Number of points: {len(heightmap_after_push)}")
                print(f"RGB image saved: {save_root_rgb.replace('.png', '_after_push.png')}")
                print(f"Depth image saved: {save_root_depth.replace('.png', '_after_push.png')}")
                print(f"{'='*60}\n")
                
                if DISPLAY_INTERMEDIATE_IMAGES:
                    display_heightmap(observation_after_push["height_map"], name="Height Map - After Push")
                
                heightmap_after_push_saved = True
                
                # Reset pick and place controller before starting grasp sequence
                pick_and_place_controller.reset()
                print("PickPlace controller reset for grasp sequence")
                
                # Get new cube position for grasping
                rgb_image = observation_after_push["rgb_image"]
                center_u = rgb_image.shape[1] // 2
                center_v = rgb_image.shape[0] // 2
                
                # Get box center after push
                box_center_after_push = my_task.pixel_to_robot_space(
                    u=center_u,
                    v=center_v,
                    camera=camera_ortho,
                    display=DISPLAY_INTERMEDIATE_IMAGES,
                    rgb_image=observation_after_push["rgb_image"],
                    depth_image=observation_after_push["depth_image"],
                    heightmap=observation_after_push["height_map"],
                )
                
                # Plan grasp: pick from current position, place at same position (just lift)
                # Convert from world frame to robot local frame
                box_center_world = box_center_after_push.copy()
                placing_position_world = box_center_after_push.copy()
                placing_position_world[2] += 0.1  # Lift 10cm higher
                
                # Get current robot pose
                robot_base_pos_current, robot_base_ori_current = my_ur5e.get_world_pose()
                
                # Convert to robot local frame (required by PickPlaceController)
                picking_position = world_to_robot_frame(box_center_world, robot_base_pos_current, robot_base_ori_current)
                placing_position = world_to_robot_frame(placing_position_world, robot_base_pos_current, robot_base_ori_current)
                
                print(f"\n{'='*60}")
                print(f"GRASP PLANNING")
                print(f"{'='*60}")
                print(f"Box center after push (world): [{box_center_world[0]:.3f}, {box_center_world[1]:.3f}, {box_center_world[2]:.3f}]")
                print(f"Picking position (robot frame): [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                print(f"Placing position (robot frame): [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
                print(f"Controller using RMPFlowController for motion planning")
                print(f"{'='*60}\n")
                
                grasp_started = True

        # Execute grasp manipulation
        if grasp_started and not pick_and_place_controller.is_done() and picking_position is not None and placing_position is not None:
            current_joint_positions = my_ur5e.get_joints_state().positions
            
            # Ensure positions are numpy arrays (already in robot local frame)
            picking_pos = np.array(picking_position, dtype=np.float32)
            placing_pos = np.array(placing_position, dtype=np.float32)
            
            # Call forward with robot local frame positions
            # CustomPickPlaceController uses RMPFlowController internally for motion planning
            actions = pick_and_place_controller.forward(
                picking_position=picking_pos,
                placing_position=placing_pos,
                current_joint_positions=current_joint_positions,
            )

        # After grasp completes, save height map
        if pick_and_place_controller.is_done() and grasp_started and not heightmap_after_grasp_saved:
            # Wait a few steps for physics to settle before capturing
            if my_world.current_time_step_index > 0:  # Ensure we're past initial steps
                observation_after_grasp = my_task.get_observations()
                heightmap_after_grasp = observation_after_grasp["height_map"]
                
                # Save height map
                np.save(save_heightmap_after_grasp, heightmap_after_grasp)
                
                # Save RGB and depth images
                cv2.imwrite(save_root_rgb.replace(".png", "_after_grasp.png"), observation_after_grasp["rgb_image"])
                cv2.imwrite(save_root_depth.replace(".png", "_after_grasp.png"), observation_after_grasp["depth_image"])
                
                print(f"\n{'='*60}")
                print(f"GRASP SEQUENCE COMPLETED")
                print(f"{'='*60}")
                print(f"HEIGHT MAP SAVED: AFTER GRASP")
                print(f"Saved to: {save_heightmap_after_grasp}")
                print(f"Height map shape: {heightmap_after_grasp.shape}")
                print(f"Number of points: {len(heightmap_after_grasp)}")
                print(f"RGB image saved: {save_root_rgb.replace('.png', '_after_grasp.png')}")
                print(f"Depth image saved: {save_root_depth.replace('.png', '_after_grasp.png')}")
                print(f"{'='*60}\n")
                
                if DISPLAY_INTERMEDIATE_IMAGES:
                    display_heightmap(observation_after_grasp["height_map"], name="Height Map - After Grasp")
                
                heightmap_after_grasp_saved = True
                
                print(f"\n{'='*60}")
                print(f"ALL SEQUENCES COMPLETED")
                print(f"{'='*60}")
                print(f"Height maps saved:")
                print(f"  1. Before push:  {save_heightmap_before_push}")
                print(f"  2. After push:  {save_heightmap_after_push}")
                print(f"  3. After grasp: {save_heightmap_after_grasp}")
                print(f"{'='*60}\n")

        if actions is not None:
            articulation_controller.apply_action(actions)


simulation_app.close()

