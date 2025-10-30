#!/usr/bin/env python3
"""
UR5e Pick and Place Simulation

This script demonstrates a UR5e robot performing pick and place operations.
It uses camera data to detect objects and plan manipulation trajectories.

Command-line arguments:
  --display-images    Enable visualization of intermediate camera images and detections
                     (default: disabled for better performance)
                     
                     When enabled, displays:
                     - Camera RGB image with detected box center pixel
                     - Height map (3D projection colored by height)
                     
                     Images are also saved to camera_image/ directory

Example usage:
  python main_ure5.py                    # Run without visualizations
  python main_ure5.py --display-images   # Run with visualizations enabled
"""

from unittest import skip
import argparse
from isaacsim import SimulationApp

# Parse command line arguments
parser = argparse.ArgumentParser(description="UR5e Pick and Place Simulation")
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


# Save the depth image to a file
save_root_depth = os.path.join(os.getcwd(), "camera_image/depth.png")
save_root_rgb = os.path.join(os.getcwd(), "camera_image/rgb.png")
save_root_semantic = os.path.join(os.getcwd(), "camera_image/semantic.png")


# Define the path to the objects directory
PATH_to_objects = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "objects"
)
print(PATH_to_objects)

# Define the number of objects
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

# Create PickPlace controller
pick_and_place_controller = CustomPickPlaceController(
    name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e
)

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
# viewport.set_active_camera("/World/OverheadCamera")
viewport.set_active_camera("/OmniverseKit_Persp")

# Declare found_obj to check if the target object is found (initially not found, so False)
found_obj = False


print("-" * 60)
print("Starting Push Manipulation Simulation")
print(f"Display images: {'ENABLED' if DISPLAY_INTERMEDIATE_IMAGES else 'DISABLED'}")
print("-" * 60)

first_observation = None
last_observation = None
start_point = None
end_point = None
initial_marker_position = None
marker_cube_created = False

push_started = False
closest_distance_to_target = float('inf')

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
            closest_distance_to_target = float('inf')
            marker_cube_created = False

        if first_observation is None and my_world.current_time_step_index > 100:
            first_observation = my_task.get_observations()
            
            rgb_image = first_observation["rgb_image"]
            center_u = rgb_image.shape[1] // 2
            center_v = rgb_image.shape[0] // 2
            
            # Get box center by converting center pixel to world coordinates
            box_center = my_task.pixel_to_robot_space(
                u=center_u,
                v=center_v,
                camera=camera_ortho,
                display=DISPLAY_INTERMEDIATE_IMAGES,
                rgb_image=first_observation["rgb_image"],
                depth_image=first_observation["depth_image"],
                heightmap=first_observation["height_map"],
            )
            
            marker_position = box_center.copy()
            marker_position[2] = max(marker_position[2], 0.06)
            print(f"Box detected: [{box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}]")
            
            if not marker_cube_created:
                try:
                    marker_cube = DynamicCuboid(
                        prim_path="/World/MarkerCube",
                        name="marker_cube",
                        position=marker_position,
                        scale=np.array([0.05, 0.05, 0.05]),
                        color=np.array([1.0, 0.0, 0.0]),
                        mass=0.001,
                    )
                    my_world.scene.add(marker_cube)
                    marker_cube_created = True
                    print("Marker cube spawned")
                except Exception as e:
                    print(f"Could not spawn marker: {e}")
            
            if DISPLAY_INTERMEDIATE_IMAGES:
                display_heightmap(first_observation["height_map"], name="Height Map")
            
            # Calculate push start and end positions
            start_point = box_center.copy()
            start_point[0] -= 0.05  # 5cm to left of box
            start_point[1] = 0.0
            start_point[2] = max(marker_position[2], 0.21)  # Min 21cm base height
            
            end_point = start_point.copy()
            end_point[0] += 0.10  # Push 10cm in +X direction
            
            push_vector = end_point - start_point
            print(f"Push: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}] → [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"Vector: [{push_vector[0]:.3f}, {push_vector[1]:.3f}, {push_vector[2]:.3f}] ({np.linalg.norm(push_vector)*100:.1f}cm)")
            
            initial_marker_position = marker_position.copy()
            cv2.imwrite(save_root_depth, first_observation["depth_image"])
            cv2.imwrite(save_root_rgb, first_observation["rgb_image"])

        if start_point is not None and end_point is not None and not push_controller.is_done():
            push_started = True
            ee_position, _ = my_ur5e.end_effector.get_world_pose()
            if initial_marker_position is not None:
                distance_to_marker = np.linalg.norm(ee_position[:2] - initial_marker_position[:2])
                closest_distance_to_target = min(closest_distance_to_target, distance_to_marker)
            
            if my_world.current_time_step_index % 100 == 0:
                print(f"Step {my_world.current_time_step_index}: EE=[{ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f}], Closest={closest_distance_to_target*100:.1f}cm")
            
            actions = push_controller.forward(push_start_position=start_point, push_end_position=end_point)

        if push_controller.is_done() and last_observation is None and push_started:
            print("\n" + "="*60)
            print("PUSH SEQUENCE COMPLETED")
            print("="*60)
            
            # Get final marker cube position and analyze if we have initial position
            if initial_marker_position is not None and start_point is not None and end_point is not None:
                try:
                    marker_obj = my_world.scene.get_object("marker_cube")
                    if marker_obj is None:
                        raise ValueError("Marker cube not found in scene")
                    final_marker_position, _ = marker_obj.get_world_pose()
                except Exception as e:
                    print(f"ERROR: Could not get marker cube position: {e}")
                    print("Cannot analyze push results without marker cube.")
                    # Just record the observation and skip analysis
                    if last_observation is None:
                        last_observation = my_task.get_observations()
                        cv2.imwrite(save_root_depth, last_observation["depth_image"])
                        cv2.imwrite(save_root_rgb, last_observation["rgb_image"])
                        if "goal_mask" in last_observation:
                            cv2.imwrite(save_root_semantic, last_observation["goal_mask"])
                else:
                    displacement = final_marker_position - initial_marker_position
                    horizontal_distance = np.linalg.norm(displacement[:2])
                    
                    print(f"Initial: [{initial_marker_position[0]:.3f}, {initial_marker_position[1]:.3f}, {initial_marker_position[2]:.3f}]")
                    print(f"Final:   [{final_marker_position[0]:.3f}, {final_marker_position[1]:.3f}, {final_marker_position[2]:.3f}]")
                    print(f"Moved: {horizontal_distance*100:.2f}cm (XY plane)")
                    print(f"Closest approach: {closest_distance_to_target*100:.2f}cm")
                    
                    success = closest_distance_to_target < 0.15 and horizontal_distance > 0.01
                    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
                    print("="*60)
            
            if last_observation is None:
                last_observation = my_task.get_observations()
                cv2.imwrite(save_root_depth, last_observation["depth_image"])
                cv2.imwrite(save_root_rgb, last_observation["rgb_image"])

        if actions is not None:
            articulation_controller.apply_action(actions)


simulation_app.close()
