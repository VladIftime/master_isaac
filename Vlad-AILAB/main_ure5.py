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
    end_effector_offset=[0, 0, 0.09],
    original_position=[0.5, 0, 0.3],
)


# Specify the view point in the GUI (when converting from Depth camera view to Perspective view, it is generally easier to see)
viewport = get_active_viewport()
viewport.set_active_camera("/World/OverheadCamera")
# viewport.set_active_camera("/OmniverseKit_Persp")

# Declare found_obj to check if the target object is found (initially not found, so False)
found_obj = False


print("---------------------------------Start simulation---------------------------------")
print(f"Display intermediate images: {'ENABLED' if DISPLAY_INTERMEDIATE_IMAGES else 'DISABLED'}")
if DISPLAY_INTERMEDIATE_IMAGES:
    print("  Will display: RGB image with detections + Height map visualization")
    print("  Images will be saved to: camera_image/")
else:
    print("  (Use --display-images flag to enable visualization)")
print("----------------------------------------------------------------------------------")

first_observation = None
last_observation = None
start_point = None
initial_marker_position = None

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        actions = None
        rgb_image, depth_image, distance_image = None, None, None

        if my_world.current_time_step_index == 0:
            my_world.reset()
            pick_and_place_controller.reset()

            current_joint_positions = (my_ur5e.get_joint_positions(),)
            end_effector_offset = (np.array([0, 0, 0.9]),)

        if first_observation is None and my_world.current_time_step_index > 100:
            first_observation = my_task.get_observations()
            
            # Get image dimensions to find center
            rgb_image = first_observation["rgb_image"]
            height, width = rgb_image.shape[:2]
            center_u = width // 2
            center_v = height // 2
            
            print(f"\n{'='*60}")
            print(f"Detecting box from camera at image center ({center_u}, {center_v})")
            print(f"{'='*60}\n")
            
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
            
            print(f"\nBox center detected at: [{box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}]")
            
            # Spawn a marker cube at the detected center (5cm above platform)
            marker_position = box_center.copy()
            marker_position[2] = 0.07  # 5cm above the platform base (platform center is at 0.025, so 0.05 is above it)
            
            print(f"Spawning marker cube at: [{marker_position[0]:.3f}, {marker_position[1]:.3f}, {marker_position[2]:.3f}]")
            
            # Create a small 1cm x 1cm x 1cm cube as a marker
            marker_cube = DynamicCuboid(
                prim_path="/World/MarkerCube",
                name="marker_cube",
                position=marker_position,
                scale=np.array([0.05, 0.05, 0.05]),  # 1cm x 1cm x 1cm
                color=np.array([1.0, 0.0, 0.0]),  # Red color for visibility
                mass=0.001,  # Very light weight
            )
            my_world.scene.add(marker_cube)
            print("Marker cube spawned successfully!")
            
            # Display heightmap if visualization is enabled
            if DISPLAY_INTERMEDIATE_IMAGES:
                print("Displaying heightmap visualization...")
                display_heightmap(
                    first_observation["height_map"],
                    name="Height Map - Top View (Z color-coded)"
                )
            
            # First point: at the box center (same height as marker)
            start_point = marker_position.copy()
            
            # Second point: 5cm forward in Y direction from first point
            end_point = start_point.copy()
            end_point[1] += 0.05  # 5cm forward in Y
            
            print(f"Start point (at marker): [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"End point (+5cm Y):      [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
            print(f"End effector offset:     {push_controller._end_effector_offset}")
            print(f"Marker cube position:    [{marker_position[0]:.3f}, {marker_position[1]:.3f}, {marker_position[2]:.3f}]")
            print(f"{'='*60}\n")
            
            # Store marker position for comparison later
            initial_marker_position = marker_position.copy()
            
            # Save images for inspection
            cv2.imwrite(save_root_depth, first_observation["depth_image"])
            cv2.imwrite(save_root_rgb, first_observation["rgb_image"])
            if "goal_mask" in first_observation:
                cv2.imwrite(save_root_semantic, first_observation["goal_mask"])
            
            # Save heightmap visualization if enabled
            if DISPLAY_INTERMEDIATE_IMAGES:
                save_root_heightmap = os.path.join(os.getcwd(), "camera_image/heightmap.png")
                plt.figure(figsize=(10, 10))
                plt.scatter(
                    first_observation["height_map"][:, 0],
                    first_observation["height_map"][:, 1],
                    c=first_observation["height_map"][:, 2],
                    cmap="viridis",
                    s=1
                )
                plt.colorbar(label="Height (Z)")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("Height Map - Top View")
                plt.savefig(save_root_heightmap, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Heightmap saved to: {save_root_heightmap}")

        if start_point is not None and end_point is not None:
            if my_world.current_time_step_index % 100 == 0:
                ee_position, ee_orientation = my_ur5e.end_effector.get_world_pose()
                print(f"[Step {my_world.current_time_step_index}] End effector at: [{ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f}]")
                print(f"[Step {my_world.current_time_step_index}] Target start: [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            
            actions = push_controller.forward(
                push_start_position=start_point,
                push_end_position=end_point,
            )

        if push_controller.is_done() and last_observation is None:
            print("\n" + "="*60)
            print("Push completed! Analyzing results...")
            print("="*60)
            
            # Get final marker cube position and analyze if we have initial position
            if initial_marker_position is not None and start_point is not None and end_point is not None:
                final_marker_position, _ = my_world.scene.get_object("marker_cube").get_world_pose()
                
                # Calculate displacement
                displacement = final_marker_position - initial_marker_position
                distance_moved = np.linalg.norm(displacement)
                
                print(f"\nMarker cube analysis:")
                print(f"  Initial position: [{initial_marker_position[0]:.3f}, {initial_marker_position[1]:.3f}, {initial_marker_position[2]:.3f}]")
                print(f"  Final position:   [{final_marker_position[0]:.3f}, {final_marker_position[1]:.3f}, {final_marker_position[2]:.3f}]")
                print(f"  Displacement:     [{displacement[0]:.3f}, {displacement[1]:.3f}, {displacement[2]:.3f}]")
                print(f"  Distance moved:   {distance_moved*100:.2f} cm")
                
                # Check if push was successful (cube moved in the expected direction)
                expected_direction = end_point - start_point
                expected_direction_normalized = expected_direction / np.linalg.norm(expected_direction)
                actual_direction = displacement[:2]  # Only X and Y
                
                if distance_moved > 0.01:  # Moved more than 1cm
                    actual_direction_normalized = actual_direction / np.linalg.norm(actual_direction)
                    dot_product = np.dot(expected_direction_normalized[:2], actual_direction_normalized)
                    print(f"\nPush direction alignment: {dot_product:.2f} (1.0 = perfect, -1.0 = opposite)")
                    
                    if dot_product > 0.7:
                        print("✓ Push was successful! Cube moved in expected direction.")
                    else:
                        print("✗ Push direction was off. Cube moved in unexpected direction.")
                else:
                    print("✗ Cube barely moved. Push may not have made contact.")
            else:
                print("Warning: Required positions not set, skipping detailed analysis")
            
            print("="*60 + "\n")
            
            last_observation = my_task.get_observations()
            cv2.imwrite(save_root_depth, last_observation["depth_image"])
            cv2.imwrite(save_root_rgb, last_observation["rgb_image"])
            if "goal_mask" in last_observation:
                cv2.imwrite(save_root_semantic, last_observation["goal_mask"])

        if actions is not None:
            articulation_controller.apply_action(actions)


simulation_app.close()
