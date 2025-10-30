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
    end_effector_offset=[0, 0, 0.11],
    original_position=[0.5, 0, 0.3],
)


# Specify the view point in the GUI (when converting from Depth camera view to Perspective view, it is generally easier to see)
viewport = get_active_viewport()
# viewport.set_active_camera("/World/OverheadCamera")
viewport.set_active_camera("/OmniverseKit_Persp")

# Declare found_obj to check if the target object is found (initially not found, so False)
found_obj = False


print("---------------------------------Start simulation---------------------------------")
print(f"Display intermediate images: {'ENABLED' if DISPLAY_INTERMEDIATE_IMAGES else 'DISABLED'}")
if DISPLAY_INTERMEDIATE_IMAGES:
    print("  Will display: RGB image with detections + Height map visualization")
    print("  Images will be saved to: camera_image/")
else:
    print("  (Use --display-images flag to enable visualization)")
print("")
print("Movement Speed Configuration:")
print("  Phase timings: [overhead, gripper, move_to_start, push, return]")
print("  Increment rates: [0.008, 0.010, 0.005, 0.003, 0.008] per step")
print("  → Overhead: ~125 steps (~2s at 60Hz)")
print("  → Move to start: ~200 steps (~3.3s at 60Hz)")
print("  → Push motion: ~333 steps (~5.5s at 60Hz)")
print("  → Total push sequence: ~10-12 seconds")
print("----------------------------------------------------------------------------------")

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
            
            # Reset state monitoring
            robot_state_valid = True
            push_started = False
            closest_distance_to_target = float('inf')
            last_joint_positions = None
            marker_cube_created = False

            current_joint_positions = (my_ur5e.get_joint_positions(),)
            end_effector_offset = (np.array([0, 0, 0.9]),)
            
            print("\n" + "="*60)
            print("SIMULATION RESET")
            print("="*60)

        if first_observation is None and my_world.current_time_step_index > 100:
            first_observation = my_task.get_observations()
            
            # DIAGNOSTIC: Check which frame is being used
            import carb
            from isaacsim.core.utils.prims import get_prim_at_path
            
            robot_prim_path = my_ur5e.prim_path
            carb.log_warn(f"\n{'='*70}")
            carb.log_warn(f"DIAGNOSTIC: USD Robot Structure")
            carb.log_warn(f"{'='*70}")
            carb.log_warn(f"Robot prim path: {robot_prim_path}")
            carb.log_warn(f"End effector prim: {my_ur5e.end_effector.prim_path}")
            
            # Check if tool0 exists
            tool0_exists = get_prim_at_path(f"{robot_prim_path}/tool0").IsValid()
            flange_exists = get_prim_at_path(f"{robot_prim_path}/flange").IsValid()
            
            carb.log_warn(f"tool0 exists: {tool0_exists}")
            carb.log_warn(f"flange exists: {flange_exists}")
            
            if not tool0_exists:
                carb.log_warn(f"⚠️  USD file: /home/vlad/IsaacLab/vlad/master_isaac/Vlad-AILAB/utils_robots/tasks/ur5e_handeye_gripper.usd")
                carb.log_warn(f"⚠️  This USD is missing 'tool0' frame - RMPFlow has been configured to use 'flange' instead")
            
            carb.log_warn(f"{'='*70}\n")
            
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
            
            # Spawn a marker cube at the detected center
            marker_position = box_center.copy()
            # Ensure marker is visible above the platform
            if marker_position[2] < 0.06:
                marker_position[2] = 0.06  # At least 6cm above ground
            
            print(f"Spawning marker cube at: [{marker_position[0]:.3f}, {marker_position[1]:.3f}, {marker_position[2]:.3f}]")
            
            # Check if marker cube already exists from previous attempt
            if not marker_cube_created:
                try:
                    existing_marker = my_world.scene.get_object("marker_cube")
                    if existing_marker is not None:
                        print("Marker cube already exists, updating position...")
                        existing_marker.set_world_pose(position=marker_position)
                        marker_cube_created = True
                        print("Marker cube position updated!")
                    else:
                        # Create new marker cube
                        marker_cube = DynamicCuboid(
                            prim_path="/World/MarkerCube",
                            name="marker_cube",
                            position=marker_position,
                            scale=np.array([0.05, 0.05, 0.05]),  # 5cm x 5cm x 5cm
                            color=np.array([1.0, 0.0, 0.0]),  # Red color for visibility
                            mass=0.001,  # Very light weight
                        )
                        my_world.scene.add(marker_cube)
                        marker_cube_created = True
                        print("Marker cube spawned successfully!")
                except Exception as e:
                    # If getting the object fails, try to create a new one
                    print(f"Note: Could not get existing marker: {e}")
                    try:
                        marker_cube = DynamicCuboid(
                            prim_path="/World/MarkerCube",
                            name="marker_cube",
                            position=marker_position,
                            scale=np.array([0.05, 0.05, 0.05]),  # 5cm x 5cm x 5cm
                            color=np.array([1.0, 0.0, 0.0]),  # Red color for visibility
                            mass=0.001,  # Very light weight
                        )
                        my_world.scene.add(marker_cube)
                        marker_cube_created = True
                        print("Marker cube spawned successfully!")
                    except Exception as e2:
                        print(f"ERROR: Could not spawn marker cube: {e2}")
                        print("Continuing without marker cube...")
            else:
                print("Marker cube already created this run, skipping...")
            
            # Display heightmap if visualization is enabled
            if DISPLAY_INTERMEDIATE_IMAGES:
                print("Displaying heightmap visualization...")
                display_heightmap(
                    first_observation["height_map"],
                    name="Height Map - Top View (Z color-coded)"
                )
            
            # First point: at safe reachable height above the box
            # UR5e has a kinematic singularity at X=0.5m, Y=0.0m (directly center)
            # SOLUTION: Push from the SIDE (Y = 0.15m) instead of center (Y = 0.0m)
            # This moves us away from the singularity and into a comfortable workspace
            start_point = box_center.copy()
            
            # Start 5cm to the right of the box
            start_point[0] -= 0.05
            # Move to the side to avoid singularity
            start_point[1] = 0.0  # 15cm to the right - much better kinematics!
            
            # At Y=0.15m, we can use a lower, more reasonable height
            MIN_EE_HEIGHT = 0.30  # 30cm EE height is fine when not at singularity
            gripper_offset_z = 0.09 if push_controller._end_effector_offset is not None else 0.09
            min_base_height = MIN_EE_HEIGHT - gripper_offset_z  # 30cm - 9cm = 21cm base
            
            # Use the higher of marker height or minimum safe height
            start_point[2] = max(marker_position[2], min_base_height)
            
            print(f"\n{'='*70}")
            print(f"KINEMATIC CONFIGURATION ANALYSIS")
            print(f"{'='*70}")
            print(f"📦 Box/Marker Z-height:        {marker_position[2]*100:.1f} cm")
            print(f"🎯 Push start base target Z:   {start_point[2]*100:.1f} cm")
            print(f"")
            
            # Second point: Push OUTWARD (positive X) to push cube away from center 
            end_point = start_point.copy()
            end_point[0] += 0.10  # 10cm outward (right) in X direction
            
            # Calculate actual EE target for kinematic analysis
            if push_controller._end_effector_offset is not None:
                actual_ee_start = start_point + push_controller._end_effector_offset
            else:
                actual_ee_start = start_point.copy()
                
            print(f"📍 Start position (BASE):      [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
            print(f"📍 Start position (EE target): [{actual_ee_start[0]:.3f}, {actual_ee_start[1]:.3f}, {actual_ee_start[2]:.3f}] ({actual_ee_start[2]*100:.1f} cm high)")
            print(f"")
            print(f"⚠️  KINEMATIC NOTE:")
            print(f"   UR5e at X=0.5m, Y=0.0m has a singularity - AVOIDED by using Y=0.15m")
            print(f"   Current Y position: {actual_ee_start[1]*100:.1f} cm (OFF-CENTER - GOOD!)")
            print(f"   Current Z height: {actual_ee_start[2]*100:.1f} cm")
            print(f"   ✓ This configuration should be kinematically reachable")
            print(f"{'='*70}\n")
            
            # Calculate actual end effector target (with offset)
            actual_ee_target = start_point.copy()
            if push_controller._end_effector_offset is not None:
                actual_ee_target[2] += push_controller._end_effector_offset[2]
            else:
                actual_ee_target[2] += 0.09  # Default offset
            
            print(f"\n{'='*60}")
            print(f"🎯 PUSH CONFIGURATION")
            print(f"{'='*60}")
            print(f"📍 Marker position:      [{marker_position[0]:.3f}, {marker_position[1]:.3f}, {marker_position[2]:.3f}] ({marker_position[2]*100:.1f}cm high)")
            print(f"")
            print(f"🎬 Start configuration:")
            print(f"   Base target:          [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}] ({start_point[2]*100:.1f}cm high)")
            print(f"   Gripper offset:       {push_controller._end_effector_offset}")
            print(f"   End Effector target:  [{actual_ee_target[0]:.3f}, {actual_ee_target[1]:.3f}, {actual_ee_target[2]:.3f}] ({actual_ee_target[2]*100:.1f}cm high)")
            print(f"   Height above marker:  {(actual_ee_target[2] - marker_position[2])*100:.1f}cm")
            print(f"")
            print(f"🏁 End configuration:")
            print(f"   End point (-10cm Y):  [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}] (pushing inward)")
            print(f"")
            push_vector = end_point - start_point
            print(f"➡️  Push vector:          [{push_vector[0]:.3f}, {push_vector[1]:.3f}, {push_vector[2]:.3f}] ({np.linalg.norm(push_vector)*100:.1f}cm)")
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

        # Control push
        if start_point is not None and end_point is not None and not push_controller.is_done():
            push_started = True
            
            # Track closest approach to target
            ee_position, _ = my_ur5e.end_effector.get_world_pose()
            if initial_marker_position is not None:
                distance_to_marker = np.linalg.norm(ee_position[:2] - initial_marker_position[:2])
                if distance_to_marker < closest_distance_to_target:
                    closest_distance_to_target = distance_to_marker
            
            # Periodic status updates
            if my_world.current_time_step_index % 100 == 0:
                print(f"\n{'─'*60}")
                print(f"📊 STATUS UPDATE - Step {my_world.current_time_step_index}")
                print(f"{'─'*60}")
                print(f"  Current EE Position:  [{ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f}]")
                print(f"  Push Start Target:    [{start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f}]")
                print(f"  Push End Target:      [{end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f}]")
                print(f"  Closest to Marker:    {closest_distance_to_target*100:.2f}cm")
                
                # Calculate distance to targets
                dist_to_start = np.linalg.norm(ee_position - start_point)
                dist_to_end = np.linalg.norm(ee_position - end_point)
                print(f"  Distance to Start:    {dist_to_start*100:.2f}cm")
                print(f"  Distance to End:      {dist_to_end*100:.2f}cm")
                
                # Show which phase we're likely in
                if dist_to_start > 0.10:
                    print(f"  Phase: 🚶 Approaching start position")
                elif dist_to_end > 0.10:
                    print(f"  Phase: 👉 Executing push")
                else:
                    print(f"  Phase: 🏠 Near end position / returning")
                print(f"{'─'*60}")
            
            actions = push_controller.forward(
                push_start_position=start_point,
                push_end_position=end_point,
            )

        if push_controller.is_done() and last_observation is None and push_started:
            print("\n" + "="*60)
            print("✅ PUSH SEQUENCE COMPLETED - Analyzing Results...")
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
                    # Calculate displacement
                    displacement = final_marker_position - initial_marker_position
                    distance_moved = np.linalg.norm(displacement)
                    
                    print(f"\n📦 MARKER CUBE ANALYSIS:")
                    print(f"{'─'*60}")
                    print(f"  Initial position:     [{initial_marker_position[0]:.3f}, {initial_marker_position[1]:.3f}, {initial_marker_position[2]:.3f}]")
                    print(f"  Final position:       [{final_marker_position[0]:.3f}, {final_marker_position[1]:.3f}, {final_marker_position[2]:.3f}]")
                    print(f"  Displacement (XYZ):   [{displacement[0]:.3f}, {displacement[1]:.3f}, {displacement[2]:.3f}]")
                    print(f"  Total distance moved: {distance_moved*100:.2f} cm")
                    print(f"  Closest approach:     {closest_distance_to_target*100:.2f} cm")
                    print(f"{'─'*60}")
                    
                    # Determine success based on multiple criteria
                    success = True
                    reasons = []
                    
                    # Criterion 1: Robot reached close to the marker
                    if closest_distance_to_target > 0.15:  # More than 15cm away
                        success = False
                        reasons.append(f"Robot never got close to marker (closest: {closest_distance_to_target*100:.2f}cm)")
                    
                    # Criterion 2: Marker actually moved IN THE HORIZONTAL PLANE (X,Y)
                    horizontal_displacement = displacement[:2]  # Only X and Y
                    horizontal_distance = np.linalg.norm(horizontal_displacement)
                    
                    print(f"  Horizontal movement (XY): {horizontal_distance*100:.2f} cm")
                    
                    if horizontal_distance < 0.01:  # Less than 1cm horizontal movement
                        success = False
                        reasons.append(f"No horizontal movement (<1cm) - cube not pushed")
                    
                    # Criterion 3: Movement direction matches expected (only if moved)
                    if horizontal_distance > 0.005:  # Only check direction if it moved >0.5cm
                        expected_direction = end_point - start_point
                        expected_direction_normalized = expected_direction / np.linalg.norm(expected_direction)
                        actual_direction_normalized = horizontal_displacement / horizontal_distance
                        dot_product = np.dot(expected_direction_normalized[:2], actual_direction_normalized)
                        print(f"  Direction alignment: {dot_product:.2f} (1.0 = perfect)")
                        
                        if dot_product < 0.5:  # Less than 60 degree alignment
                            success = False
                            reasons.append(f"Wrong direction (alignment: {dot_product:.2f})")
                    else:
                        print(f"  Direction alignment: N/A (no movement)")
                    
                    # Print verdict
                    print(f"\n{'═'*60}")
                    if success:
                        print("✅ PUSH SUCCESSFUL! 🎉")
                        print(f"{'═'*60}")
                        print("  All criteria met:")
                        print(f"    ✓ Robot approached marker (< 15cm)")
                        print(f"    ✓ Marker moved horizontally (> 1cm in XY plane)")
                        print(f"    ✓ Movement direction correct (> 50% aligned)")
                    else:
                        print("❌ PUSH FAILED!")
                        print(f"{'═'*60}")
                        print("  Failure reasons:")
                        for reason in reasons:
                            print(f"    ✗ {reason}")
                    print(f"{'═'*60}")
            else:
                print("Warning: Required positions not set, skipping detailed analysis")
            
            print("="*60 + "\n")
            
            # Save final observations if not already saved
            if last_observation is None:
                last_observation = my_task.get_observations()
                cv2.imwrite(save_root_depth, last_observation["depth_image"])
                cv2.imwrite(save_root_rgb, last_observation["rgb_image"])
                if "goal_mask" in last_observation:
                    cv2.imwrite(save_root_semantic, last_observation["goal_mask"])

        if actions is not None:
            articulation_controller.apply_action(actions)


simulation_app.close()
