#!/usr/bin/env python3
"""
UR5e Box Pick and Place Simulation

This script demonstrates a UR5e robot performing pick and place operations with a box.
It creates a cube inside the box, picks it up, and places it next to the box.
Height maps are saved at three stages: initial state, after picking, and after placing.

Command-line arguments:
  --display-images    Enable visualization of intermediate camera images and detections
                     (default: disabled for better performance)

Example usage:
  python main_box_pick_place.py                    # Run without visualizations
  python main_box_pick_place.py --display-images   # Run with visualizations enabled
"""

import argparse
from isaacsim import SimulationApp

# Parse command line arguments
parser = argparse.ArgumentParser(description="UR5e Box Pick and Place Simulation")
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
from utils_robots.controllers.simple_pick_place_controller import SimplePickPlaceController
from utils_robots.tasks.pick_place_task import UR5ePickPlace


# Create directory for saving height maps
heightmap_dir = os.path.join(os.getcwd(), "heightmaps")
os.makedirs(heightmap_dir, exist_ok=True)

# Save paths for height maps
save_heightmap_initial = os.path.join(heightmap_dir, "heightmap_initial.npy")
save_heightmap_after_pick = os.path.join(heightmap_dir, "heightmap_after_pick.npy")
save_heightmap_after_place = os.path.join(heightmap_dir, "heightmap_after_place.npy")

# Save paths for images
save_root_depth = os.path.join(os.getcwd(), "camera_image/depth.png")
save_root_rgb = os.path.join(os.getcwd(), "camera_image/rgb.png")
save_root_semantic = os.path.join(os.getcwd(), "camera_image/semantic.png")


# Define the path to the objects directory
PATH_to_objects = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "objects"
)
print(PATH_to_objects)

# Define the number of objects (0, we'll create our own cube in a box)
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

# Declare instance for robot control (PD control)
articulation_controller = my_ur5e.get_articulation_controller()

# Create PickPlace controller using SimplePickPlaceController (works with ParallelGripper)
pick_and_place_controller = SimplePickPlaceController(
    name="pick_place_controller",
    gripper=my_ur5e.gripper,
    robot_articulation=my_ur5e,
    events_dt=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # 9 phases now
    end_effector_offset=np.array([0, 0, 0.0]),
    lift_height=0.20,  # 20cm safe height above box
    grasp_height_offset=0.02,  # 2cm above object to grasp
    place_height=0.05,  # 5cm above ground to place
)

# Specify the view point in the GUI
viewport = get_active_viewport()
viewport.set_active_camera("/OmniverseKit_Persp")

# State variables
box_created = False
cube_created = False
box_position = None
cube_position = None
pick_place_started = False
heightmap_initial_saved = False
heightmap_after_pick_saved = False
heightmap_after_place_saved = False

# Observation storage
observation_initial = None
observation_after_pick = None
observation_after_place = None

# Pick and place planning variables
picking_position = None
placing_position = None
box_cube = None
target_cube = None

# Tracking variables for after pick detection
pick_phase_started = False
place_phase_completed = False

print("-" * 60)
print("Starting Box Pick and Place Simulation")
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
            
            pick_place_started = False
            box_created = False
            cube_created = False
            heightmap_initial_saved = False
            heightmap_after_pick_saved = False
            heightmap_after_place_saved = False
            box_position = None
            cube_position = None
            picking_position = None
            placing_position = None
            pick_phase_started = False
            place_phase_completed = False

        # Create box and cube after initial settling
        if not box_created and my_world.current_time_step_index > 100:
            # Platform center is at [0.5, 0, 0.025] based on pick_place_task.py
            platform_center = np.array([0.5, 0, 0.025])
            
            # Create a box (hollow container)
            box_outer_size = 0.12  # 12cm outer dimension
            box_wall_thickness = 0.01  # 1cm wall thickness
            box_inner_size = box_outer_size - 2 * box_wall_thickness
            box_height = 0.08  # 8cm tall box
            
            # Position box on platform
            box_position = platform_center.copy()
            box_position[2] = platform_center[2] + box_height / 2
            
            try:
                # Create box as a visual container (using DynamicCuboid for simplicity)
                # In a more complex scenario, you'd create a proper hollow box with walls
                box_cube = DynamicCuboid(
                    prim_path="/World/BoxContainer",
                    name="box_container",
                    position=box_position,
                    scale=np.array([box_outer_size, box_outer_size, box_height]),
                    color=np.array([0.5, 0.5, 0.5]),  # Gray box
                    mass=0.1,
                )
                my_world.scene.add(box_cube)
                box_created = True
                print(f"\n{'='*60}")
                print(f"BOX CREATED")
                print(f"{'='*60}")
                print(f"Position: [{box_position[0]:.3f}, {box_position[1]:.3f}, {box_position[2]:.3f}]")
                print(f"Outer size: {box_outer_size*100:.1f}cm x {box_outer_size*100:.1f}cm x {box_height*100:.1f}cm")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Could not spawn box: {e}")

        # Create cube inside the box after box settles
        if box_created and not cube_created and my_world.current_time_step_index > 150:
            # Create a smaller cube to fit inside the box
            cube_size = 0.04  # 4cm cube
            
            # Position cube inside/on top of the box
            cube_position = box_position.copy()
            cube_position[2] = box_position[2] + 0.08 / 2 + cube_size / 2 + 0.005  # On top of box
            
            try:
                target_cube = DynamicCuboid(
                    prim_path="/World/TargetCube",
                    name="target_cube",
                    position=cube_position,
                    scale=np.array([cube_size, cube_size, cube_size]),
                    color=np.array([1.0, 0.0, 0.0]),  # Red cube
                    mass=0.01,
                )
                my_world.scene.add(target_cube)
                cube_created = True
                print(f"\n{'='*60}")
                print(f"CUBE CREATED IN BOX")
                print(f"{'='*60}")
                print(f"Position: [{cube_position[0]:.3f}, {cube_position[1]:.3f}, {cube_position[2]:.3f}]")
                print(f"Size: {cube_size*100:.1f}cm")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Could not spawn cube: {e}")

        # Wait for objects to settle, then capture initial observation
        if cube_created and not heightmap_initial_saved and my_world.current_time_step_index > 250:
            observation_initial = my_task.get_observations()
            heightmap_initial = observation_initial["height_map"]
            
            # Save height map
            np.save(save_heightmap_initial, heightmap_initial)
            
            # Save RGB and depth images
            cv2.imwrite(save_root_rgb.replace(".png", "_initial.png"), observation_initial["rgb_image"])
            cv2.imwrite(save_root_depth.replace(".png", "_initial.png"), observation_initial["depth_image"])
            
            print(f"\n{'='*60}")
            print(f"HEIGHT MAP SAVED: INITIAL STATE")
            print(f"{'='*60}")
            print(f"Saved to: {save_heightmap_initial}")
            print(f"Height map shape: {heightmap_initial.shape}")
            print(f"Number of points: {len(heightmap_initial)}")
            print(f"RGB image saved: {save_root_rgb.replace('.png', '_initial.png')}")
            print(f"Depth image saved: {save_root_depth.replace('.png', '_initial.png')}")
            print(f"{'='*60}\n")
            
            heightmap_initial_saved = True
            
            if DISPLAY_INTERMEDIATE_IMAGES:
                display_heightmap(observation_initial["height_map"], name="Height Map - Initial State")
            
            # Get actual cube position for picking
            try:
                target_cube_obj = my_world.scene.get_object("target_cube")
                if target_cube_obj is not None:
                    cube_world_pos, cube_world_ori = target_cube_obj.get_world_pose()
                    picking_position = np.array(cube_world_pos)
                    
                    # Calculate placing position: next to the box (to the right)
                    placing_position = picking_position.copy()
                    placing_position[0] += 0.20  # 20cm to the right
                    
                    print(f"\n{'='*60}")
                    print(f"PICK AND PLACE PLANNING")
                    print(f"{'='*60}")
                    print(f"Cube position:     [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                    print(f"Placing position:  [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
                    print(f"Distance: {np.linalg.norm(placing_position - picking_position)*100:.1f}cm")
                    print(f"{'='*60}\n")
                    
                    pick_place_started = True
                else:
                    print("WARNING: Could not find target cube in scene")
            except Exception as e:
                print(f"ERROR: Could not get cube position: {e}")

        # Execute pick and place manipulation
        if pick_place_started and not pick_and_place_controller.is_done() and picking_position is not None and placing_position is not None:
            if not pick_phase_started:
                pick_phase_started = True
                print(f"\n{'='*60}")
                print(f"STARTING PICK AND PLACE SEQUENCE")
                print(f"{'='*60}\n")
            
            current_joint_positions = my_ur5e.get_joint_positions()
            actions = pick_and_place_controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
            )

        # After pick and place completes, save height maps
        if pick_and_place_controller.is_done() and pick_place_started and not place_phase_completed:
            # Wait a few steps for physics to settle before capturing
            if my_world.current_time_step_index > 0:
                observation_after_place = my_task.get_observations()
                heightmap_after_place = observation_after_place["height_map"]
                
                # Save height map after placing
                np.save(save_heightmap_after_place, heightmap_after_place)
                
                # Save RGB and depth images
                cv2.imwrite(save_root_rgb.replace(".png", "_after_place.png"), observation_after_place["rgb_image"])
                cv2.imwrite(save_root_depth.replace(".png", "_after_place.png"), observation_after_place["depth_image"])
                
                print(f"\n{'='*60}")
                print(f"PICK AND PLACE SEQUENCE COMPLETED")
                print(f"{'='*60}")
                print(f"HEIGHT MAP SAVED: AFTER PLACE")
                print(f"Saved to: {save_heightmap_after_place}")
                print(f"Height map shape: {heightmap_after_place.shape}")
                print(f"Number of points: {len(heightmap_after_place)}")
                print(f"RGB image saved: {save_root_rgb.replace('.png', '_after_place.png')}")
                print(f"Depth image saved: {save_root_depth.replace('.png', '_after_place.png')}")
                print(f"{'='*60}\n")
                
                if DISPLAY_INTERMEDIATE_IMAGES:
                    display_heightmap(observation_after_place["height_map"], name="Height Map - After Place")
                
                place_phase_completed = True
                
                print(f"\n{'='*60}")
                print(f"BOX PICK AND PLACE SIMULATION COMPLETED")
                print(f"{'='*60}")
                print(f"Height maps saved:")
                print(f"  1. Initial state:  {save_heightmap_initial}")
                print(f"  2. After place:    {save_heightmap_after_place}")
                print(f"{'='*60}\n")

        if actions is not None:
            articulation_controller.apply_action(actions)


simulation_app.close()

