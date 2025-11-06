#!/usr/bin/env python3
"""
UR5e Pick and Place Simulation

This script demonstrates a UR5e robot performing pick and place manipulation.
It spawns a single object, picks it up using its actual coordinates, and places it
at a target position to the right of the object.

Command-line arguments:
  --display-images    Enable visualization of intermediate camera images and detections
                     (default: disabled for better performance)

Example usage:
  python main_pick_place_only.py                    # Run without visualizations
  python main_pick_place_only.py --display-images   # Run with visualizations enabled
"""

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
args, _ = parser.parse_known_args()

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
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.kit.viewport.utility import get_active_viewport
from dev_utils.point_cloud_utils import display_heightmap
from isaacsim.core.api.objects import DynamicCuboid

import isaacsim
from isaacsim.core.api import World
from utils_robots.controllers.pick_place_controller_ext import CustomPickPlaceController
from utils_robots.tasks.pick_place_task import UR5ePickPlace

# Add the path to the custom utility scripts
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Save paths for images
save_root_depth = os.path.join(os.getcwd(), "camera_image/depth.png")
save_root_rgb = os.path.join(os.getcwd(), "camera_image/rgb.png")
save_root_semantic = os.path.join(os.getcwd(), "camera_image/semantic.png")

# Define the number of objects (1 for single object pick and place)
number_of_objects = 1

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

# Log robot base position for debugging coordinate issues
robot_base_pos, robot_base_ori = my_ur5e.get_world_pose()
print(f"\n{'='*70}")
print(f"ROBOT CONFIGURATION")
print(f"{'='*70}")
print(f"Robot base position (world frame): [{robot_base_pos[0]:.3f}, {robot_base_pos[1]:.3f}, {robot_base_pos[2]:.3f}]")
print(f"Robot base orientation (world):    [{robot_base_ori[0]:.3f}, {robot_base_ori[1]:.3f}, {robot_base_ori[2]:.3f}, {robot_base_ori[3]:.3f}]")
print(f"{'='*70}\n")

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

# Declare instance for robot control (PD control)
articulation_controller = my_ur5e.get_articulation_controller()

# Create PickPlace controller with end_effector_offset in constructor (like push controller)
pick_and_place_controller = CustomPickPlaceController(
    name="pick_place_controller",
    gripper=my_ur5e.gripper,
    robot_articulation=my_ur5e,
    end_effector_offset=np.array([0, 0, 0.02]),
)

# Specify the view point in the GUI
viewport = get_active_viewport()
viewport.set_active_camera("/OmniverseKit_Persp")

# State variables
pick_place_started = False
observation_captured = False
object_position_found = False

# Pick and place planning variables
picking_position = None
placing_position = None
target_object = None

print("-" * 60)
print("Starting Pick and Place Simulation")
print(f"Display images: {'ENABLED' if DISPLAY_INTERMEDIATE_IMAGES else 'DISABLED'}")
print("-" * 60)

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        actions = None

        if my_world.current_time_step_index == 0:
            my_world.reset()
            pick_and_place_controller.reset()
            pick_place_started = False
            observation_captured = False
            object_position_found = False
            picking_position = None
            placing_position = None
            target_object = None

        # Wait for scene to settle, then find object and plan pick/place
        if not object_position_found and my_world.current_time_step_index > 200:
            # Get the first object from the scene (since number_of_objects=1)
            # Objects are stored in the task with prim paths like "/World/object_0"
            try:
                # Try to get the object from the scene using the prim path
                object_prim_path = f"{my_task.imported_objects_prim_path}_0"
                target_object = my_world.scene.get_object(object_prim_path)
                
                if target_object is not None:
                    # Get object's actual position in world coordinates
                    object_world_pos, object_world_ori = target_object.get_world_pose()
                    
                    # Convert from world frame to robot local frame (required by controller)
                    robot_base_pos_current, robot_base_ori_current = my_ur5e.get_world_pose()
                    picking_position = world_to_robot_frame(
                        np.array(object_world_pos),
                        robot_base_pos_current,
                        robot_base_ori_current
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"OBJECT FOUND")
                    print(f"{'='*60}")
                    print(f"Object prim path: {object_prim_path}")
                    print(f"Object world position: [{object_world_pos[0]:.3f}, {object_world_pos[1]:.3f}, {object_world_pos[2]:.3f}]")
                    print(f"Picking position (robot frame): [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                    print(f"{'='*60}\n")
                    
                    object_position_found = True
                else:
                    raise ValueError("Object not found in scene")
            except Exception as e:
                print(f"Could not find object directly ({e}), using camera estimation")
                # Fallback: use observation to find object position
                observation = my_task.get_observations()
                rgb_image = observation["rgb_image"]
                center_u = rgb_image.shape[1] // 2
                center_v = rgb_image.shape[0] // 2
                
                picking_position = my_task.pixel_to_robot_space(
                    u=center_u,
                    v=center_v,
                    camera=camera_ortho,
                    display=DISPLAY_INTERMEDIATE_IMAGES,
                    rgb_image=observation["rgb_image"],
                    depth_image=observation["depth_image"],
                    heightmap=observation["height_map"],
                )
                object_position_found = True
                print(f"\n{'='*60}")
                print(f"OBJECT POSITION ESTIMATED FROM CAMERA")
                print(f"{'='*60}")
                print(f"Estimated position: [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                print(f"{'='*60}\n")

        # Calculate placing position (to the right of the object) and capture observation
        if object_position_found and not observation_captured and my_world.current_time_step_index > 250:
            observation = my_task.get_observations()
            
            # Save images
            cv2.imwrite(save_root_depth, observation["depth_image"])
            cv2.imwrite(save_root_rgb, observation["rgb_image"])
            if "goal_mask" in observation:
                cv2.imwrite(save_root_semantic, observation["goal_mask"])
            
            # Calculate placing position: to the right of the object (+X direction)
            if picking_position is not None:
                placing_position = np.array(picking_position)
                placing_position[0] += 0.15  # Move 15cm to the right (+X direction)
                placing_position[1] = picking_position[1]  # Keep same Y
                placing_position[2] = picking_position[2]  # Keep same height
            else:
                print("ERROR: Picking position not found, cannot calculate placing position")
                placing_position = None
            
            if DISPLAY_INTERMEDIATE_IMAGES:
                display_heightmap(observation["height_map"], name="Height Map - Pick and Place")
            
            print(f"\n{'='*60}")
            print(f"OBSERVATION CAPTURED AND PLACING POSITION CALCULATED")
            print(f"{'='*60}")
            print(f"RGB image saved: {save_root_rgb}")
            print(f"Depth image saved: {save_root_depth}")
            if "goal_mask" in observation:
                print(f"Semantic image saved: {save_root_semantic}")
            print(f"\nPICK AND PLACE PLANNING")
            if picking_position is not None and placing_position is not None:
                print(f"Picking position:  [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                print(f"Placing position:  [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
                print(f"Distance: {np.linalg.norm(placing_position - picking_position)*100:.1f}cm")
            else:
                print("ERROR: Positions not properly initialized")
            print(f"{'='*60}\n")
            
            observation_captured = True
            pick_place_started = True

        # Execute pick and place manipulation
        if pick_place_started and not pick_and_place_controller.is_done() and picking_position is not None and placing_position is not None:
            current_joint_positions = my_ur5e.get_joint_positions()
            actions = pick_and_place_controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
            )

        # After pick and place completes
        if pick_and_place_controller.is_done() and pick_place_started:
            print(f"\n{'='*60}")
            print(f"PICK AND PLACE SEQUENCE COMPLETED")
            print(f"{'='*60}\n")
            pick_place_started = False  # Prevent repeated messages

        if actions is not None:
            articulation_controller.apply_action(actions)


simulation_app.close()
