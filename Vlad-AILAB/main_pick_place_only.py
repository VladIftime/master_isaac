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
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Run simulation in headless mode",
)
args, _ = parser.parse_known_args()

# Flag to control display of intermediate pictures
DISPLAY_INTERMEDIATE_IMAGES = False

# Initialize the simulation application with a GUI
simulation_app = SimulationApp({"headless": args.headless})

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
# from omni.kit.viewport.utility import get_active_viewport
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api import World
from isaacsim.core.utils.rotations import euler_angles_to_quat
from utils_robots.tasks.pick_place_task import UR5ePickPlace
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from scipy.spatial.transform import Rotation as R


# Add the path to the custom utility scripts
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Alternative: Use discrete controller (no interpolation, simpler like push controller)
from utils_robots.controllers.pick_place_controller_rmpflow import RMPFlowPickPlaceController

# Save paths for images
save_root_depth = os.path.join(os.getcwd(), "camera_image/depth.png")
save_root_rgb = os.path.join(os.getcwd(), "camera_image/rgb.png")
save_root_semantic = os.path.join(os.getcwd(), "camera_image/semantic.png")

# Define the number of objects (3 for multi-object pick and place)
number_of_objects = 1

# Initialize the simulation world
my_world = World(physics_dt=0.01, stage_units_in_meters=1.0)

my_task = UR5ePickPlace(number_of_objects=number_of_objects, randomize_position=False)
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

# Declare instance for robot control (PD control)
articulation_controller = my_ur5e.get_articulation_controller()

ee_offset = np.array([0, 0, 0.22])

# Use RMPFlow Controller with Interpolation
pick_and_place_controller = RMPFlowPickPlaceController(
    name="pick_place_controller",
    gripper=my_ur5e.gripper,
    robot_articulation=my_ur5e,
    # Adjust timings as needed for RMPFlow
    # Adjust timings as needed for RMPFlow
    # 12 phases: [turned, align_pick, rotate_yaw, lower_pick, settle, close, lift, check_grasp, turned_intermediate, lower_drop, open, overhead]
    events_dt=[3.0, 3.0, 2.0, 3.0, 0.5, 1.0, 2.0, 1.0, 3.0, 3.0, 2.0, 3.0],
    # Add Z offset for gripper length + hand-eye camera mount
    # Based on push task (Flange Z ~0.32 for Object Z ~0.05), offset should be ~0.27m
    end_effector_offset=ee_offset, # Manually added to target
    end_effector_initial_height=0.55,  # 55cm above workspace (safe height for flange)
)

# Specify the view point in the GUI
if not args.headless:
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    viewport.set_active_camera("/OmniverseKit_Persp")

# State variables
current_object_idx = 0
pick_place_started = False
observation_captured = False
object_position_found = False

# Pick and place planning variables
picking_position = None
placing_position = None
target_orientation = None # Target orientation for the gripper
target_object = None
target_object = None

print("-" * 60)
print("Starting Multi-Object Pick and Place Simulation")
print(f"Display images: {'ENABLED' if DISPLAY_INTERMEDIATE_IMAGES else 'DISABLED'}")
print("-" * 60)

my_world.reset()
try:
    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_playing():
            if my_world.current_time_step_index % 100 == 0:
                ee_pos, _ = my_ur5e.end_effector.get_world_pose()
                print(f"Step: {my_world.current_time_step_index}, EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            actions = None

            if my_world.current_time_step_index == 0:
                my_world.reset()
                pick_and_place_controller.reset()
                current_object_idx = 0
                pick_place_started = False
                observation_captured = False
                object_position_found = False
                picking_position = None
                placing_position = None
                target_orientation = None
                target_object = None

            # Check if all objects are processed
            if current_object_idx >= number_of_objects:
                # Optional: print completion message once
                pass

            # Wait for scene to settle, then find object and plan pick/place
            elif not object_position_found and my_world.current_time_step_index > 200:
                # Get the current object from the scene
                try:
                    # Try to get the object from the scene using the prim path
                    # Note: The task creates objects with suffix _0, _1, _2...
                    object_prim_path = f"{my_task.imported_objects_prim_path}_{current_object_idx}"
                    
                    if is_prim_path_valid(object_prim_path):
                        target_object = XFormPrim(object_prim_path)
                        target_object.initialize()
                        
                        # Get object's actual position in world coordinates
                        object_world_poses, object_world_oris = target_object.get_world_poses()
                        object_world_pos = object_world_poses[0]
                        object_world_ori = object_world_oris[0]
                        
                        picking_position = np.array(object_world_pos)
                        
                        # Calculate target orientation (Down + Object Yaw)
                        # Default Down: [pi, 0, pi] (from reference)
                        q_default_isaac = euler_angles_to_quat(np.array([np.pi, 0, np.pi]))
                        # Convert to Scipy [x, y, z, w]
                        r_default = R.from_quat([q_default_isaac[1], q_default_isaac[2], q_default_isaac[3], q_default_isaac[0]])
                        
                        # Object orientation
                        r_obj = R.from_quat([object_world_ori[1], object_world_ori[2], object_world_ori[3], object_world_ori[0]])
                        yaw = r_obj.as_euler('xyz')[2]
                        
                        # Rotate default by Yaw around Z
                        r_z = R.from_euler('z', yaw)
                        r_target = r_z * r_default
                        
                        # Convert back to Isaac [w, x, y, z]
                        q_target_scipy = r_target.as_quat()
                        target_orientation = np.array([q_target_scipy[3], q_target_scipy[0], q_target_scipy[1], q_target_scipy[2]])
                        
                        print(f"\n{'='*60}")
                        print(f"OBJECT {current_object_idx} FOUND")
                        print(f"{'='*60}")
                        print(f"Object prim path: {object_prim_path}")
                        print(f"Picking position: {picking_position}")
                        print(f"Object Yaw: {yaw:.3f} rad")
                        print(f"Target Orientation: {target_orientation}")
                        print(f"{'='*60}\n")
                        
                        object_position_found = True
                    else:
                        print(f"Object prim path {object_prim_path} is invalid, skipping...")
                        current_object_idx += 1 # Skip this object
                except Exception as e:
                    print(f"Error finding object {current_object_idx}: {e}")
                    current_object_idx += 1

            # Calculate placing position and capture observation
            if object_position_found and not observation_captured and my_world.current_time_step_index > 250:
                observation = my_task.get_observations()
                
                # Save images (overwrite for each object or could append index)
                cv2.imwrite(save_root_depth, observation["depth_image"])
                cv2.imwrite(save_root_rgb, observation["rgb_image"])
                
                # Calculate placing position: Stack them or place in a row
                # Place in a row along Y axis, starting from a base position
                # Base place position: [0.6, -0.2, 0.05]
                base_place_pos = np.array([0.6, -0.2, 0.05])
                
                if picking_position is not None:
                    placing_position = base_place_pos.copy()
                    # Offset each object by 15cm in Y direction
                    placing_position[1] += current_object_idx * 0.15
                    
                    # Ensure Z is correct (same as picking or slightly higher if stacking)
                    placing_position[2] = picking_position[2] 
                else:
                    print("ERROR: Picking position not found")
                    placing_position = None
                
                print(f"\n{'='*60}")
                print(f"PLANNING FOR OBJECT {current_object_idx}")
                print(f"{'='*60}")
                if picking_position is not None and placing_position is not None:
                    print(f"Picking position:  [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                    print(f"Placing position:  [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
                print(f"{'='*60}\n")
                
                observation_captured = True
                pick_place_started = True

            # Execute pick and place manipulation
            if pick_place_started and not pick_and_place_controller.is_done() and picking_position is not None and placing_position is not None:
                current_joint_positions = my_ur5e.get_joint_positions()
                actions = pick_and_place_controller.forward(
                    picking_position=picking_position,
                    current_joint_positions=current_joint_positions,
                    end_effector_orientation=target_orientation,
                )

            # After pick and place completes for current object
            if pick_and_place_controller.is_done() and pick_place_started:
                print(f"\n{'='*60}")
                print(f"OBJECT {current_object_idx} COMPLETED")
                print(f"{'='*60}\n")
                
                # Prepare for next object
                current_object_idx += 1
                pick_and_place_controller.reset()
                pick_place_started = False
                observation_captured = False
                object_position_found = False
                picking_position = None
                placing_position = None
                target_orientation = None
                target_object = None
                
                if current_object_idx >= number_of_objects:
                    print(f"\n{'='*60}")
                    print(f"ALL TASKS COMPLETED")
                    print(f"{'='*60}\n")

            if actions is not None:
                articulation_controller.apply_action(actions)

except Exception as e:
    print(f"Simulation loop error: {e}")
    import traceback
    traceback.print_exc()


simulation_app.close()
