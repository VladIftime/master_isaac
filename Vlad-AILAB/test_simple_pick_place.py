#!/usr/bin/env python3
"""
Simple Pick and Place Test

This script tests the SimplePickPlaceController with a single cube.
It creates one cube, picks it up, and places it to the right.

Command-line arguments:
  --display-images    Enable visualization of camera images
                     (default: disabled)

Example usage:
  python test_simple_pick_place.py
  python test_simple_pick_place.py --display-images
"""

import argparse
from isaacsim import SimulationApp

# Parse command line arguments
parser = argparse.ArgumentParser(description="Simple Pick and Place Test")
parser.add_argument(
    "--display-images",
    action="store_true",
    default=False,
    help="Display intermediate camera images and visualizations",
)
args, unknown = parser.parse_known_args()

DISPLAY_INTERMEDIATE_IMAGES = args.display_images

# Initialize the simulation application
simulation_app = SimulationApp({"headless": False})

# Disable fancy rendering for better performance
import carb
carb.settings.get_settings().set_bool("/rtx/post/ambientOcclusion/enabled", False)
carb.settings.get_settings().set_bool("/rtx/post/bloom/enabled", False)
carb.settings.get_settings().set_bool("/rtx/post/depthOfField/enabled", False)
carb.settings.get_settings().set_bool("/rtx/reflections/enabled", False)
carb.settings.get_settings().set_bool("/rtx/shadows/enabled", False)

import sys
import os
import numpy as np
from omni.kit.viewport.utility import get_active_viewport
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

# Add the path to the custom utility scripts
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils_robots.controllers.simple_pick_place_controller import SimplePickPlaceController
from utils_robots.tasks.pick_place_task import UR5ePickPlace

print("="*70)
print("SIMPLE PICK AND PLACE TEST")
print("="*70)

# Initialize the simulation world
my_world = World(physics_dt=0.01, stage_units_in_meters=1.0)

# Create the task (with no objects, we'll add our own)
my_task = UR5ePickPlace(number_of_objects=0)
my_world.add_task(my_task)
my_world.reset()

# Get the robot
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])

# Log robot configuration
robot_base_pos, robot_base_ori = my_ur5e.get_world_pose()
print(f"\nRobot Configuration:")
print(f"  Base position: [{robot_base_pos[0]:.3f}, {robot_base_pos[1]:.3f}, {robot_base_pos[2]:.3f}]")
print(f"  Base orientation: [{robot_base_ori[0]:.3f}, {robot_base_ori[1]:.3f}, {robot_base_ori[2]:.3f}, {robot_base_ori[3]:.3f}]")

# Get articulation controller
articulation_controller = my_ur5e.get_articulation_controller()

# Create the simple pick and place controller
pick_place_controller = SimplePickPlaceController(
    name="simple_pick_place",
    gripper=my_ur5e.gripper,
    robot_articulation=my_ur5e,
    events_dt=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # 9 phases now
    end_effector_offset=np.array([0, 0, 0.0]),
    lift_height=0.20,  # 20cm safe height above objects
    grasp_height_offset=0.02,  # 2cm above object to grasp
    place_height=0.05,  # 5cm above ground to place
)

# Set viewport
viewport = get_active_viewport()
viewport.set_active_camera("/OmniverseKit_Persp")

# State variables
cube_created = False
pick_place_started = False
target_cube = None

# Define positions based on platform center
platform_center = np.array([0.5, 0, 0.025])  # From pick_place_task.py

# Picking position: center of platform
picking_position = platform_center.copy()
picking_position[2] = 0.075  # 7.5cm above platform (cube will be at this height)

# Placing position: 15cm to the right
placing_position = picking_position.copy()
placing_position[0] += 0.15  # 15cm to the right in +X direction

print(f"\nPlanned Positions:")
print(f"  Pick position:  [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
print(f"  Place position: [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
print("="*70)

print("\nStarting simulation...")
print("  Waiting for scene to settle...")

# Main simulation loop
while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.is_playing():
        actions = None
        
        # Reset on timestep 0
        if my_world.current_time_step_index == 0:
            my_world.reset()
            pick_place_controller.reset()
            cube_created = False
            pick_place_started = False
            target_cube = None
        
        # Create cube after initial settling
        if not cube_created and my_world.current_time_step_index > 100:
            cube_size = 0.05  # 5cm cube
            cube_position = picking_position.copy()
            
            try:
                target_cube = DynamicCuboid(
                    prim_path="/World/TestCube",
                    name="test_cube",
                    position=cube_position,
                    scale=np.array([cube_size, cube_size, cube_size]),
                    color=np.array([0.0, 0.0, 1.0]),  # Blue cube
                    mass=0.05,  # 50g
                )
                my_world.scene.add(target_cube)
                cube_created = True
                
                print(f"\n{'='*70}")
                print(f"CUBE CREATED")
                print(f"{'='*70}")
                print(f"  Position: [{cube_position[0]:.3f}, {cube_position[1]:.3f}, {cube_position[2]:.3f}]")
                print(f"  Size: {cube_size*100:.1f}cm")
                print(f"  Mass: 50g")
                print(f"{'='*70}\n")
            except Exception as e:
                print(f"ERROR: Could not create cube: {e}")
        
        # Start pick and place after cube settles
        if cube_created and not pick_place_started and my_world.current_time_step_index > 200:
            # Get actual cube position
            if target_cube is not None:
                try:
                    cube_obj = my_world.scene.get_object("test_cube")
                    if cube_obj is not None:
                        actual_pos, _ = cube_obj.get_world_pose()
                        picking_position = np.array(actual_pos)
                        
                        # Update placing position based on actual pick position
                        placing_position = picking_position.copy()
                        placing_position[0] += 0.15
                        
                        print(f"\n{'='*70}")
                        print(f"STARTING PICK AND PLACE")
                        print(f"{'='*70}")
                        print(f"  Pick from:  [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                        print(f"  Place at:   [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
                        print(f"  Distance: {np.linalg.norm(placing_position - picking_position)*100:.1f}cm")
                        print(f"{'='*70}\n")
                        
                        pick_place_started = True
                except Exception as e:
                    print(f"ERROR: Could not get cube position: {e}")
        
        # Execute pick and place
        if pick_place_started and not pick_place_controller.is_done():
            current_joint_positions = my_ur5e.get_joint_positions()
            
            actions = pick_place_controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
            )
            
            # Log progress every 5 seconds
            if my_world.current_time_step_index % 500 == 0:
                phase = pick_place_controller.get_current_event()
                print(f"  Step {my_world.current_time_step_index}: Phase {phase}")
        
        # Completion message
        if pick_place_controller.is_done() and pick_place_started:
            print(f"\n{'='*70}")
            print(f"PICK AND PLACE COMPLETED!")
            print(f"{'='*70}")
            
            # Get final cube position
            try:
                cube_obj = my_world.scene.get_object("test_cube")
                if cube_obj is not None:
                    final_pos, _ = cube_obj.get_world_pose()
                    print(f"  Initial position: [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
                    print(f"  Final position:   [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
                    
                    distance_moved = np.linalg.norm(np.array(final_pos[:2]) - picking_position[:2])
                    print(f"  Distance moved: {distance_moved*100:.2f}cm")
                    
                    success = distance_moved > 0.05  # Moved more than 5cm
                    print(f"\n  Result: {'SUCCESS' if success else 'FAILED'}")
            except Exception as e:
                print(f"  Could not verify final position: {e}")
            
            print(f"{'='*70}\n")
            pick_place_started = False  # Prevent repeated messages
        
        # Apply actions
        if actions is not None:
            articulation_controller.apply_action(actions)

simulation_app.close()

