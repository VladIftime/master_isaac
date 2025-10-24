#!/usr/bin/env python3

"""
Demo script showing how to detect a box using camera data and move the gripper above it.

This script demonstrates:
1. Getting camera observations (RGB, depth, heightmap)
2. Detecting object center from camera data
3. Moving the gripper to a position above the detected box
"""

import numpy as np
import cv2
from isaacsim import SimulationApp

def main():
    # Setup simulation - MUST be instantiated first
    simulation_app = SimulationApp({"headless": False})
    
    # Now import Isaac Sim modules after SimulationApp is instantiated
    from isaacsim.core.api import World
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    import isaacsim.core.utils.stage as stage_utils
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.sensors.camera.camera import Camera
    
    from utils_robots.robots.ur5e_handeye import UR5eHandeye
    from utils_robots.tasks.pick_place_task import UR5ePickPlace
    from utils_robots.controllers.basic_manipulation_controller import BasicManipulationController
    
    enable_extension("omni.isaac.ros2_bridge")
    
    # Create world
    my_world = World(stage_units_in_meters=1.0, backend="numpy")
    
    # Setup task (it will load the stage and set up everything)
    print("Setting up task...")
    my_task = UR5ePickPlace(name="pick_place_task")
    my_world.add_task(my_task)
    
    # Reset world
    print("Resetting world...")
    my_world.reset()
    
    # Get references to the robot and cameras from the task
    my_ur5e = my_world.scene.get_object(my_task.robot_name)
    camera_ortho = my_world.scene.get_object(my_task.overhead_camera_name)
    
    print("\n" + "="*60)
    print("Starting box detection and gripper positioning demo")
    print("="*60 + "\n")
    
    # Run simulation
    detection_complete = False
    box_position = None
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
            
            # After 100 steps, perform detection
            if not detection_complete and my_world.current_time_step_index == 100:
                print("\n--- Step 1: Capturing camera observations ---")
                observations = my_task.get_observations()
                
                print(f"RGB image shape: {observations['rgb_image'].shape}")
                print(f"Depth image shape: {observations['depth_image'].shape}")
                print(f"Heightmap points: {len(observations['height_map'])}")
                
                # Save images for inspection
                cv2.imwrite("/tmp/box_detection_rgb.png", observations["rgb_image"])
                cv2.imwrite("/tmp/box_detection_depth.png", observations["depth_image"])
                print("Saved images to /tmp/box_detection_*.png")
                
                print("\n--- Step 2: Detecting box from camera data ---")
                # Detect box using only camera data
                box_position = my_task.detect_object_center_from_camera(
                    camera=camera_ortho,
                    object_name=None,  # Use depth-based detection
                    rgb_image=observations["rgb_image"],
                    depth_image=observations["depth_image"],
                    heightmap=observations["height_map"],
                    display=False,  # Set to True to see visualization
                )
                
                if box_position is not None:
                    print(f"\nBox detected!")
                    print(f"  Bottom center position: [{box_position[0]:.3f}, {box_position[1]:.3f}, {box_position[2]:.3f}]")
                    
                    print("\n--- Demo complete! ---")
                    print("You can now use this position to move the gripper")
                    print("="*60)
                else:
                    print("No box detected!")
                
                detection_complete = True
    
    simulation_app.close()


if __name__ == "__main__":
    main()

