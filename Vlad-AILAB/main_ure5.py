from unittest import skip
from isaacsim import SimulationApp
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

# Add the path to the Omniverse Isaac Sim examples
sys.path.append(
    #
    "/home/vladi/.local/share/ov/pkg/isaac-sim-4.5.0/exts/isaacsim.core.api.examples"
)
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
number_of_objects = 5

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
viewport.set_active_camera("/OmniverseKit_Persp")

# Declare found_obj to check if the target object is found (initially not found, so False)
found_obj = False


print(
    "---------------------------------Start simulation---------------------------------"
)
first_observation = None
last_observation = None

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

        actions = push_controller.forward(
            push_start_position=np.array([0.5, 0, 0.14]),
            push_end_position=np.array([0.55, 0.1, 0.14]),
        )
        if first_observation is None and my_world.current_time_step_index > 10:
            first_observation = my_task.get_observations()
        if push_controller.is_done():
            print("Push done")
            last_observation = my_task.get_observations()
            
        if actions is not None:
            articulation_controller.apply_action(actions)

    if last_observation is not None:
        display_heightmap(last_observation["height_map"], "last_height_map")
    if first_observation is not None:
        display_heightmap(first_observation["height_map"], "first_height_map")
simulation_app.close()
