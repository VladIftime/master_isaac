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
    "/home/vladi/.local/share/ov/pkg/isaac-sim-4.2.0/exts/omni.isaac.examples"
)

import omni
from omni.isaac.nucleus import get_assets_root_path, is_file
from omni.isaac.examples.ailab_script import AILabExtension
from omni.isaac.examples.ailab_examples import AILab
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.kit.viewport.utility import get_active_viewport
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.viewports import backproject_depth
from omni.isaac.core.objects import DynamicCuboid

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path

# Add the path to the custom utility scripts
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils_robots.controllers.pick_place_controller_robotiq import PickPlaceController
from utils_robots.controllers.pick_place_controller_ext import CustomPickPlaceController

from utils_robots.controllers.basic_manipulation_controller import (
    BasicManipulationController,
)
from utils_robots.controllers.RMPFflow_pickplace import RMPFlowController
from utils_robots.tasks.pick_place_task import UR5ePickPlace
from dev_utils.pc_to_png import save_point_cloud_as_png
from dev_utils.pc_to_png import depth_image_from_distance_image

# Define the path to the objects directory
PATH_to_objects = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "objects"
)
print(PATH_to_objects)


# Define a custom extension for the AILab
class AILabExtensions(AILabExtension):
    def __init__(self):
        super().__init__()

    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        # Start the AILab extension with the specified parameters
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="AILab extension",
            title="AILab extension Example",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_core_hello_world.html",
            overview="This Example introduces the user on how to work with Isaac Sim through scripting in asynchronous mode.",
            file_path=os.path.abspath(__file__),
            sample=AILab(),
        )
        return


# Initialize and start the custom AILab extension
gui_test = AILabExtensions()
gui_test.on_startup(ext_id="omni.isaac.examples-1.5.1")

# Try to load the stage
stage_usd_path = os.path.realpath("ure5_stage_basket.usd")

# Initialize the simulation world
my_world = World(physics_dt=0.01, stage_units_in_meters=1.0)

my_task = UR5ePickPlace()
# Add the task to the simulation world and reset the world
my_world.add_task(my_task)
my_world.reset()

# Obtain ur5e and camera from the task
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
camera_ortho = my_task.get_camera_ortho()
camera_perspective = my_task.get_camera_persp()

# Create PickPlace controller
# pick_and_place_controller = PickPlaceController(
#     name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e
# )

pick_and_place_controller = CustomPickPlaceController(
    name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e
)

# Create EndEffector Controller
basic_controller = BasicManipulationController(
    name="basic_manipulation_controller",
    cspace_controller=RMPFlowController(
        name="basic_manipulation_controller_cspace_controller",
        robot_articulation=my_ur5e,
        attach_gripper=True,
    ),
    gripper=my_ur5e.gripper,
    events_dt=[0.008],
)


# Declare instance for robot control (PD control)
articulation_controller = my_ur5e.get_articulation_controller()

# Specify the view point in the GUI (when converting from Depth camera view to Perspective view, it is generally easier to see)
viewport = get_active_viewport()
viewport.set_active_camera("/World/OverheadCamera")
viewport.set_active_camera("/World/PerspCamera")
viewport.set_active_camera('/OmniverseKit_Persp')

# Declare found_obj to check if the target object is found (initially not found, so False)
found_obj = False


print(
    "---------------------------------Start simulation---------------------------------"
)

cube_prim_path = "/World/Cube"
cube_name = "cube"
pos_x = 0.5
pos_y = 0
pos_z = 0.1
objects_position = np.array([[pos_x, pos_y, pos_z]])
objects_position[0][2] = objects_position[0][2] + 0.01

_object = DynamicCuboid(
    prim_path=cube_prim_path,
    name=cube_name,
    position=objects_position[0],
    color=np.array([0, 0, 1]),
    size=0.04,
    mass=0.01,
)

my_world.scene.add(_object)



change_world_center = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            pick_and_place_controller.reset()

        rgb_image = camera_perspective.get_rgba()[:, :, :3]
        distance_image = camera_perspective.get_current_frame()["distance_to_camera"]
                    
        # Convert distance image to depth image using camera intrinsics
        if my_world.current_time_step_index % 10000 == 0:
            camera_intrinsics = camera_perspective.get_intrinsics_matrix()
            depth_image_pc = []
            if distance_image is not None:
                # depth_image_pc = backproject_depth(
                #     depth_image=distance_image,
                #     viewport_api=viewport,
                #     max_clip_depth=np.max(distance_image),
                # )
                depth_image_pc = depth_image_from_distance_image(
                    distance=distance_image, intrinsics=camera_intrinsics
                )
            new_depth_image = np.uint16(depth_image_pc)
            plt.imshow(depth_image_pc)
            plt.show()
            carb.log_warn(f"Current event: {distance_image}")
            # Save the point cloud to a file

            save_point_cloud_as_png(
                depth_image_pc, "camera_image/point_cloud_xy.png", projection="xy"
            )

        if my_world.current_time_step_index % 10 == 0:
            # Save the RGB image to a file
            save_root = os.path.join(os.getcwd(), "camera_image/rgb.png")
            image = Image.fromarray(rgb_image)
            image.save(save_root)

        observations = my_world.get_observations()
        actions = pick_and_place_controller.forward(
            picking_position=np.array([0.5, 0, 0.14]),
            placing_position=np.array([0.5, 0.7, 0.14]),
            current_joint_positions=my_ur5e.get_joint_positions(),
            end_effector_offset=np.array([0, 0, 0.14]),
            end_effector_orientation=euler_angles_to_quat(
                np.array([0, np.pi, -np.pi / 2])
            ),
        )
        if pick_and_place_controller.is_done():
            print(pick_and_place_controller.get_grasp())

        articulation_controller.apply_action(actions)

simulation_app.close()
