# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import carb
import omni

from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import create_prim, get_prim_path, define_prim
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom
from omni.isaac.nucleus import get_assets_root_path, is_file
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from utils_robots.robots.ur5e_handeye import UR5eHandeye
import os, random
import numpy as np
from typing import Optional
from PIL import Image


class UR5ePickPlace(tasks.PickPlace):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "ur5_pick_place".
    """

    def __init__(
        self,
        name: str = "ur5e_pick_place",
        stage_usd_path: Optional[str] = None,
        robot_name: Optional[str] = "ur5e",
        overhead_camera_name: Optional[str] = "overhead_camera",
        pespective_camera_name: Optional[str] = "perspective_camera",
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
        )
        self.stage_usd_path = stage_usd_path
        self.load_stage = True
        self.imported_objects_prim_path = "/World/object"
        self.objects_position_list = []
        self.objects_orientation_list = []
        self.objects_name_list = []
        self.robot_name = robot_name
        self.overhead_camera_name = overhead_camera_name
        self.perspective_camera_name = pespective_camera_name
        return

    def set_up_scene(self, scene: Scene) -> None:
        # Try first to import the scene from the usd file
        self._scene = scene
        # stage_utils.open_stage(os.path.realpath("ure5_stage_basket.usd"))

        # -----------------------------------------------------------
        # Set up the robot
        # -----------------------------------------------------------
        scene.add_default_ground_plane()
        self._robot = self.set_robot()
        scene.add(self._robot)
        # -----------------------------------------------------------
        # Platform with welded edges using FixedCuboid and physics material
        # -----------------------------------------------------------
        platform_center = np.array([0.5, 0, 0.025])
        base_size = 0.5
        edge_height = 0.15

        # 1. Base plate with high-friction material
        self.platform_base = FixedCuboid(  # Changed to FixedCuboid
            prim_path="/World/PlatformBase",
            name="platform_base",
            position=platform_center,
            scale=np.array([base_size, base_size, 0.05]),
            color=np.array([0.3, 0.3, 0.3]),
        )
        self.platform_base.set_collision_approximation("convexHull")  # Better collision
        scene.add(self.platform_base)

        # 2. Create physics material for edges
        physics_material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/PlatformMaterial",
            static_friction=2.0,
            dynamic_friction=2.0,
        )
        self.platform_base.apply_physics_material(physics_material)

        # 3. Welded edges using FixedCuboid
        edge_thickness = 0.02
        edge_vertical_offset = 0.025 + (edge_height / 2)  # Proper Z positioning

        # Front/back edges
        for direction, x_sign in [("Front", 1), ("Back", -1)]:
            edge_position = platform_center + np.array(
                [
                    x_sign * (base_size / 2 - edge_thickness / 2),
                    0,
                    edge_vertical_offset,
                ]
            )
            edge = FixedCuboid(  # Changed to FixedCuboid
                prim_path=f"/World/PlatformEdge_{direction}",
                name=f"platform_edge_{direction.lower()}",  # Unique name
                position=edge_position,
                scale=np.array(
                    [edge_thickness, base_size + edge_thickness, edge_height]
                ),
                color=np.array([0.5, 0.5, 0.5]),
            )
            edge.set_collision_approximation("convexHull")
            edge.apply_physics_material(physics_material)
            scene.add(edge)

        # Left/right edges
        for direction, y_sign in [("Left", 1), ("Right", -1)]:
            edge_position = platform_center + np.array(
                [
                    0,
                    y_sign * (base_size / 2 - edge_thickness / 2),
                    edge_vertical_offset,
                ]
            )
            edge = FixedCuboid(  # Changed to FixedCuboid
                prim_path=f"/World/PlatformEdge_{direction}",
                name=f"platform_edge_{direction.lower()}",  # Unique name
                position=edge_position,
                scale=np.array([base_size, edge_thickness, edge_height]),
                color=np.array([0.5, 0.5, 0.5]),
            )
            edge.set_collision_approximation("convexHull")
            edge.apply_physics_material(physics_material)
            scene.add(edge)

        # -----------------------------------------------------------
        # Rest of setup
        # -----------------------------------------------------------

        self.set_camera()
        return

    def set_robot(self) -> UR5eHandeye:
        """[summary]

        Returns:
            UR5e: [description]
        """
        working_dir = os.path.dirname(
            os.path.realpath(__file__)
        )  # same directory with this code
        # ur5e_usd_path = os.path.join(working_dir, "ur5e_handeye_gripper.usd")
        ur5e_usd_path = os.path.join(working_dir, "ur5e_handeye_gripper.usd")
        if os.path.isfile(ur5e_usd_path):
            pass
        else:
            raise Exception(f"{ur5e_usd_path} not found")

        ur5e_prim_path = find_unique_string_name(
            initial_name="/World/ur5e", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        ur5e_robot_name = find_unique_string_name(
            initial_name="my_ur5e",
            is_unique_fn=lambda x: not self.scene.object_exists(x),
        )
        return UR5eHandeye(
            prim_path=ur5e_prim_path, name=ur5e_robot_name, usd_path=ur5e_usd_path
        )

    def get_params(self) -> dict:
        params_representation = dict()
        params_representation["robot_name"] = {
            "value": self._robot.name,
            "modifiable": False,
        }
        params_representation["overhead_camera_name"] = {
            "value": self.overhead_camera_name,
            "modifiable": False,
        }
        params_representation["perspective_camera_name"] = {
            "value": self.perspective_camera_name,
            "modifiable": False,
        }
        return params_representation

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()

        observation_dict = dict()

        observation_dict[self._robot.name] = {
            "joint_positions": joints_state.positions,
            "end_effector_position": end_effector_position,
        }
        return observation_dict

    def set_camera(self):
        # Position camera 1m above the front platform, facing downward
        overhead_position = np.array([0.5, 0, 0.5])  # Centered above the platform
        overhead_orientation = np.array(
            [0.5, -0.5, 0.5, 0.5]
        )  # Facing downward (90 degrees around X-axis)
        perspective_position = np.array([1.0, 0, 0.5])  # Centered above the platform
        perspective_orientation = euler_angles_to_quat(np.array([np.pi*2, np.pi/2, np.pi]))

        # Create the camera
        self.camera = Camera(
            prim_path="/World/OverheadCamera",  # Unique prim path
            frequency=20,
            resolution=(540, 540),
            position=overhead_position,
            orientation=overhead_orientation,
        )
        self.camera.initialize()
        APERTURE_SIZE = 0.5
        self.camera.set_projection_mode("orthographic")  # Set orthographic projection
        self.camera.set_focal_length(1.93)
        self.camera.set_focus_distance(4)
        self.camera.set_horizontal_aperture(APERTURE_SIZE)  # Set horizontal aperture
        self.camera.set_vertical_aperture(
            APERTURE_SIZE
        )  # Set vertical aperture to the same value
        self.camera.set_clipping_range(0.01, 10000)

        # Add necessary frame processing

        self.camera.add_distance_to_camera_to_frame()
        self.camera.add_instance_segmentation_to_frame()
        self.camera.add_instance_id_segmentation_to_frame()
        self.camera.add_semantic_segmentation_to_frame()
        self.camera.add_bounding_box_2d_loose_to_frame()
        self.camera.add_bounding_box_2d_tight_to_frame()
        
        # Create the camera
        self.camera_persp = Camera(
            prim_path="/World/PerspCamera",  # Unique prim path
            frequency=20,
            resolution=(640, 480),
            position=perspective_position,
            orientation=perspective_orientation,
        )
        self.camera_persp.initialize()
        APERTURE_SIZE = 0.5
        self.camera_persp.set_focal_length(1.93)
        self.camera_persp.set_focus_distance(4)
        self.camera_persp.set_horizontal_aperture(APERTURE_SIZE)  # Set horizontal aperture
        self.camera_persp.set_vertical_aperture(
            APERTURE_SIZE
        )  # Set vertical aperture to the same value
        self.camera_persp.set_clipping_range(0.01, 10000)

        # Add necessary frame processing

        self.camera_persp.add_distance_to_camera_to_frame()
        self.camera_persp.add_instance_segmentation_to_frame()
        self.camera_persp.add_instance_id_segmentation_to_frame()
        self.camera_persp.add_semantic_segmentation_to_frame()
        self.camera_persp.add_bounding_box_2d_loose_to_frame()
        self.camera_persp.add_bounding_box_2d_tight_to_frame()
        return

    def get_camera_ortho(self):
        return self.camera

    def get_camera_persp(self):
        return self.camera_persp