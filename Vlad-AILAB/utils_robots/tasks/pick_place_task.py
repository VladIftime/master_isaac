# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import time
import carb
from matplotlib import pyplot as plt
import omni
import copy
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.tasks import PickPlace
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import create_prim, get_prim_path, define_prim
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.prims import RigidPrim, GeometryPrim
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom, UsdPhysics
from isaacsim.core.utils.rotations import euler_angles_to_quat
from dev_utils.point_cloud_utils import depth_image_from_distance_image
from dev_utils.point_cloud_utils import get_heightmap
from scipy.spatial.transform import Rotation as R

from utils_robots.robots.ur5e_handeye import UR5eHandeye
import os, random
import numpy as np
from typing import Optional
from pathlib import Path
from PIL import Image


class UR5ePickPlace(PickPlace):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "ur5_pick_place".
    """

    def __init__(
        self,
        name: str = "ur5e_pick_place",
        stage_usd_path: Optional[str] = None,
        number_of_objects: Optional[int] = 1,
        robot_name: Optional[str] = "ur5e",
        overhead_camera_name: Optional[str] = "overhead_camera",
        pespective_camera_name: Optional[str] = "perspective_camera",
    ) -> None:
        super().__init__(name=name)
        self.stage_usd_path = stage_usd_path
        self.load_stage = True
        self.imported_objects_prim_path = "/World/object"
        self.objects_position_list = []
        self.objects_orientation_list = []
        self.objects_name_list = []
        self.robot_name = robot_name
        self.overhead_camera_name = overhead_camera_name
        self.perspective_camera_name = pespective_camera_name
        self.number_of_objects = number_of_objects
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
        self.platform_center = np.array([0.5, 0, 0.025])
        self.base_size = 0.5
        self.edge_height = 0.15

        # 1. Base plate with high-friction material
        self.platform_base = FixedCuboid(  # Changed to FixedCuboid
            prim_path="/World/PlatformBase",
            name="platform_base",
            position=self.platform_center,
            scale=np.array([self.base_size, self.base_size, 0.05]),
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
        self.edge_thickness = 0.02
        self.edge_vertical_offset = 0.025 + (
            self.edge_height / 2
        )  # Proper Z positioning

        # Front/back edges
        for direction, x_sign in [("Front", 1), ("Back", -1)]:
            edge_position = self.platform_center + np.array(
                [
                    x_sign * (self.base_size / 2 - self.edge_thickness / 2),
                    0,
                    self.edge_vertical_offset,
                ]
            )
            edge = FixedCuboid(  # Changed to FixedCuboid
                prim_path=f"/World/PlatformEdge_{direction}",
                name=f"platform_edge_{direction.lower()}",  # Unique name
                position=edge_position,
                scale=np.array(
                    [
                        self.edge_thickness,
                        self.base_size + self.edge_thickness,
                        self.edge_height,
                    ]
                ),
                color=np.array([0.5, 0.5, 0.5]),
            )
            edge.set_collision_approximation("convexHull")
            edge.apply_physics_material(physics_material)
            scene.add(edge)

        # Left/right edges
        for direction, y_sign in [("Left", 1), ("Right", -1)]:
            edge_position = self.platform_center + np.array(
                [
                    0,
                    y_sign * (self.base_size / 2 - self.edge_thickness / 2),
                    self.edge_vertical_offset,
                ]
            )
            edge = FixedCuboid(  # Changed to FixedCuboid
                prim_path=f"/World/PlatformEdge_{direction}",
                name=f"platform_edge_{direction.lower()}",  # Unique name
                position=edge_position,
                scale=np.array([self.base_size, self.edge_thickness, self.edge_height]),
                color=np.array([0.5, 0.5, 0.5]),
            )
            edge.set_collision_approximation("convexHull")
            edge.apply_physics_material(physics_material)
            scene.add(edge)
        self.workspace_limits = np.asarray([[0.25, 0.75], [-0.25, 0.25], [0.05, 0.15]])
        # -----------------------------------------------------------
        # Rest of setup
        # -----------------------------------------------------------
        self.set_objects()
        self.set_camera()
        return

    def set_objects(self):
        self._stage = get_current_stage()
        # Get information about YCB Dataset objects
        working_dir = os.path.dirname(os.path.realpath(__file__))
        ycb_path = os.path.join(Path(Path(working_dir).parent).parent, "dataset/ycb")
        obj_dirs = [
            os.path.join(ycb_path, obj_name) for obj_name in os.listdir(ycb_path)
        ]
        obj_dirs.sort()
        object_info = {}
        label2name = {}
        total_object_num = len(obj_dirs)
        for obj_idx, obj_dir in enumerate(obj_dirs):
            usd_file = os.path.join(obj_dir, "final.usd")
            object_info[obj_idx] = {
                "name": os.path.basename(obj_dir),
                "usd_file": usd_file,
                "label": obj_idx,
            }
            label2name[obj_idx] = os.path.basename(obj_dir)

        # Select usd file path for random objects
        self.objects_list = random.sample(
            list(object_info.values()), self.number_of_objects
        )
        self.objects_usd_list = []
        for obj_info in self.objects_list:
            self.objects_usd_list.append(obj_info["usd_file"])

        # Place objects randomly on the platform
        platform_half_size = (
            (self.base_size - self.edge_thickness) / 2,
            (self.base_size - self.edge_thickness) / 2,
        )
        platform_center = self.platform_center

        # Store placed object positions
        placed_positions = []

        # Define a function to check for overlap
        def is_overlapping(new_pos, existing_positions, min_distance):
            for pos in existing_positions:
                if np.linalg.norm(new_pos - pos) < min_distance:
                    return True
            return False

        # Place objects randomly on the platform
        for i in range(len(self.objects_list)):
            while True:
                new_position = np.array(
                    [
                        random.uniform(
                            platform_center[0] - platform_half_size[0],
                            platform_center[0] + platform_half_size[0],
                        ),
                        random.uniform(
                            platform_center[1] - platform_half_size[1],
                            platform_center[1] + platform_half_size[1],
                        ),
                        random.uniform(0.1, 0.3),
                    ]
                )
                # Check for overlap with a minimum distance of 0.1 (adjust as needed)
                if not is_overlapping(new_position, placed_positions, min_distance=0.1):
                    placed_positions.append(new_position)
                    self.set_usd_objects(i, new_position)
                    break

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
        perspective_orientation = euler_angles_to_quat(
            np.array([np.pi * 2, np.pi / 2, np.pi])
        )

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
        self.camera_persp.set_horizontal_aperture(
            APERTURE_SIZE
        )  # Set horizontal aperture
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

    def get_semantic_mask(self, camera: Camera, goal: str = None):
        """Generate a semantic mask from the camera's instance segmentation data.

        Args:
            camera: The camera object to capture images from.
            goal: Optional; the ID of the object to isolate in the mask.

        Returns:
            np.ndarray: The semantic mask image.
        """
        # Capture the current frame
        frame = camera.get_current_frame()

        # Extract instance segmentation data
        instance_segmentation_image = frame["instance_segmentation"]["data"]
        instance_segmentation_dict = frame["instance_segmentation"]["info"][
            "idToSemantics"
        ]
        carb.log_warn(instance_segmentation_dict)

        # Initialize the mask with zeros
        semantic_mask = np.zeros_like(instance_segmentation_image, dtype=np.uint8)

        if goal is not None:
            # Find the integer value corresponding to the goal string
            goal_value = None
            for key, value in instance_segmentation_dict.items():
                if value["class"] == goal:
                    goal_value = int(key)
                    break

            if goal_value is not None:
                # Create a mask for the goal object
                semantic_mask[instance_segmentation_image == goal_value] = 255
            else:
                print(f"Goal '{goal}' not found in instance segmentation dictionary.")
        else:
            print("No goal specified, returning empty mask.")

        return semantic_mask

    def get_rgb_depth_images(self, camera: Camera):
        """Capture and return RGB, normalized depth, and semantic mask images from the given camera.

        Args:
            camera: The camera object to capture images from.
            goal: Optional; the ID of the object to isolate in the semantic mask.

        Returns:
            tuple: A tuple containing the RGB image, the normalized depth image, and the semantic mask.
        """
        # Capture the current frame
        frame = camera.get_current_frame()

        # Extract RGB image
        rgb_image = frame["rgba"][:, :, :3]

        # Extract and process distance image to get depth image
        distance_image = frame["distance_to_camera"]
        intrinsics = camera.get_intrinsics_matrix()
        depth_image = depth_image_from_distance_image(distance_image, intrinsics)

        # Normalize the depth image to 0-255 and convert to uint8
        depth_image_normalized = (depth_image / np.max(depth_image) * 255).astype(
            np.uint8
        )

        return rgb_image, depth_image_normalized, distance_image

    def set_usd_objects(self, object_number: int, object_position: np.ndarray) -> None:
        # Define the prim path for the object
        object_prim_path = self.imported_objects_prim_path + f"_{object_number}"
        geometry_prim_path = (
            self.imported_objects_prim_path + f"/geometry_prim_{object_number}"
        )

        # Create the prims
        define_prim(object_prim_path)
        define_prim(geometry_prim_path)

        # Add reference to the stage
        self._task_object = add_reference_to_stage(
            usd_path=self.objects_usd_list[object_number],
            prim_path=object_prim_path,
        )

        # Remove RigidBodyAPI from intermediate Xform nodes
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(object_prim_path)
        a_name_prim = prim.GetChild("a_name")
        if a_name_prim and a_name_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            a_name_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

        # Ensure orientations is a 2D array
        orientation = np.array([1, 0, 0, 0])
        orientations = np.expand_dims(orientation, axis=0)

        # Ensure scales is a 2D array
        scale = np.array([0.2, 0.2, 0.2])
        scales = np.expand_dims(scale, axis=0)

        # Ensure masses is a 1D array
        mass = np.array([0.01])

        # Ensure collisions is a 1D array
        collision = np.array([True])

        # Create the rigid prim
        rigid_prim = RigidPrim(
            prim_paths_expr=object_prim_path,
            positions=np.expand_dims(object_position, axis=0),  # Ensure positions is 2D
            orientations=orientations,
            name="rigid_prim",
            scales=scales,
            masses=mass,  # Ensure masses is a 1D array
        )
        rigid_prim.enable_rigid_body_physics()

        # Create the geometry prim
        geometry_prim = GeometryPrim(
            prim_paths_expr=geometry_prim_path,
            name="geometry_prim",
            positions=np.expand_dims(object_position, axis=0),  # Ensure positions is 2D
            orientations=orientations,
            scales=scales,
            collisions=collision,  # Ensure collisions is a 1D array
        )
        geometry_prim.apply_physics_materials(
            PhysicsMaterial(
                prim_path=self.imported_objects_prim_path
                + f"/physics_material_{object_number}",
                static_friction=10,
                dynamic_friction=10,
                restitution=None,
            )
        )

        # Add or update semantics
        add_update_semantics(
            prim=prim, semantic_label=self.get_object_name(object_number)
        )

    def get_object_name(self, object_number: int) -> str:
        return self.objects_list[object_number]["name"]

    def get_height_at_position(self, camera: Camera, position: np.ndarray) -> float:
        """Get the height at a specific position using the depth camera.

        Args:
            camera (Camera): The camera object to capture images from.
            position (np.ndarray): The world coordinates (x, y) to measure height.

        Returns:
            float: The height at the specified position.
        """
        self.heightmap_resolution = 0.001

        # Capture the current frame
        rgb_image, depth_image, distance_image = self.get_rgb_depth_images(camera)

        # Get camera intrinsics and pose
        cam_intrinsics = camera.get_intrinsics_matrix()
        cam_position, cam_orientation = camera.get_local_pose()
        carb.log_warn(f"cam_position: {cam_position}, cam_orientation: {cam_orientation}")

        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(cam_orientation).as_matrix()

        # Construct the full transformation matrix
        cam_pose = np.eye(4)
        cam_pose[0:3, 0:3] = rotation_matrix
        cam_pose[0:3, 3] = cam_position

        # Compute the heightmap
        _, depth_heightmap = get_heightmap(
            rgb_image,
            depth_image,
            cam_intrinsics,
            cam_pose,
            self.workspace_limits,
            self.heightmap_resolution,
        )

        # Convert the world position to heightmap pixel coordinates
        pixel_x = int(
            (position[0] - self.workspace_limits[0][0]) / self.heightmap_resolution
        )
        pixel_y = int(
            (position[1] - self.workspace_limits[1][0]) / self.heightmap_resolution
        )

        # Get the height value from the heightmap
        height = depth_heightmap[pixel_y, pixel_x]

        return height
