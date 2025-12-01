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
import copy
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.tasks import PickPlace
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api import World
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
        randomize_position: bool = True,
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
        self.randomize_position = randomize_position
        self._robot_name = None
        return

    def set_up_scene(self, scene) -> None:
        # Try first to import the scene from the usd file
        self._scene = scene
        # stage_utils.open_stage(os.path.realpath("ure5_stage_basket.usd"))

        # -----------------------------------------------------------
        # Set up the robot
        # -----------------------------------------------------------
        scene.add_default_ground_plane()
        self._robot = self.set_robot()

        scene.add(self._robot)
        carb.log_warn(f"Robot state: {self._robot.get_joints_state()}")
        self._robot.set_joints_default_state(
            positions=np.array(
                [
                    0,
                    -np.pi / 2,
                    -np.pi / 2,
                    -np.pi / 2,
                    np.pi / 2,
                    np.pi / 2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
        )
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
        self.workspace_limits = np.asarray([[0.2, 0.8], [-0.25, 0.35], [0.05, 0.15]])
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
            if not self.randomize_position:
                # Place in the center
                new_position = np.array(
                    [
                        platform_center[0],
                        platform_center[1],
                        0.175,
                    ]
                )
                placed_positions.append(new_position)
                self.set_usd_objects(i, new_position)
            else:
                while True:
                    new_position = np.array(
                        [
                            random.uniform(
                                platform_center[0] - 0.15,
                                platform_center[0] + 0.15,
                            ),
                            random.uniform(
                                platform_center[1] - 0.15,
                                platform_center[1] + 0.15,
                            ),
                            random.uniform(0.15, 0.2),
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
        self._robot_name = ur5e_robot_name
        return UR5eHandeye(
            prim_path=ur5e_prim_path, 
            name=ur5e_robot_name, 
            usd_path=ur5e_usd_path,
            end_effector_prim_name="flange"  # CRITICAL: Match USD structure and RMPFlow config
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

    def get_observations(self, goal_object_name: Optional[str] = None) -> dict:
        """
        Get the observations from the robot and the environment.
        Args:
            goal_object_name: Optional; the name of the goal object.
        Returns:
            dict: observation dictionary
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        rgb_image, depth_image = self.get_rgb_depth_images(self.camera)
        pointcloud = self.pointcloud_camera.get_pointcloud()
        # Display the pointcloud for debugging

        height_map = self.get_height_map(
            rgb_image=rgb_image,
            depth_image=depth_image,
            pointcloud=pointcloud,
            camera=self.camera,
            position=end_effector_position,
        )

        if goal_object_name is not None:
            goal_mask = self.get_semantic_mask(self.camera, goal_object_name)
        observation_dict = dict()

        if goal_object_name is not None:
            observation_dict = {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "pointcloud": pointcloud,
                "height_map": height_map,
                "goal_mask": goal_mask,
            }
        else:
            observation_dict = {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "pointcloud": pointcloud,
                "height_map": height_map,
            }
        return observation_dict

    def set_camera(self):
        # Position camera 1m above the front platform, facing downward
        overhead_position = np.array([0.5, 0, 0.3])  # Centered above the platform
        overhead_orientation = np.array(
            [0.5, -0.5, 0.5, 0.5]
        )  # Facing downward (90 degrees around X-axis)

        # Create the main camera
        self.camera = Camera(
            prim_path="/World/OverheadCamera",  # Unique prim path
            frequency=20,
            resolution=(540, 540),
            position=overhead_position,
            orientation=overhead_orientation,
        )
        self.camera.initialize()
        APERTURE_SIZE = 0.5
        # Set orthographic projection
        # Note: You may see "Unknown projection type, defaulting to pinhole" warnings from Hydra.
        # This is a rendering warning only - the camera sensor correctly uses orthographic projection.
        self.camera.set_projection_mode("orthographic")
        self.camera.set_focal_length(1.93)
        self.camera.set_focus_distance(4)
        self.camera.set_horizontal_aperture(APERTURE_SIZE)  # Set horizontal aperture
        self.camera.set_vertical_aperture(
            APERTURE_SIZE
        )  # Set vertical aperture to the same value
        self.camera.set_clipping_range(0.01, 10000)

        # Add necessary frame processing
        self.camera.add_pointcloud_to_frame()
        self.camera.add_distance_to_image_plane_to_frame()
        self.camera.add_distance_to_camera_to_frame()
        self.camera.add_instance_segmentation_to_frame()
        self.camera.add_instance_id_segmentation_to_frame()
        self.camera.add_semantic_segmentation_to_frame()
        self.camera.add_bounding_box_2d_loose_to_frame()
        self.camera.add_bounding_box_2d_tight_to_frame()

        # Create a second camera 3 meters above the first one, specifically for point cloud capture
        pointcloud_camera_position = np.array(
            [0.5, 0, 3.0]
        )  # 3m above the original camera
        pointcloud_camera_orientation = (
            overhead_orientation  # Same orientation as the main camera
        )

        self.pointcloud_camera = Camera(
            prim_path="/World/PointCloudCamera",  # Unique prim path
            frequency=20,
            resolution=(540, 540),
            position=pointcloud_camera_position,
            orientation=pointcloud_camera_orientation,
        )
        self.pointcloud_camera.initialize()

        # Configure the point cloud camera
        PC_APERTURE_SIZE = 0.5
        self.pointcloud_camera.set_projection_mode("orthographic")
        self.pointcloud_camera.set_focal_length(1.93)
        self.pointcloud_camera.set_focus_distance(4)
        self.pointcloud_camera.set_horizontal_aperture(PC_APERTURE_SIZE)
        self.pointcloud_camera.set_vertical_aperture(PC_APERTURE_SIZE)
        self.pointcloud_camera.set_clipping_range(0.01, 10000)

        # Only add point cloud processing to this camera
        self.pointcloud_camera.add_pointcloud_to_frame()
        self.pointcloud_camera.add_distance_to_image_plane_to_frame()
        self.pointcloud_camera.add_distance_to_camera_to_frame()

        return self.camera, self.pointcloud_camera

    def get_camera_ortho(self):
        """Get the orthographic camera."""
        return self.camera

    def get_camera_pointcloud(self):
        """Get the point cloud camera."""
        return self.pointcloud_camera

    def get_semantic_mask(self, camera: Camera, goal: str = None):
        """Generate a semantic mask from the camera's instance segmentation data.

        Args:
            camera: The camera object to capture images from.
            goal: Optional; the ID of the object to isolate in the mask.

        Returns:
            np.ndarray: The semantic mask image.
        """
        # Get instance segmentation data using the new API
        instance_segmentation_data = camera.get_instance_segmentation()
        instance_segmentation_image = instance_segmentation_data["data"]
        instance_segmentation_dict = instance_segmentation_data["info"]["idToSemantics"]
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
        # Extract RGB image using the new API
        rgb_image = camera.get_rgba()[:, :, :3]

        depth_image = camera.get_depth()

        # Normalize the depth image to 0-255 and convert to uint8
        depth_image_normalized = (depth_image / np.max(depth_image) * 255).astype(
            np.uint8
        )

        return rgb_image, depth_image_normalized

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
        scale = np.array([0.15, 0.15, 0.15])
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

    def get_height_map(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        pointcloud: np.ndarray,
        camera: Camera,
        position: np.ndarray,
    ) -> float:
        """Get the height at a specific position using the depth camera.

        Args:
            camera (Camera): The camera object to capture images from.
            position (np.ndarray): The world coordinates (x, y) to measure height.

        Returns:
            float: The height at the specified position.
        """
        self.heightmap_resolution = 0.001

        # Get camera intrinsics and pose
        cam_intrinsics = camera.get_intrinsics_matrix()
        cam_position, cam_orientation = camera.get_local_pose()
        carb.log_warn(
            f"cam_position: {cam_position}, cam_orientation: {cam_orientation}"
        )

        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(cam_orientation).as_matrix()

        # Construct the full transformation matrix
        cam_pose = np.eye(4)
        cam_pose[0:3, 0:3] = rotation_matrix
        cam_pose[0:3, 3] = cam_position
        depth_heightmap = get_heightmap(
            rgb_image,
            depth_image,
            pointcloud,
            cam_intrinsics,
            cam_pose,
            self.workspace_limits,
            self.heightmap_resolution,
        )

        return depth_heightmap

    def get_pointcloud(self, camera: Camera):
        pointcloud = camera.get_pointcloud()

        return pointcloud

    def pixel_to_robot_space(
        self,
        u: int,
        v: int,
        camera: Camera,
        display: bool = True,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        heightmap: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert pixel coordinates from orthographic camera view to robot space coordinates.

        Args:
            u (int): Pixel x-coordinate in the image
            v (int): Pixel y-coordinate in the image
            camera (Camera): The camera object to use for the conversion
            display (bool): Whether to display the image with the selected pixel marked
            rgb_image (Optional[np.ndarray]): Pre-captured RGB image
            depth_image (Optional[np.ndarray]): Pre-captured depth image

        Returns:
            np.ndarray: 3D point in robot space coordinates [x, y, z]
        
        """
        x, y, z = 0, 0, 0.14
        # Get current camera frame if not provided
        if rgb_image is None or depth_image is None:
            rgb_image = camera.get_rgba()[:, :, :3]
            depth_image = camera.get_depth()

        # Get camera parameters
        cam_position, cam_orientation = camera.get_local_pose()

        # Get image dimensions
        height, width = rgb_image.shape[:2]

        # Get aperture sizes
        horizontal_aperture = camera.get_horizontal_aperture()
        vertical_aperture = camera.get_vertical_aperture()

        # For orthographic projection with camera looking down:
        # The camera's X axis in image maps to Y in world space
        # The camera's Y axis in image maps to X in world space
        # This is due to the camera's orientation quaternion [0.5, -0.5, 0.5, 0.5]
        world_y = ((u / width) - 0.5) * horizontal_aperture
        world_x = cam_position[0] + ((v / height) - 0.5) * vertical_aperture

        # Get the depth at this pixel from heightmap
        # Find the nearest point in heightmap instead of exact match
        if heightmap is not None and len(heightmap) > 0:
            # Calculate distances to all heightmap points (only X and Y)
            distances = np.sqrt((heightmap[:, 0] - world_x)**2 + (heightmap[:, 1] - world_y)**2)
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            
            # Only use the heightmap Z if the nearest point is reasonably close (within 1cm)
            if nearest_distance < 0.01:
                z = heightmap[nearest_idx, 2]
                carb.log_info(f"Found heightmap Z={z:.4f} at distance {nearest_distance*1000:.2f}mm from target XY")
            else:
                carb.log_warn(f"Nearest heightmap point is {nearest_distance*100:.2f}cm away, using default Z")

        # Create point in world space
        point_world = np.array([world_x, world_y, z])

        # Log for debugging
        carb.log_warn(f"Camera position: {cam_position}")
        carb.log_warn(f"Pixel coordinates (u,v): ({u}, {v})")
        carb.log_warn(f"World coordinates: {point_world}")

        return point_world
    
    def detect_object_center_from_camera(
        self,
        camera: Camera,
        object_name: Optional[str] = None,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        heightmap: Optional[np.ndarray] = None,
        display: bool = True,
    ) -> Optional[np.ndarray]:
        """Detect an object's center position using camera data.
        
        Args:
            camera: The camera object
            object_name: Optional name of the object to detect (uses semantic segmentation)
            rgb_image: Pre-captured RGB image
            depth_image: Pre-captured depth image
            heightmap: Pre-computed heightmap
            display: Whether to visualize the detection
            
        Returns:
            np.ndarray: 3D coordinates [x, y, z] of the object's bottom center, or None if not found
        """
        # Get images if not provided
        if rgb_image is None:
            rgb_image = camera.get_rgba()[:, :, :3]
        if depth_image is None:
            depth_image = camera.get_depth()
        if heightmap is None:
            pointcloud = camera.get_pointcloud()
            cam_position, cam_orientation = camera.get_local_pose()
            heightmap = self.get_height_map(
                rgb_image=rgb_image,
                depth_image=depth_image,
                pointcloud=pointcloud,
                camera=camera,
                position=cam_position,
            )
        
        # Method 1: Use semantic segmentation if object name is provided
        if object_name is not None:
            try:
                semantic_mask = self.get_semantic_mask(camera, object_name)
                if semantic_mask is not None and np.any(semantic_mask > 0):
                    # Find the center of the mask
                    mask_points = np.where(semantic_mask > 0)
                    center_v = int(np.mean(mask_points[0]))  # row = v
                    center_u = int(np.mean(mask_points[1]))  # col = u
                    
                    carb.log_info(f"Object '{object_name}' detected at pixel ({center_u}, {center_v})")
                    
                    # Convert to robot space
                    object_center = self.pixel_to_robot_space(
                        u=center_u,
                        v=center_v,
                        camera=camera,
                        display=False,
                        rgb_image=rgb_image,
                        depth_image=depth_image,
                        heightmap=heightmap,
                    )
                    
                    # Find the minimum z in the mask region to get the bottom of the object
                    if heightmap is not None:
                        mask_heightmap_indices = []
                        for point in heightmap:
                            # Check if this heightmap point corresponds to a masked pixel
                            pixel_u, pixel_v = self.world_to_pixel(point[0], point[1], camera)
                            if 0 <= pixel_v < semantic_mask.shape[0] and 0 <= pixel_u < semantic_mask.shape[1]:
                                if semantic_mask[pixel_v, pixel_u] > 0:
                                    mask_heightmap_indices.append(point)
                        
                        if mask_heightmap_indices:
                            mask_heights = [p[2] for p in mask_heightmap_indices]
                            min_z = np.min(mask_heights)
                            object_center[2] = min_z + 0.01  # Add small offset above the bottom
                    

                    
                    return object_center
            except Exception as e:
                carb.log_warn(f"Semantic segmentation failed: {e}")
        
        # Method 2: Use depth-based segmentation (find highest point = object on platform)
        # Find objects by looking for height above the platform
        platform_height = 0.05  # Approximate platform height
        object_threshold = platform_height + 0.02  # Objects at least 2cm above platform
        
        # Filter heightmap for potential object points
        if heightmap is None:
            carb.log_warn("Heightmap is None, cannot detect object")
            return None
        
        object_points = [p for p in heightmap if p[2] > object_threshold]
        
        if len(object_points) > 0:
            # Find center of object points
            object_points_array = np.array(object_points)
            center_x = np.mean(object_points_array[:, 0])
            center_y = np.mean(object_points_array[:, 1])
            min_z = np.min(object_points_array[:, 2])
            
            object_center = np.array([center_x, center_y, min_z + 0.01])
            
            carb.log_info(f"Object detected at world position: {object_center}")
            
            if display and rgb_image is not None:
                # Create a mask for visualization
                height, width = rgb_image.shape[:2]
                detection_mask = np.zeros((height, width), dtype=np.uint8)
                
                for point in object_points:
                    pixel_u, pixel_v = self.world_to_pixel(point[0], point[1], camera)
                    if 0 <= pixel_v < height and 0 <= pixel_u < width:
                        detection_mask[pixel_v, pixel_u] = 255
                
                # Find approximate center pixel
                center_u, center_v = self.world_to_pixel(float(center_x), float(center_y), camera)
                # self._display_detection(rgb_image, detection_mask, center_u, center_v, object_center)
                pass
            
            return object_center
        
        carb.log_warn("No object detected in camera view")
        return None
    
    def world_to_pixel(self, world_x: float, world_y: float, camera: Camera) -> tuple:
        """Convert world coordinates to pixel coordinates.
        
        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
            camera: Camera object
            
        Returns:
            tuple: (u, v) pixel coordinates
        """
        cam_position, _ = camera.get_local_pose()
        horizontal_aperture = camera.get_horizontal_aperture()
        vertical_aperture = camera.get_vertical_aperture()
        
        # Get image resolution
        resolution = camera.get_resolution()
        width, height = resolution
        
        # Inverse of pixel_to_robot_space conversion
        u = int(((world_y / horizontal_aperture) + 0.5) * width)
        v = int((((world_x - cam_position[0]) / vertical_aperture) + 0.5) * height)
        
        return u, v
    

        
        carb.log_info(f"Object center at pixel ({center_u}, {center_v}) = world {world_coords}")
    
    def get_box_center_from_image_center(
        self,
        camera: Camera,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        heightmap: Optional[np.ndarray] = None,
        display: bool = False,
    ) -> np.ndarray:
        """Get the 3D world coordinates of the box at the center of the camera image.
        
        This is a convenience method that assumes the camera is positioned above the box,
        so the center of the image corresponds to the center of the box.
        
        Args:
            camera: The camera object
            rgb_image: Pre-captured RGB image (optional)
            depth_image: Pre-captured depth image (optional)
            heightmap: Pre-computed heightmap (optional)
            display: Whether to visualize the detection
            
        Returns:
            np.ndarray: 3D coordinates [x, y, z] of the box center
        """
        # Get images if not provided
        if rgb_image is None:
            rgb_image = camera.get_rgba()[:, :, :3]
        if depth_image is None:
            depth_image = camera.get_depth()
        if heightmap is None:
            pointcloud = camera.get_pointcloud()
            cam_position, cam_orientation = camera.get_local_pose()
            heightmap = self.get_height_map(
                rgb_image=rgb_image,
                depth_image=depth_image,
                pointcloud=pointcloud,
                camera=camera,
                position=cam_position,
            )
        
        # Get image center
        height, width = rgb_image.shape[:2]
        center_u = width // 2
        center_v = height // 2
        
        carb.log_info(f"Getting box center from image center: pixel ({center_u}, {center_v})")
        
        # Convert center pixel to world coordinates
        box_center = self.pixel_to_robot_space(
            u=center_u,
            v=center_v,
            camera=camera,
            display=display,
            rgb_image=rgb_image,
            depth_image=depth_image,
            heightmap=heightmap,
        )
        
        return box_center
