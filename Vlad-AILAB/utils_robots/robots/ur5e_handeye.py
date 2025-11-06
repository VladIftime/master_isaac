# Seongho Bak, Taewon Kim, 2023

from typing import Optional, List
import numpy as np
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
import carb
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.sensors.camera import Camera


class UR5eHandeye(SingleArticulation):
    """
    modified 'Franka' class
    modified from '~/.local/share/ov/pkg/isaac_sim-2022.2.0/exts/omni.isaac.franka/omni/isaac/franka/franka.py'

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "ur5e".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "ur5e",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._usd_path = usd_path  # Store USD path for RMPFlow configuration
        if end_effector_prim_name is not None:
            self._end_effector_prim_name = end_effector_prim_name
        else:
            self._end_effector_prim_name = "flange"
        
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = (
                    assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
                )
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/flange"

            else:
                self._end_effector_prim_path = prim_path + "/" + self._end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = [
                    "left_outer_knuckle_joint",
                    "right_outer_knuckle_joint",
                ]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.0, 0.0])
            if gripper_closed_position is None:
                gripper_closed_position = np.array([np.pi * 2 / 9, -np.pi * 2 / 9])

        else:
            if self._end_effector_prim_name is None:
                # Use tool0 - the standard UR robot tool center point
                # This should match what RMPFlow controls for position-based checks
                self._end_effector_prim_path = prim_path + "/tool0"
            else:
                self._end_effector_prim_path = prim_path + "/" + self._end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = [
                    "left_outer_knuckle_joint",
                    "right_outer_knuckle_joint",
                ]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.0, 0.0])
            if gripper_closed_position is None:
                gripper_closed_position = np.array([np.pi * 2 / 9, -np.pi * 2 / 9])

        # Get the number of DOFs for this robot after initialization
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
        )
        
        # Store gripper parameters to create it during initialization
        self._gripper_dof_names = gripper_dof_names
        self._gripper_open_position = gripper_open_position
        self._gripper_closed_position = gripper_closed_position
        self._gripper_deltas = deltas if deltas is not None else np.array([-np.pi * 2 / 9, np.pi * 2 / 9])

    @property
    def end_effector(self) -> SingleRigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper

    @property
    def rgb_cam(self) -> Camera:
        """[summary]

        Returns:
            Camera: [description]
        """
        return self._rgb_cam

    @property
    def depth_cam(self) -> Camera:
        """[summary]

        Returns:
            Camera: [description]
        """
        return self._depth_cam

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        
        self._end_effector = SingleRigidPrim(
            prim_path=self._end_effector_prim_path, name=self.name + "_end_effector"
        )
        self._end_effector.initialize(physics_sim_view)

        # Create and initialize the gripper after articulation is initialized
        if self._gripper_dof_names is not None:
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=self._gripper_dof_names,
                joint_opened_positions=self._gripper_open_position,
                joint_closed_positions=self._gripper_closed_position,
                action_deltas=self._gripper_deltas,
            )
            
            # Initialize gripper using articulation's methods directly
            self._gripper.initialize(
                physics_sim_view=physics_sim_view,
                articulation_apply_action_func=self.apply_action,
                get_joint_positions_func=self.get_joint_positions,
                set_joint_positions_func=self.set_joint_positions,
                dof_names=self.dof_names,
            )
        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        if self._gripper is not None:
            self._gripper.post_reset()
        return
