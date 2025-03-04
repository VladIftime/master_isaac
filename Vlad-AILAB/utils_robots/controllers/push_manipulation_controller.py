# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import List, Optional

import carb
import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper
import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller import (
    RMPFlowController,
)
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
    ArticulationKinematicsSolver,
)
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
    KinematicsSolver,
)


class PushManipulationController(manipulators_controllers.PickPlaceController):
    """
    A controller for pushing objects with a robot.

    The controller follows these phases:
    - Phase 0: Close the gripper.
    - Phase 1: Move end_effector to the push start position.
    - Phase 2: Execute the push by moving in a straight line to the push end position.
    - Phase 3: Return to the original position.

    Args:
        name (str): Name id of the controller
        gripper (Gripper): a gripper controller for open/close actions.
        robot_articulation (SingleArticulation): The robot articulation to control.
        events_dt (Optional[List[float]], optional): Dt of each phase/event step. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt: Optional[List[float]] = None,
        end_effector_offset: Optional[np.ndarray] = None,
        original_position: Optional[np.ndarray] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [0.008, 0.03, 0.01, 0.008, 0.008]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
                attach_gripper=True,
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        self._end_effector_offset = end_effector_offset
        self._h0 = None
        self._h1 = 0.3 / get_stage_units()
        self._end_effector_orientation = None
        self._overhead_position = np.array(
            [
                np.pi,
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
        )
        self._end_effector_position = None
        self._original_position = original_position
        self._turned_position = np.array(
            [
                np.pi / 2,
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
        )
        return

    def forward(
        self,
        push_start_position: np.ndarray,
        push_end_position: np.ndarray,
        original_position: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            push_start_position (np.ndarray): The position to start the push.
            push_end_position (np.ndarray): The position to end the push.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if original_position is not None:
            self._original_position = original_position
        if end_effector_orientation is not None:
            self._end_effector_orientation = end_effector_orientation

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * self._overhead_position.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event == 0:
            carb.log_warn("Moving to overhead position")
            target_joint_positions = ArticulationAction(
                joint_positions=self._overhead_position
            )
        elif self._event == 1:
            # Close the gripper
            carb.log_warn("Closing gripper")
            target_joint_positions = self._gripper.forward(action="close")

        elif self._event == 2:
            # Move to push start position
            carb.log_warn("Moving to push start position")
            self._current_target_x = push_start_position[0]
            self._current_target_y = push_start_position[1]
            self._h0 = push_start_position[2]

            interpolated_xy = self._get_interpolated_xy(
                push_start_position[0],
                push_start_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            h = self._get_target_hs(self._h0)
            position_target = np.array(
                [
                    interpolated_xy[0] + self._end_effector_offset[0],
                    interpolated_xy[1] + self._end_effector_offset[1],
                    h + self._end_effector_offset[2],
                ]
            )
            if self._end_effector_orientation is None:
                self._end_effector_orientation = euler_angles_to_quat(
                    np.array([np.pi / 2, np.pi / 2, -np.pi / 2])
                )
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 3:
            # Execute the push
            carb.log_warn("Executing push")
            self._current_target_x = push_end_position[0]
            self._current_target_y = push_end_position[1]
            self._h0 = push_end_position[2]

            interpolated_xy = self._get_interpolated_xy(
                push_end_position[0],
                push_end_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            h = self._get_target_hs(self._h0)
            position_target = np.array(
                [
                    interpolated_xy[0] + self._end_effector_offset[0],
                    interpolated_xy[1] + self._end_effector_offset[1],
                    h + self._end_effector_offset[2],
                ]
            )
            carb.log_warn(f"Position target: {position_target}")
            if self._end_effector_orientation is None:
                self._end_effector_orientation = euler_angles_to_quat(
                    np.array([np.pi / 2, np.pi / 2, -np.pi / 2])
                )
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        else:
            # Return to original position using XYZ coordinates
            carb.log_warn("Returning to turned position")
            target_joint_positions = ArticulationAction(
                joint_positions=self._turned_position
            )

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 2:
            h = target_height
        elif self._event == 3:
            h = target_height
        else:
            raise ValueError()
        return h
