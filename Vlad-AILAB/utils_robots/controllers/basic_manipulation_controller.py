# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
import carb
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.universal_robots.controllers import RMPFlowController


class BasicManipulationController(BaseController):
    """
    A simple end-effector position, open/close control state machine for tutorials

    The phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - Phase 0: Move end_effector at the target position.

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
            from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 1 phase dt has to
            be defined. Defaults to None.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 1
    """

    def __init__(
        self,
        name: str,  # Controller name
        gripper: Gripper,  # Gripper
        cspace_controller: BaseController = RMPFlowController,  # Motion controller
        events_dt: typing.Optional[typing.List[float]] = None,  # Event dt
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0  # Set event
        self._t = 0  # Set time
        self._events_dt = events_dt  # Set event dt. The smaller the dt, the more motions are performed within one phase
        if self._events_dt is None:
            self._events_dt = [0.008]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(
                self._events_dt, list
            ):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 1:
                raise Exception("events dt length must be less than 1")
        self._cspace_controller = cspace_controller  # Set motion controller
        self._gripper = gripper  # Set gripper
        self._pause = False  # Disable simulation pause
        return

    # Function to move the end effector to the desired position
    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            target_position (np.ndarray):  The end-effector's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.14])
        # Execute when the simulation is paused or finished
        if self._pause or self.is_done():
            # Pause the simulation
            self.pause()
            # Return None in Action
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # Calculate the target x, y, z position considering the end effector offset
        # The end effector offset refers to the offset caused by the length from the end effector to the end of the gripper
        position_target = np.array(
            [
                target_position[0] + end_effector_offset[0],
                target_position[1] + end_effector_offset[1],
                target_position[2] + end_effector_offset[2],
            ]
        )
        # Set end effector orientation
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        # Create ArticulationAction
        target_joint_positions = self._cspace_controller.forward(
            target_end_effector_position=position_target,
            target_end_effector_orientation=end_effector_orientation,
        )

        # Increment event time by _event_dt
        # If _event_dt accumulates to 1 unit time, end the phase and reset event time
        self._t += self._events_dt[self._event]
        carb.log_warn(f"T: {self._t}")
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        # Return information about the position (x, y, _z) and orientation the end effector should move towards
        return target_joint_positions

    # Function to open the gripper fingers
    def open(
        self,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.14])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # Calculate target joint positions through gripper open action
        target_joint_positions = self._gripper.forward(action="open")

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        # Return the target joint positions of the gripper
        return target_joint_positions

    # Function to close the gripper fingers
    def close(
        self,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.14])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # Calculate target joint positions through gripper close action
        target_joint_positions = self._gripper.forward(action="close")

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        # Return the target joint positions of the gripper
        return target_joint_positions

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 1 phase dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 1
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) or not isinstance(
                self._events_dt, list
            ):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 1:
                raise Exception("events dt length must be less than 1")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return

    def _calculate_target_position(self, target_position, current_joint_positions):
        # Use linear interpolation for straight-line movement
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array(
            [current_joint_positions[0], current_joint_positions[1]]
        ) + alpha * np.array([target_position[0], target_position[1]])
        return xy_target

    def _get_alpha(self):
        # Use a linear progression for alpha
        if self._event == 0:
            return 0
        elif self._event == 1:
            return self._t  # Linear interpolation
        elif self._event == 2:
            return 1.0
        elif self._event == 3:
            return 1.0
        else:
            raise ValueError()
