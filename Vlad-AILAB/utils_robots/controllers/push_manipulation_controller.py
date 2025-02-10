import typing
import carb
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.universal_robots.controllers import RMPFlowController
from utils_robots.controllers.basic_manipulation_controller import (
    BasicManipulationController,
)


class PushManipulationController(BaseController):
    def __init__(
        self,
        robot,
        name: str,
        gripper: Gripper,
        cspace_controller: RMPFlowController,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        super().__init__(name=name)
        self._basic_controller = BasicManipulationController(
            name=name + "_basic",
            gripper=gripper,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
        )
        self._event = 0
        self._t = 0
        self._pause = False
        self._robot = robot
        self._total_events = 2

    def forward(
        self,
        pushing_position: np.ndarray,
        end_effector_orientation: np.ndarray,
        current_joint_positions: np.ndarray,
        workspace_limits: np.ndarray,
        original_joint_positions: np.ndarray,
        push_length: float,
    ) -> ArticulationAction:
        print(f"Executing: push at ({pushing_position[0]}, {pushing_position[1]}, {pushing_position[2]})")

        current_joint_positions = self._robot.get_joint_positions()
        # Ensure the position and orientation are numpy arrays
        pushing_position = np.array(pushing_position)
        end_effector_orientation = np.array(end_effector_orientation)
        
        # Define the phases of the push operation
        carb.log_warn(f"Event: {self._event}")
        carb.log_warn(f"Is done: {self._basic_controller.is_done()}")
        if self._event == 0:
            # Close gripper
            if self._basic_controller.is_done():
                current_joint_positions = self._robot.get_joint_positions()
                self._basic_controller.reset()
            action = self._basic_controller.close(
                current_joint_positions=current_joint_positions,
                end_effector_offset=np.array([0, 0, 0.14]),
            )

        elif self._event == 1:
            # Approach pushing point
            carb.log_warn(f"Event: approach pushing point")
            if self._basic_controller.is_done():
                current_joint_positions = self._robot.get_joint_positions()
                self._basic_controller.reset()
            # add a breakpoint here
            action = self._basic_controller.forward(
                target_position=np.array([pushing_position[0], pushing_position[1], pushing_position[2] + 0.06]),
                current_joint_positions=current_joint_positions,
                end_effector_orientation=end_effector_orientation,
            )
            # breakpoint()
        elif self._event == 2:
            carb.log_warn(f"Event: compute pushing direction and target location")
            if self._basic_controller.is_done():
                current_joint_positions = self._robot.get_joint_positions()
                self._basic_controller.reset()
            tool_rotation_angle = (end_effector_orientation[2] % np.pi) - np.pi / 2
            push_orientation = [1.0, 0.0]
            push_direction = np.array(
                [
                    push_orientation[0] * np.cos(tool_rotation_angle)
                    - push_orientation[1] * np.sin(tool_rotation_angle),
                    push_orientation[0] * np.sin(tool_rotation_angle)
                    + push_orientation[1] * np.cos(tool_rotation_angle),
                ]
            )
            target_x = min(
                max(
                    pushing_position[0] + push_direction[0] * push_length,
                    workspace_limits[0][0],
                ),
                workspace_limits[0][1],
            )
            target_y = min(
                max(
                    pushing_position[1] + push_direction[1] * push_length,
                    workspace_limits[1][0],
                ),
                workspace_limits[1][1],
            )
            carb.log_warn(f"Target position: {np.array([target_x, target_y, pushing_position[2]])}")
            action = self._basic_controller.forward(
                target_position=np.array([target_x, target_y, pushing_position[2]]),
                current_joint_positions=current_joint_positions,
                end_effector_orientation=end_effector_orientation,
            )
        elif self._event == 3:
            carb.log_warn(f"Event: move gripper to location above pushing point")
            if self._basic_controller.is_done():
                current_joint_positions = self._robot.get_joint_positions()
                self._basic_controller.reset()

            location_above_pushing_point = np.array(
                [
                    position[0],
                    position[1],
                    position[2] + 0.06,
                ]
            )
            action = self._basic_controller.forward(
                target_position=location_above_pushing_point,
                current_joint_positions=current_joint_positions,
                end_effector_orientation=end_effector_orientation,
            )
        elif self._event == 4:
            carb.log_warn(f"Event reset robot to original position")
            if self._basic_controller.is_done():
                current_joint_positions = self._robot.get_joint_positions()
                self._basic_controller.reset()
            action = ArticulationAction(joint_positions=original_joint_positions)

        carb.log_warn(f"Event: {self._event}")
        carb.log_warn(f"Is done: {self._basic_controller.is_done()}")
        # Update time and event
        if self._basic_controller.is_done():
            self._event += 1
            self._t = 0

        return action

    def reset(self) -> None:
        self._basic_controller.reset()
        self._event = 0
        self._t = 0
        self._pause = False

    def is_done(self) -> bool:
        return self._event >= self._total_events

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False
