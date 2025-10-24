from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import Articulation
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller import (
    RMPFlowController,
)
import numpy as np
import typing
from typing import Optional, List
import carb


class CustomPickPlaceController(manipulators_controllers.PickPlaceController):
    """Simple class that just extends the PickPlaceController class  from isaac.manipulators.controllers to lift heigher after grasping the object

        Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [0.01, 0.01, 1, 0.01, 0.1, 0.01, 0.005, 1, 0.01, 0.1]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
                attach_gripper=True,
            ),
            gripper=gripper,
            end_effector_initial_height=0.55,
            events_dt=events_dt,
        )
        self._grasp = False
        return

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        carb.log_warn(f"Current event: {self._event}")
        target_joint_positions = None
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 2:
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        # elif self._event == 5:
        #     target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0],
                placing_position[1],
                self._current_target_x,
                self._current_target_y,
            )
            target_height = self._get_target_hs(placing_position[2])
            carb.log_warn(f"Target height: {target_height}")
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
            if self._event == 6:
                self._grasp = self._is_grasped()
        return target_joint_positions

    def _is_grasped(self) -> bool:
        """Returns True if the gripper is not closed, False otherwise."""
        carb.log_warn(f"Gripper joint positions: {self._gripper.get_joint_positions()}")
        carb.log_warn(
            f"Gripper joint closed positions: {self._gripper.joint_closed_positions}"
        )
        return not (
            self._gripper.get_joint_positions()[0]
            >= self._gripper.joint_closed_positions[0]
            and self._gripper.get_joint_positions()[1]
            <= self._gripper.joint_closed_positions[1]
        )

    def get_grasp(self):
        """Returns True if the gripper is not closed, False otherwise."""
        return self._grasp

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            # Modified logic for phase 4
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError()
        return h
