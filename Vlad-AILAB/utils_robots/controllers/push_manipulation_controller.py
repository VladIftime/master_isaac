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
from utils_robots.controllers.RMPFflow_pickplace import RMPFlowController
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

    Phases:
    0. Move to overhead position
    1. Close gripper
    2. Move to push start position
    3. Execute push
    4. Lift up vertically
    5. Return to home position
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
        # Phase timing: [overhead, close_gripper, move_to_start, push, lift_up, return]
        if events_dt is None:
            events_dt = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        robot_usd_path = getattr(robot_articulation, '_usd_path', None)
        
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
                attach_gripper=True,
                usd_path=robot_usd_path,
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        self._robot_articulation = robot_articulation
        self._end_effector_offset = end_effector_offset
        self._position_threshold = 0.05
        self._end_effector_orientation = None
        self._overhead_position = np.array([np.pi, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        self._turned_position = np.array([np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        self._current_phase_target = None
        self._lift_height = 0.15
        return
    
    def _check_position_reached(self, target_position: np.ndarray) -> bool:
        ee_position, _ = self._robot_articulation.end_effector.get_world_pose()
        distance = np.linalg.norm(ee_position - target_position)
        return bool(distance < self._position_threshold)

    def forward(
        self,
        push_start_position: np.ndarray,
        push_end_position: np.ndarray,
        original_position: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if self._pause or self.is_done():
            self.pause()
            return ArticulationAction(joint_positions=[None] * self._overhead_position.shape[0])

        if self._t == 0:
            if self._event == 0:
                carb.log_warn("PHASE 0: Overhead position")
                self._current_phase_target = None
            elif self._event == 1:
                carb.log_warn("PHASE 1: Close gripper")
                self._current_phase_target = None
            elif self._event == 2:
                carb.log_warn(f"PHASE 2: Move to push start [{push_start_position[0]:.3f}, {push_start_position[1]:.3f}, {push_start_position[2]:.3f}]")
                self._current_phase_target = push_start_position + (self._end_effector_offset if self._end_effector_offset is not None else np.zeros(3))
            elif self._event == 3:
                push_vector = push_end_position - push_start_position
                carb.log_warn(f"PHASE 3: Execute push - vector [{push_vector[0]:.3f}, {push_vector[1]:.3f}, {push_vector[2]:.3f}]")
                self._current_phase_target = push_end_position + (self._end_effector_offset if self._end_effector_offset is not None else np.zeros(3))
            elif self._event == 4:
                ee_pos, _ = self._robot_articulation.end_effector.get_world_pose()
                lift_target = ee_pos.copy()
                lift_target[2] += self._lift_height
                self._current_phase_target = lift_target
                carb.log_warn(f"PHASE 4: Lift up {self._lift_height*100:.1f}cm to [{lift_target[0]:.3f}, {lift_target[1]:.3f}, {lift_target[2]:.3f}]")
            elif self._event == 5:
                carb.log_warn("PHASE 5: Return home")
                self._current_phase_target = None

        # Calculate position target with offset
        def get_target(base_pos):
            return base_pos + (self._end_effector_offset if self._end_effector_offset is not None else np.zeros(3))
        
        if self._end_effector_orientation is None:
            self._end_effector_orientation = euler_angles_to_quat(np.array([np.pi/2, np.pi/2, -np.pi/2]))
        
        if self._event == 0:
            target_joint_positions = ArticulationAction(joint_positions=self._overhead_position)
        elif self._event == 1:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 2:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(push_start_position)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 3:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(push_end_position)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 4:
            if self._current_phase_target is None:
                ee_pos, _ = self._robot_articulation.end_effector.get_world_pose()
                self._current_phase_target = ee_pos.copy()
                self._current_phase_target[2] += self._lift_height
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._current_phase_target[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        else:
            target_joint_positions = ArticulationAction(joint_positions=self._turned_position)

        self._t += self._events_dt[self._event]
        
        if self._t >= 1.0:
            if self._current_phase_target is not None:
                if self._check_position_reached(self._current_phase_target):
                    self._event += 1
                    self._t = 0
                elif self._t < 2.0:
                    pass  # Allow extra time
                else:
                    carb.log_warn(f"Phase {self._event} timeout, advancing anyway")
                    self._event += 1
                    self._t = 0
            else:
                self._event += 1
                self._t = 0
            
        return target_joint_positions

