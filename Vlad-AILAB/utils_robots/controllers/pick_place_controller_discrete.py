from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import Articulation
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

import isaacsim.robot.manipulators.controllers as manipulators_controllers
from utils_robots.controllers.RMPFflow_pickplace import RMPFlowController
import numpy as np
import typing
from typing import Optional, List
import carb


class DiscretePickPlaceController(manipulators_controllers.PickPlaceController):
    """
    Discrete Pick and Place Controller - uses discrete waypoints with SINGLE-AXIS movements
    
    All movements after Phase 0 are constrained to single axis to avoid collisions:
    - Vertical movements: Only Z changes (X,Y constant)
    - Horizontal movements: Only X,Y change (Z constant at safe height)
    
    Phases (10 total - parent class limit):
    0. Move to overhead position (joint space initialization)
    1. Move horizontally above picking position (X,Y at safe height)
    2. Lower vertically to picking position (only Z changes)
    3. Wait for robot to settle
    4. Close gripper
    5. Lift vertically to safe height (only Z changes)
    6. Move horizontally to above placing position (only X,Y change)
    7. Lower vertically to placing position (only Z changes)
    8. Open gripper to release
    9. Return to turned position (joint space)
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        events_dt: Optional[List[float]] = None,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_initial_height: Optional[float] = None,
    ) -> None:
        """
        Args:
            name: Controller name
            gripper: Parallel gripper instance
            robot_articulation: Robot articulation
            events_dt: Time for each phase (10 phases - parent class limit)
                       [overhead, horiz_to_pick, lower_z, settle, close, lift_z, horiz_to_place, lower_z, open, turned]
            end_effector_offset: Offset for end effector positioning
            end_effector_initial_height: Safe height for end effector (default: 0.35m = 35cm)
        """
        # Phase timings: [overhead, horiz_to_pick, lower_z, settle, close, lift_z, horiz_to_place, lower_z, open, turned]
        if events_dt is None:
            events_dt = [0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.01, 1.0, 0.01]
        
        # Set default end effector height if not provided
        if end_effector_initial_height is None:
            end_effector_initial_height = 0.35  # 50cm above workspace (safe height)
        
        # Get USD path from robot if available
        robot_usd_path = None
        if hasattr(robot_articulation, '_usd_path'):
            robot_usd_path = robot_articulation._usd_path
        
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
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )
        self._grasp = False
        self._end_effector_offset = end_effector_offset
        self._robot_articulation = robot_articulation
        self._position_threshold = 0.09
        self._end_effector_orientation = None
        self._current_phase_target = None
        self._safe_height = end_effector_initial_height
        
        # Joint space positions for overhead and turned positions (like push controller)
        self._overhead_position = np.array([np.pi, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        self._turned_position = np.array([np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        self._right_box_position = np.array([np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        self._left_box_position = np.array([3*np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        carb.log_warn(f"DiscretePickPlaceController '{name}' initialized")
        carb.log_warn(f"  Safe height: {self._safe_height*100:.1f}cm")
        carb.log_warn(f"  Phase timings: {events_dt}")
        return
    
    def _check_position_reached(self, target_position: np.ndarray) -> bool:
        """Check if end effector has reached target position."""
        ee_position, _ = self._robot_articulation.end_effector.get_world_pose()
        distance = np.linalg.norm(ee_position - target_position)
        return bool(distance < self._position_threshold)

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
            picking_position (np.ndarray): The object's position to be picked in world coordinates.
            placing_position (np.ndarray): The object's position to be placed in world coordinates.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if self._pause or self.is_done():
            self.pause()
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # Log phase start and set discrete target positions
        if self._t == 0:
            if self._event == 0:
                # Phase 0: Overhead position (joint space)
                carb.log_warn(f"PHASE 0: Move to overhead position (joint space)")
                self._current_phase_target = None
            elif self._event == 1:
                # Phase 1: Move horizontally above picking position (X,Y at safe height)
                self._current_phase_target = np.array([picking_position[0], picking_position[1], self._safe_height])
                carb.log_warn(f"PHASE 1: Move horizontally above pick (X,Y only) [{self._current_phase_target[0]:.3f}, {self._current_phase_target[1]:.3f}, {self._current_phase_target[2]:.3f}]")
            elif self._event == 2:
                # Phase 2: Lower vertically to picking position (only Z changes)
                self._current_phase_target = picking_position.copy()
                carb.log_warn(f"PHASE 2: Lower vertically to pick (Z only) [{self._current_phase_target[0]:.3f}, {self._current_phase_target[1]:.3f}, {self._current_phase_target[2]:.3f}]")
            elif self._event == 3:
                # Phase 3: Wait/settle
                carb.log_warn(f"PHASE 3: Wait for robot to settle (1.0s)")
                self._current_phase_target = None
            elif self._event == 4:
                # Phase 4: Close gripper
                carb.log_warn(f"PHASE 4: Close gripper")
                self._current_phase_target = None
            elif self._event == 5:
                # Phase 5: Lift vertically to safe height (only Z changes)
                self._current_phase_target = np.array([picking_position[0], picking_position[1], self._safe_height])
                carb.log_warn(f"PHASE 5: Lift vertically with object (Z only) [{self._current_phase_target[0]:.3f}, {self._current_phase_target[1]:.3f}, {self._current_phase_target[2]:.3f}]")
            elif self._event == 6:
                # Phase 6: Move horizontally to above placing position (only X,Y change)
                self._current_phase_target = np.array([placing_position[0], placing_position[1], self._safe_height])
                carb.log_warn(f"PHASE 6: Move horizontally to above place (X,Y only) [{self._current_phase_target[0]:.3f}, {self._current_phase_target[1]:.3f}, {self._current_phase_target[2]:.3f}]")
            elif self._event == 7:
                # Phase 7: Lower vertically to placing position (only Z changes)
                self._current_phase_target = placing_position.copy()
                carb.log_warn(f"PHASE 7: Lower vertically to place (Z only) [{self._current_phase_target[0]:.3f}, {self._current_phase_target[1]:.3f}, {self._current_phase_target[2]:.3f}]")
            elif self._event == 8:
                # Phase 8: Open gripper
                carb.log_warn(f"PHASE 8: Open gripper (1.0s settle)")
                self._current_phase_target = None
            elif self._event == 9:
                # Phase 9: Return to turned position (joint space)
                carb.log_warn(f"PHASE 9: Return to turned position (joint space)")
                self._current_phase_target = None

        # Calculate position target with offset
        def get_target(base_pos):
            return base_pos + (self._end_effector_offset if self._end_effector_offset is not None else np.zeros(3))
        
        # Set default orientation if not set
        if self._end_effector_orientation is None:
            self._end_effector_orientation = euler_angles_to_quat(np.array([np.pi/2, np.pi/2, -np.pi/2]))
        
        # Execute current phase - SINGLE-AXIS movements only after phase 0
        if self._event == 0:
            # Phase 0: Move to overhead position (joint space)
            target_joint_positions = ArticulationAction(joint_positions=self._overhead_position)
        elif self._event == 1:
            # Phase 1: Move horizontally above picking position (X,Y at safe height)
            # Single-axis: Only X,Y change, Z stays at safe height
            target_pos = np.array([picking_position[0], picking_position[1], self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 2:
            # Phase 2: Lower vertically to picking position (only Z changes)
            # Single-axis: Only Z changes, X,Y stay constant
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(picking_position)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 3:
            # Phase 3: Wait for robot to settle
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
        elif self._event == 4:
            # Phase 4: Close gripper
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 5:
            # Phase 5: Lift vertically to safe height (only Z changes)
            # Single-axis: Only Z changes, X,Y stay at picking position
            target_pos = np.array([picking_position[0], picking_position[1], self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 6:
            # Phase 6: Move horizontally to above placing position (only X,Y change)
            # Single-axis: Only X,Y change, Z stays at safe height
            target_pos = np.array([placing_position[0], placing_position[1], self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 7:
            # Phase 7: Lower vertically to placing position (only Z changes)
            # Single-axis: Only Z changes, X,Y stay constant
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(placing_position)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 8:
            # Phase 8: Open gripper
            target_joint_positions = self._gripper.forward(action="open")
        elif self._event == 9:
            # Phase 9: Return to turned position (joint space)
            target_joint_positions = ArticulationAction(joint_positions=self._turned_position)
        else:
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
        
        # Update timing
        self._t += self._events_dt[self._event]
        
        if self._t >= 1.0:
            if self._current_phase_target is not None:
                if self._check_position_reached(self._current_phase_target):
                    self._event += 1
                    self._t = 0
                    if self._event == 5:  # After phase 4 (lift with object)
                        self._grasp = self._is_grasped()
                elif self._t < 2.0:
                    pass  # Allow extra time
                else:
                    carb.log_warn(f"Phase {self._event} timeout, advancing anyway")
                    self._event += 1
                    self._t = 0
                    if self._event == 5:  # After phase 4 (lift with object)
                        self._grasp = self._is_grasped()
            else:
                self._event += 1
                self._t = 0
                if self._event == 5:  # After phase 4 (lift with object)
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

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event"""
        manipulators_controllers.PickPlaceController.reset(
            self, end_effector_initial_height=end_effector_initial_height, events_dt=events_dt
        )
        self._current_phase_target = None
        self._grasp = False
        return

