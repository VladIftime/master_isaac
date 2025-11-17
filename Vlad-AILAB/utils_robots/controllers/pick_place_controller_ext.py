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


class CustomPickPlaceController(manipulators_controllers.PickPlaceController):
    """
    Custom Pick and Place Controller with proper timing for UR5e with Parallel Gripper
    
    Phases:
    0. Move above picking position (lift_height above object)
    1. Lower down to picking position  
    2. Wait for robot inertia to settle
    3. Close gripper to grasp object
    4. Lift up while keeping grip (lifting the block)
    5. Move horizontally toward placing position (keeping height constant)
    6. Lower vertically toward placing height
    7. Open gripper to release object
    8. Lift up from placing position
    9. Return to picking position xy (at height)
    
    Timing strategy:
    - Quick movements for simple motions (0.01s)
    - Longer waits for settling (1.0s) 
    - Medium time for gripper operations (0.01-0.1s)
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
            events_dt: Time for each phase [above, lower, settle, close, lift, move, lower, open, lift, return]
            end_effector_offset: Offset for end effector positioning
            end_effector_initial_height: Safe height for end effector above workspace in meters
                                        (default: 0.50m = 50cm). This is the height the robot
                                        will move to during lifting and horizontal movements.
        """
        # Phase timings optimized for reliability
        # [above, lower, settle, close, lift, move_horiz, lower_place, open, lift, return]
        if events_dt is None:
            events_dt = [0.01, 0.01, 1.0, 0.01, 0.1, 0.01, 0.005, 1.0, 0.01, 0.1]
        
        # Set default end effector height if not provided
        if end_effector_initial_height is None:
            end_effector_initial_height = 0.50  # 50cm above workspace (safe height)
        
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
        self._position_threshold = 0.05
        self._end_effector_orientation = None
        self._current_phase_target = None
        
        # Use our own safe height variable instead of parent's self._h1
        self._safe_height = end_effector_initial_height
        
        carb.log_warn(f"CustomPickPlaceController '{name}' initialized")
        carb.log_warn(f"  Safe height (self._safe_height): {self._safe_height*100:.1f}cm")
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
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if self._pause or self.is_done():
            self.pause()
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # Log phase start and set targets
        if self._t == 0:
            if self._event == 0:
                target_pos = np.array([picking_position[0], picking_position[1], self._safe_height])
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 0: Move above pick [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 1:
                self._current_phase_target = picking_position.copy()
                carb.log_warn(f"PHASE 1: Lower to pick [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
            elif self._event == 2:
                carb.log_warn(f"PHASE 2: Wait for robot to settle (1.0s)")
                self._current_phase_target = None
            elif self._event == 3:
                carb.log_warn(f"PHASE 3: Close gripper")
                self._current_phase_target = None
            elif self._event == 4:
                target_pos = np.array([picking_position[0], picking_position[1], self._safe_height])
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 4: Lift with object [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 5:
                target_pos = np.array([placing_position[0], placing_position[1], self._safe_height])
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 5: Move horizontally to place [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 6:
                self._current_phase_target = placing_position.copy()
                carb.log_warn(f"PHASE 6: Lower to place [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
            elif self._event == 7:
                carb.log_warn(f"PHASE 7: Open gripper (1.0s settle)")
                self._current_phase_target = None
            elif self._event == 8:
                target_pos = np.array([placing_position[0], placing_position[1], self._safe_height])
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 8: Lift from place [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 9:
                target_pos = np.array([picking_position[0], picking_position[1], self._safe_height])
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 9: Return to pick position [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        # Calculate position target with offset
        def get_target(base_pos):
            return base_pos + (self._end_effector_offset if self._end_effector_offset is not None else np.zeros(3))
        
        # Set default orientation if not set
        if self._end_effector_orientation is None:
            self._end_effector_orientation = euler_angles_to_quat(np.array([np.pi/2, np.pi/2, -np.pi/2]))
        
        # Execute current phase
        if self._event == 0:
            # Phase 0: Move above picking position
            target_pos = np.array([picking_position[0], picking_position[1], self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 1:
            # Phase 1: Lower to picking position
            a = self._mix_sin(max(0, self._t))
            start_height = self._safe_height
            target_height = self._combine_convex(start_height, picking_position[2], a)
            target_pos = np.array([picking_position[0], picking_position[1], target_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 2:
            # Phase 2: Wait for robot to settle
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
        elif self._event == 3:
            # Phase 3: Close gripper
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 4:
            # Phase 4: Lift up while keeping grip
            target_pos = np.array([picking_position[0], picking_position[1], self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 5:
            # Phase 5: Move horizontally toward placing position
            a = self._mix_sin(self._t)
            target_x = (1 - a) * picking_position[0] + a * placing_position[0]
            target_y = (1 - a) * picking_position[1] + a * placing_position[1]
            target_pos = np.array([target_x, target_y, self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 6:
            # Phase 6: Lower to placing height
            a = self._mix_sin(self._t)
            start_height = self._safe_height
            target_height = self._combine_convex(start_height, placing_position[2], a)
            target_pos = np.array([placing_position[0], placing_position[1], target_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 7:
            # Phase 7: Open gripper
            target_joint_positions = self._gripper.forward(action="open")
        elif self._event == 8:
            # Phase 8: Lift up from placing position
            target_pos = np.array([placing_position[0], placing_position[1], self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 9:
            # Phase 9: Return to picking position xy (at height)
            a = self._mix_sin(self._t)
            target_x = (1 - a) * placing_position[0] + a * picking_position[0]
            target_y = (1 - a) * placing_position[1] + a * picking_position[1]
            target_pos = np.array([target_x, target_y, self._safe_height])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
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
                    if self._event == 6:
                        self._grasp = self._is_grasped()
                elif self._t < 2.0:
                    pass  # Allow extra time
                else:
                    carb.log_warn(f"Phase {self._event} timeout, advancing anyway")
                    self._event += 1
                    self._t = 0
                    if self._event == 6:
                        self._grasp = self._is_grasped()
            else:
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

    def _get_target_hs(self, target_height):
        """Legacy method - not used in this controller but kept for compatibility."""
        if self._event == 0:
            h = self._safe_height
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._safe_height, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            # Modified logic for phase 4
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._safe_height, a)
        elif self._event == 5:
            h = self._safe_height
        elif self._event == 6:
            h = self._combine_convex(self._safe_height, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._safe_height, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._safe_height
        else:
            raise ValueError()
        return h
