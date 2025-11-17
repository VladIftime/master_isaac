"""
Simple Pick and Place Controller for UR5e with Parallel Gripper

This controller implements a straightforward state machine for pick and place operations.
It takes world coordinates directly and performs the manipulation in clear phases.
"""

import numpy as np
import carb
from typing import Optional, List
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.core.prims import SingleArticulation
import isaacsim.robot.manipulators.controllers as manipulators_controllers
from utils_robots.controllers.RMPFflow_pickplace import RMPFlowController


class SimplePickPlaceController(manipulators_controllers.PickPlaceController):
    """
    Simple pick and place controller that works in world coordinates.
    
    Phases (designed to avoid obstacles like boxes):
    0. Open gripper (preparation)
    1. Move above pick position (safe height, x,y from pick target)
    2. Lower straight down to grasp height (2cm above object)
    3. Close gripper
    4. Lift straight up to safe height
    5. Move horizontally to above place position
    6. Lower to placing height (5cm above ground)
    7. Open gripper
    8. Lift straight up
    9. Done
    """
    
    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt: Optional[List[float]] = None,
        end_effector_offset: Optional[np.ndarray] = None,
        lift_height: float = 0.50,
        grasp_height_offset: float = 0.02,
        place_height: float = 0.05,
    ) -> None:
        """
        Args:
            name: Controller name
            gripper: Parallel gripper instance
            robot_articulation: Robot articulation
            events_dt: Time duration for each phase (default provided)
            end_effector_offset: Offset for end effector positioning
            lift_height: Height to lift above objects (default: 0.15m = 15cm)
            grasp_height_offset: Distance above object to grasp (default: 0.02m = 2cm)
            place_height: Height above ground to place (default: 0.05m = 5cm)
        """
        # Phase timing: [open, above_pick, lower, close, lift, move_horiz, lower_place, open, lift]
        if events_dt is None:
            events_dt = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        
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
        self._lift_height = lift_height
        self._grasp_height_offset = grasp_height_offset
        self._place_height = place_height
        self._position_threshold = 0.05
        self._end_effector_orientation = None
        self._current_phase_target = None
        
        carb.log_info(f"SimplePickPlaceController '{name}' initialized")
        carb.log_info(f"  Lift height: {lift_height*100:.1f}cm")
        carb.log_info(f"  Grasp offset: {grasp_height_offset*100:.1f}cm above object")
        carb.log_info(f"  Place height: {place_height*100:.1f}cm above ground")
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
        end_effector_orientation: Optional[np.ndarray] = None,
        end_effector_offset: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute one step of the pick and place controller.
        
        Args:
            picking_position: World coordinates of pick location [x, y, z]
            placing_position: World coordinates of place location [x, y, z]
            current_joint_positions: Current robot joint positions
            end_effector_orientation: End effector orientation (optional)
            
        Returns:
            ArticulationAction to be applied to robot
        """
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([np.pi, 0, np.pi]))

        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.15])

        if self._pause or self.is_done():
            self.pause()
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        
        # Log phase start
        if self._t == 0:
            if self._event == 0:
                carb.log_warn("PHASE 0: Open gripper (preparation)")
                self._current_phase_target = None
            elif self._event == 1:
                target_pos = picking_position.copy()
                target_pos[2] += self._lift_height
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 1: Move above pick position [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 2:
                target_pos = picking_position.copy()
                target_pos[2] += self._grasp_height_offset
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 2: Lower to grasp height [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 3:
                carb.log_warn("PHASE 3: Close gripper")
                self._current_phase_target = None
            elif self._event == 4:
                target_pos = picking_position.copy()
                target_pos[2] += self._lift_height
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 4: Lift straight up [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 5:
                target_pos = placing_position.copy()
                target_pos[2] += self._lift_height
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 5: Move horizontally to above place [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 6:
                target_pos = placing_position.copy()
                target_pos[2] = self._place_height
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 6: Lower to place height [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            elif self._event == 7:
                carb.log_warn("PHASE 7: Open gripper")
                self._current_phase_target = None
            elif self._event == 8:
                target_pos = placing_position.copy()
                target_pos[2] += self._lift_height
                self._current_phase_target = target_pos
                carb.log_warn(f"PHASE 8: Lift straight up [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # Calculate position target with offset
        def get_target(base_pos):
            return base_pos + (self._end_effector_offset if self._end_effector_offset is not None else np.zeros(3))
        
        # Set default orientation if not provided
        if self._end_effector_orientation is None:
            self._end_effector_orientation = euler_angles_to_quat(np.array([np.pi/2, np.pi/2, -np.pi/2]))
        
        # Execute current phase
        if self._event == 0:
            # Phase 0: Open gripper (preparation)
            target_joint_positions = self._gripper.forward(action="open")
        elif self._event == 1:
            # Phase 1: Move above pick position (safe height, x,y from pick target)
            target_pos = picking_position.copy()
            target_pos[2] += self._lift_height
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 2:
            # Phase 2: Lower straight down to grasp height (2cm above object)
            target_pos = picking_position.copy()
            target_pos[2] += self._grasp_height_offset
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 3:
            # Phase 3: Close gripper
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 4:
            # Phase 4: Lift straight up to safe height
            target_pos = picking_position.copy()
            target_pos[2] += self._lift_height
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 5:
            # Phase 5: Move horizontally to above place position
            target_pos = placing_position.copy()
            target_pos[2] += self._lift_height
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 6:
            # Phase 6: Lower to placing height (5cm above ground)
            target_pos = placing_position.copy()
            target_pos[2] = self._place_height
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        elif self._event == 7:
            # Phase 7: Open gripper
            target_joint_positions = self._gripper.forward(action="open")
        elif self._event == 8:
            # Phase 8: Lift straight up
            target_pos = placing_position.copy()
            target_pos[2] += self._lift_height
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=get_target(target_pos)[:3],
                target_end_effector_orientation=self._end_effector_orientation,
            )
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        
        # Update timing
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

