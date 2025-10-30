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
        # Event time increments per simulation step
        # Format: [overhead, close_gripper, move_to_start, push, return]
        # At 60Hz physics: 0.01 increment = ~100 steps to reach t=1.0 (~1.67 seconds per phase)
        # Faster values = quicker phase progression
        if events_dt is None:
            events_dt = [0.005, 0.01, 0.005, 0.003, 0.005]  # Balanced speed
            # Element1: move to overhead position
            # Element2: close gripper
            # Element3: move to push start position
            # Element4: push
            # Element5: return to original position
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
                usd_path=robot_usd_path,  # Pass USD path for accurate kinematics
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        self._robot_articulation = robot_articulation
        self._end_effector_offset = end_effector_offset
        self._position_threshold = 0.05  # 5cm threshold for reaching waypoint
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
        self._current_phase_target = None
        
        return
    
    def _check_position_reached(self, target_position: np.ndarray) -> bool:
        """Check if end effector has reached the target position.
        
        Args:
            target_position: Target XYZ position in world frame
            
        Returns:
            bool: True if within threshold distance
        """
        try:
            ee_position, _ = self._robot_articulation.end_effector.get_world_pose()
            distance = np.linalg.norm(ee_position - target_position)
            return bool(distance < self._position_threshold)
        except Exception as e:
            carb.log_warn(f"Could not check position: {e}")
            return False

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

        # Log phase transitions and set targets
        if self._t == 0:
            if self._event == 0:
                carb.log_info("="*60)
                carb.log_info("PHASE 0: Moving to overhead position")
                carb.log_info("="*60)
                self._current_phase_target = None  # Joint space control
            elif self._event == 1:
                carb.log_info("="*60)
                carb.log_info("PHASE 1: Closing gripper")
                carb.log_info("="*60)
                self._current_phase_target = None  # Gripper control
            elif self._event == 2:
                carb.log_info("="*60)
                carb.log_info("PHASE 2: Moving to push start position")
                carb.log_info(f"  Base Target: [{push_start_position[0]:.3f}, {push_start_position[1]:.3f}, {push_start_position[2]:.3f}]")
                # Calculate actual EE target with offset
                if self._end_effector_offset is not None:
                    self._current_phase_target = push_start_position + self._end_effector_offset
                    carb.log_info(f"  Gripper Offset: [{self._end_effector_offset[0]:.3f}, {self._end_effector_offset[1]:.3f}, {self._end_effector_offset[2]:.3f}]")
                else:
                    self._current_phase_target = push_start_position.copy()
                carb.log_info(f"  EE Target (World): [{self._current_phase_target[0]:.3f}, {self._current_phase_target[1]:.3f}, {self._current_phase_target[2]:.3f}]")
                carb.log_info("  RMPFlow will plan collision-free path to this target")
                carb.log_info("="*60)
            elif self._event == 3:
                carb.log_info("="*60)
                carb.log_info("PHASE 3: Executing push")
                carb.log_info(f"  Base Start: [{push_start_position[0]:.3f}, {push_start_position[1]:.3f}, {push_start_position[2]:.3f}]")
                carb.log_info(f"  Base End:   [{push_end_position[0]:.3f}, {push_end_position[1]:.3f}, {push_end_position[2]:.3f}]")
                push_vector = push_end_position - push_start_position
                carb.log_info(f"  Push vector: [{push_vector[0]:.3f}, {push_vector[1]:.3f}, {push_vector[2]:.3f}]")
                # Calculate actual EE target with offset
                if self._end_effector_offset is not None:
                    self._current_phase_target = push_end_position + self._end_effector_offset
                    carb.log_info(f"  Gripper Offset: [{self._end_effector_offset[0]:.3f}, {self._end_effector_offset[1]:.3f}, {self._end_effector_offset[2]:.3f}]")
                else:
                    self._current_phase_target = push_end_position.copy()
                carb.log_info(f"  EE Target (World): [{self._current_phase_target[0]:.3f}, {self._current_phase_target[1]:.3f}, {self._current_phase_target[2]:.3f}]")
                carb.log_info("  RMPFlow will plan collision-free path to this target")
                carb.log_info("="*60)
            elif self._event == 4:
                carb.log_info("="*60)
                carb.log_info("PHASE 4: Returning to home position")
                carb.log_info("="*60)
                self._current_phase_target = None  # Joint space control

        if self._event == 0:
            target_joint_positions = ArticulationAction(
                joint_positions=self._overhead_position
            )
        elif self._event == 1:
            # Close the gripper
            target_joint_positions = self._gripper.forward(action="close")

        elif self._event == 2:
            # Move to push start position
            # RMPFlow handles motion planning, so just command the target directly
            
            # Apply end effector offset if available, otherwise use default
            if self._end_effector_offset is not None:
                position_target = np.array(
                    [
                        push_start_position[0] + self._end_effector_offset[0],
                        push_start_position[1] + self._end_effector_offset[1],
                        push_start_position[2] + self._end_effector_offset[2],
                    ]
                )
            else:
                position_target = push_start_position.copy()
            
            # Debug logging every 500 steps (less spam)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            # Debug logging to verify tool0 coordinates match RMPFlow
            if self._debug_counter % 200 == 0:
                ee_pos, ee_ori = self._robot_articulation.end_effector.get_world_pose()
                robot_base_pos, robot_base_ori = self._robot_articulation.get_world_pose()
                carb.log_warn(f"[Phase 2 Coordinate Debug - Step {self._debug_counter}]")
                carb.log_warn(f"  Robot base (world):    [{robot_base_pos[0]:.3f}, {robot_base_pos[1]:.3f}, {robot_base_pos[2]:.3f}]")
                carb.log_warn(f"  push_start_position:   [{push_start_position[0]:.3f}, {push_start_position[1]:.3f}, {push_start_position[2]:.3f}]")
                carb.log_warn(f"  End effector offset:   [{self._end_effector_offset[0] if self._end_effector_offset is not None else 0:.3f}, {self._end_effector_offset[1] if self._end_effector_offset is not None else 0:.3f}, {self._end_effector_offset[2] if self._end_effector_offset is not None else 0:.3f}]")
                carb.log_warn(f"  Commanding EE target:  [{position_target[0]:.3f}, {position_target[1]:.3f}, {position_target[2]:.3f}]")
                carb.log_warn(f"  Actual EE position:    [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                carb.log_warn(f"  Position error:        [{position_target[0]-ee_pos[0]:.3f}, {position_target[1]-ee_pos[1]:.3f}, {position_target[2]-ee_pos[2]:.3f}]")
                
                # Check if RMPFlow has the correct robot base pose
                if hasattr(self._cspace_controller, '_motion_policy'):
                    mp = self._cspace_controller._motion_policy
                    carb.log_warn(f"  RMPFlow controller type: {type(self._cspace_controller).__name__}")
                    
                    # Check internal robot state
                    base_status = []
                    if hasattr(mp, '_robot_pos'):
                        base_status.append(f"_robot_pos={mp._robot_pos}")
                    if hasattr(mp, '_robot_rot'):
                        base_status.append(f"_robot_rot={mp._robot_rot}")
                    if hasattr(mp, '_robot_base_moved'):
                        base_status.append(f"_robot_base_moved={mp._robot_base_moved}")
                    
                    if base_status:
                        carb.log_warn(f"  ✓ RMPFlow internal state: {', '.join(base_status)}")
                    else:
                        carb.log_warn(f"  ⚠️  RMPFlow internal state: Cannot access _robot_pos or _robot_rot")
                
                # NEW: Check what frame RMPFlow is actually controlling
                try:
                    if hasattr(self._cspace_controller, '_articulation_motion_policy'):
                        amp = self._cspace_controller._articulation_motion_policy
                        if hasattr(amp, '_motion_policy'):
                            rmpflow_policy = amp._motion_policy
                            if hasattr(rmpflow_policy, '_robot_description'):
                                carb.log_warn(f"  RMPFlow robot description found")
                            if hasattr(rmpflow_policy, 'end_effector_frame_name'):
                                carb.log_warn(f"  RMPFlow EE frame: {rmpflow_policy.end_effector_frame_name}")
                except Exception as e:
                    carb.log_warn(f"  Could not introspect RMPFlow: {e}")
                
                # Calculate position error magnitude
                error_mag = np.linalg.norm(position_target - ee_pos)
                carb.log_warn(f"  Position error magnitude: {error_mag*100:.1f}cm")
            
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
            # RMPFlow handles motion planning, so just command the target directly
            
            # Apply end effector offset if available, otherwise use default
            if self._end_effector_offset is not None:
                position_target = np.array(
                    [
                        push_end_position[0] + self._end_effector_offset[0],
                        push_end_position[1] + self._end_effector_offset[1],
                        push_end_position[2] + self._end_effector_offset[2],
                    ]
                )
            else:
                position_target = push_end_position.copy()
            
            # Log progress every 10% of phase completion
            progress_percent = int(self._t * 100)
            if progress_percent % 10 == 0 and progress_percent > 0:
                carb.log_info(f"  Push progress: {progress_percent}% - EE target: [{position_target[0]:.3f}, {position_target[1]:.3f}, {position_target[2]:.3f}]")
            
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
            target_joint_positions = ArticulationAction(
                joint_positions=self._turned_position
            )

        # Increment time and check if phase should advance
        self._t += self._events_dt[self._event]
        
        # Check if we should advance to next phase
        should_advance = False
        
        if self._t >= 1.0:
            # Time-based: phase duration exceeded
            if self._current_phase_target is not None:
                # For cartesian space phases, check if we reached the target
                if self._check_position_reached(self._current_phase_target):
                    should_advance = True
                    carb.log_info(f"✓ Phase {self._event} complete - Target reached!")
                else:
                    # Haven't reached target yet, give more time
                    ee_position, _ = self._robot_articulation.end_effector.get_world_pose()
                    distance = np.linalg.norm(ee_position - self._current_phase_target)
                    
                    # Only warn every 100 steps to reduce spam
                    if int(self._t * 1000) % 100 == 0:
                        carb.log_warn(f"⚠ Phase {self._event} time limit reached but target not reached yet (dist: {distance*100:.1f}cm, t={self._t:.2f})")
                    
                    # Allow extra time for the robot to reach target
                    if self._t < 2.0:  # Allow up to 2x the normal time
                        should_advance = False
                    else:
                        carb.log_error(f"⚠ Phase {self._event} forcing advance after timeout (t={self._t:.2f})")
                        should_advance = True
            else:
                # For joint space phases (overhead, gripper, return), use time only
                should_advance = True
        
        if should_advance:
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
