from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import Articulation
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller import RMPFlowController
import numpy as np
import typing
from typing import Optional, List
import carb
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation

class UR5eRMPFlowController(mg.MotionPolicyController):
    """
    RMPFlow Controller specifically for UR5e.
    Overrides the default RMPFlowController which hardcodes UR10.
    """
    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:
        # Load UR5e config. 
        # Note: We ignore attach_gripper for config selection as we only saw one config for UR5e.
        # If a specific gripper config exists, it should be added here.
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR5e", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

class RMPFlowPickPlaceController(manipulators_controllers.PickPlaceController):
    """
    RMPFlow Pick and Place Controller with Interpolation
    
    Uses RMPFlowController to compute joint configurations, but feeds it 
    interpolated intermediate targets to ensure smooth, straight-line motion.
    
    Phases:
    0. Move to overhead position (joint space initialization)
    1. Move to turned position (joint space)
    2. Move horizontally above picking position (X,Y at safe height)
    3. Lower vertically to picking position (only Z changes)
    4. Wait for robot to settle
    5. Close gripper
    6. Lift vertically to safe height (only Z changes)
    7. Move horizontally to above placing position (only X,Y change)
    8. Lower vertically to placing position (only Z changes)
    9. Open gripper to release
    10. Return to turned position (joint space)
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
            events_dt: Time for each phase
            end_effector_offset: Offset for end effector positioning
            end_effector_initial_height: Safe height for end effector
        """
        # Phase timings
        if events_dt is None:
            # 10 phases
            events_dt=[3.0, 2.0, 2.0, 1.0, 1.0, 3.5, 3.5, 2.0, 2.0, 2.0]
        
        if end_effector_initial_height is None:
            end_effector_initial_height = 0.35
        
        self._events_dt = events_dt
        self._event = 0
        self._t = 0.0
        self._pause = False
        self._total_time = 0.0
        
        self.name = name
        self._gripper = gripper
        self._robot_articulation = robot_articulation
        self._end_effector_initial_height = end_effector_initial_height
        self._end_effector_offset = end_effector_offset
        
        # Initialize RMPFlow Controller
        # We use our custom UR5e controller to ensure correct kinematics
        self._rmpflow = UR5eRMPFlowController(
            name=name + "_rmpflow", 
            robot_articulation=robot_articulation, 
            attach_gripper=True
        )
        
        self._grasp = False
        self._position_threshold = 0.05 # 5cm tolerance
        self._end_effector_orientation = None
        self._current_phase_target = None
        self._phase_start_pos = None
        self._safe_height = 0.45
        self._last_print_time = 0.0
        
        # Joint space positions for overhead and turned positions
        self._overhead_position = np.array([np.pi, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
    
        self._turned_position = np.array([np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        
        carb.log_warn(f"RMPFlowPickPlaceController '{name}' initialized")
        return

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[List[float]] = None,
    ) -> None:
        if end_effector_initial_height is not None:
            self._end_effector_initial_height = end_effector_initial_height
            self._safe_height = end_effector_initial_height
        if events_dt is not None:
            self._events_dt = events_dt
        self._event = 0
        self._t = 0.0
        self._pause = False
        self._total_time = 0.0
        self._current_phase_target = None
        self._phase_start_pos = None
        self._grasp = False
        self._last_print_time = 0.0
        self._rmpflow.reset()

    def _check_position_reached(self, target_position: np.ndarray, apply_offset: bool = True) -> bool:
        """Check if end effector has reached target position."""
        ee_position, _ = self._robot_articulation.end_effector.get_world_pose()
        
        # Target is for the tip/gripper, but ee_position is the flange.
        # We must apply the offset to the target to compare apples to apples (flange to flange target).
        # effectively: target_flange = target_tip + offset
        target_flange = target_position
        if apply_offset and self._end_effector_offset is not None:
             target_flange = target_position + self._end_effector_offset
             
        distance = np.linalg.norm(ee_position - target_flange)
        # carb.log_warn(f"Distance to target: {distance}")
        return bool(distance < self._position_threshold)

    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        
        if self._pause or self.is_done():
            self.pause()
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # Print phase dt info every 0.5 seconds
        if self._t - self._last_print_time >= 0.5 or self._t == 0:
            # print(f"Phase {self._event} dt: {self._t:.4f}/{self._events_dt[self._event]}")
            # print(f"Joints: {current_joint_positions[:6]}") # Print first 6 joints
            self._last_print_time = self._t

        # Get current EE position for interpolation start
        current_ee_pos, _ = self._robot_articulation.end_effector.get_world_pose()

        # Log phase start and set targets
        if self._t == 0:
            # carb.log_warn(f"\n{'='*40}")
            # carb.log_warn(f"STARTING PHASE {self._event}")
            # carb.log_warn(f"Current Joints: {current_joint_positions}")
            self._phase_start_pos = current_ee_pos
            
            if self._event == 0:
                self._current_phase_target = None # Joint space
                # carb.log_warn("Action: Move to Turned Position (Joint Space)")
            elif self._event == 1: # Move to pre-pick
                self._current_phase_target = np.array([picking_position[0], picking_position[1], self._safe_height])
                # carb.log_warn(f"Action: Move to Pre-Pick [X, Y, SafeZ]")
                # carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 2: # Align
                self._current_phase_target = np.array([picking_position[0], picking_position[1], self._safe_height])
                # carb.log_warn(f"Action: Align Orientation [Yaw]")
                # carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 3: # Lower to pick
                self._current_phase_target = picking_position.copy()
                # carb.log_warn(f"Action: Lower to Pick [X, Y, PickZ]")
                # carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 4: # Settle
                self._current_phase_target = None
                # carb.log_warn("Action: Settle")
            elif self._event == 5: # Close gripper
                self._current_phase_target = None
                # carb.log_warn("Action: Close Gripper")
            elif self._event == 6: # Lift
                self._current_phase_target = np.array([picking_position[0], picking_position[1], self._safe_height])
                # carb.log_warn(f"Action: Lift to Safe Height")
                # carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 7: # Grasp Check
                self._current_phase_target = None
                # carb.log_warn("Action: Grasp Check (Re-close)")
            elif self._event == 8: # Turned Intermediate
                self._current_phase_target = np.array([0.0, 0.5, 0.6])
                # carb.log_warn("Action: Move to Turned Position (Cartesian)")
                # carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 9: # Lower to Drop
                self._current_phase_target = self._phase_start_pos - np.array([0, 0, 0.20])
                # carb.log_warn(f"Action: Lower 20cm to Drop")
                # carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 10: # Open gripper
                self._current_phase_target = None
                # carb.log_warn("Action: Open Gripper")
            elif self._event == 11: # Return
                self._current_phase_target = None
                # carb.log_warn("Action: Return to Overhead Position")
            
            if self._current_phase_target is not None:
                dist = np.linalg.norm(self._current_phase_target - self._phase_start_pos)
                # carb.log_warn(f"Start Pos: {self._phase_start_pos}")
                # carb.log_warn(f"Distance to travel: {dist:.4f}m")
            # carb.log_warn(f"{'='*40}\n")

        # Update internal orientation if provided
        if end_effector_orientation is not None:
            self._end_effector_orientation = end_effector_orientation

        # Set default orientation if not set
        if self._end_effector_orientation is None:
            # Standard top-down orientation for UR5e
            self._end_effector_orientation = euler_angles_to_quat(np.array([np.pi/2, np.pi/2, -np.pi/2]))

        # Helper to get RMPFlow action with interpolation
        def get_interpolated_rmp_action(target_pos, target_ori=None, apply_offset=True):
            # Calculate interpolation factor alpha
            duration = self._events_dt[self._event]
            # Ensure duration is at least one step to avoid divide by zero
            duration = max(duration, 0.01)
            
            alpha = min(self._t / duration, 1.0)
            
            # Apply offset to target_pos if requested to get the target flange position
            target_flange = target_pos + (self._end_effector_offset if apply_offset and self._end_effector_offset is not None else np.zeros(3))
            
            # Interpolate from start (flange) to target (flange)
            # Linear interpolation: p(t) = p_start + (p_end - p_start) * alpha
            final_target = self._phase_start_pos + (target_flange - self._phase_start_pos) * alpha
            
            # Use provided orientation or internal state
            ori = target_ori if target_ori is not None else self._end_effector_orientation
            
            action = self._rmpflow.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=ori
            )
            
            return action

        # Default Down Orientation (for Phase 1)
        # [pi, 0, pi] -> [0, 1, 0, 0] (w, x, y, z) roughly?
        # Using the same as main script default
        default_down_ori = euler_angles_to_quat(np.array([np.pi, 0, np.pi]))

        # Execute Phase
        if self._event == 0: # Turned Position (Start)
            # Use turned position as the start/home position
            target_joint_positions = ArticulationAction(joint_positions=self._overhead_position)
            
        elif self._event == 1: # Pre-pick (Move to position, Default Orientation)
            # Use default down orientation (Yaw=0)
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target, target_ori=default_down_ori)
            
        elif self._event == 2: # Align (Rotate to Target Yaw)
            # Stay at Pre-Pick Position, Rotate to Target Orientation
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target, target_ori=self._end_effector_orientation)
            
        elif self._event == 3: # Pick (Lower)
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target)
            
        elif self._event == 4: # Settle
            # Just hold position
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            
        elif self._event == 5: # Close gripper
            target_joint_positions = self._gripper.forward(action="close")
            
        elif self._event == 6: # Lift
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target)

        elif self._event == 7: # Grasp Check
            # Try to close gripper again to ensure grasp
            target_joint_positions = self._gripper.forward(action="close")
            
            # Check if grasped at the end of the phase (handled in phase_done logic typically, 
            # but we can also check here for logging)
            if self._t >= self._events_dt[self._event] - 0.1: # Near end of phase
                 if self._is_grasped(current_joint_positions):
                     if self._t > self._events_dt[self._event] - 0.05: # Log once
                        carb.log_warn("GRASP SUCCESSFUL")
                 else:
                     if self._t > self._events_dt[self._event] - 0.05:
                        carb.log_warn("GRASP FAILED")

        elif self._event == 8: # Turned Position (Intermediate)
            # Move to turned position in Cartesian space to preserve orientation
            # Target: [0.0, 0.5, 0.6] (Safe side position)
            turned_target = np.array([0.0, 0.5, 0.6])
            target_joint_positions = get_interpolated_rmp_action(turned_target, apply_offset=False)
            
        elif self._event == 9: # Lower to Drop (20cm down from Turned)
            # Target is 20cm below the start of this phase
            drop_target = self._phase_start_pos - np.array([0, 0, 0.20])
            target_joint_positions = get_interpolated_rmp_action(drop_target, apply_offset=False)
            
        elif self._event == 10: # Open gripper
            target_joint_positions = self._gripper.forward(action="open")
            
        elif self._event == 11: # Return to Overhead
            target_joint_positions = ArticulationAction(joint_positions=self._overhead_position)
            
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # Update timing and check completion
        # Assuming 60Hz physics step or similar. Ideally dt should be passed in.
        dt = 1.0 / 60.0
        self._t += dt
        self._total_time += dt
        
        # Check if phase is done
        phase_done = False
        if self._t >= self._events_dt[self._event]: # Duration elapsed
            if self._current_phase_target is not None:
                # For Cartesian phases (7 and 8), target is already flange, so don't apply offset
                apply_offset = True
                if self._event in [8, 9]:
                    apply_offset = False
                    
                if self._check_position_reached(self._current_phase_target, apply_offset=apply_offset):
                    phase_done = True
                elif self._t > self._events_dt[self._event] + 8.0: # Timeout (8s extra)
                    current_dist = np.linalg.norm(current_ee_pos - self._current_phase_target)
                    carb.log_warn(f"Phase {self._event} timeout. Distance to target: {current_dist:.4f}")
                    carb.log_warn(f"Current Pos: {current_ee_pos}")
                    carb.log_warn(f"Target Pos: {self._current_phase_target}")
                    phase_done = True
            else:
                # Time-based phase
                phase_done = True
        
        if phase_done:
            self._event += 1
            self._t = 0
            self._last_print_time = 0
            if self._event == 6: # After close gripper (Phase 5 done, now in Phase 6)
                 self._grasp = self._is_grasped(current_joint_positions)
            elif self._event == 8: # After Grasp Check (Phase 7 done, now in Phase 8)
                 self._grasp = self._is_grasped(current_joint_positions)

        self._clamp_action_joints(target_joint_positions)
        self._check_action_safety(target_joint_positions)
        return target_joint_positions

    def _clamp_action_joints(self, action: ArticulationAction) -> None:
        """
        Clamps joint positions to [-2pi, 2pi] to avoid PhysX errors.
        Handles cases where some joints are None.
        """
        if action.joint_positions is None:
            return

        # Convert to list to handle mixed types (float and None)
        joints = list(action.joint_positions)
        
        # Clamp to slightly less than 2pi to be safe
        limit = 2.0 * np.pi - 0.01
        
        modified = False
        for i, val in enumerate(joints):
            if val is not None:
                # Check if value is finite
                if np.isfinite(val):
                    # Clamp
                    if val > limit:
                        joints[i] = limit
                        modified = True
                    elif val < -limit:
                        joints[i] = -limit
                        modified = True
        
        # Update action if modified
        if modified:
            action.joint_positions = joints

    def _check_action_safety(self, action: ArticulationAction) -> None:
        """
        Checks if the generated action is safe for the robot.
        Raises RuntimeError if safety violation is detected.
        """
        if action.joint_positions is None:
            return

        # Handle list of Nones or mixed types which result in object array
        if any(x is None for x in action.joint_positions):
             return

        joint_positions = np.array(action.joint_positions)
        
        # Check for NaNs or Infs
        if not np.all(np.isfinite(joint_positions)):
            carb.log_error(f"Safety Violation: NaN or Inf detected in joint positions: {joint_positions}")
            raise RuntimeError("Controller Safety Violation: NaN or Inf detected in joint positions.")

        # Check joint limits (Hardcoded for UR5e as fallback)
        # Limits: +/- 2*pi for most joints, but let's be generous and say +/- 2.5*pi to catch wild spins
        # Real limits are usually +/- 360 deg (2pi) for UR5e joints.
        limit = 2.5 * np.pi
        if np.any(np.abs(joint_positions) > limit):
             # Filter out None values which might be present if we are not controlling all joints
             # But ArticulationAction usually expects all joints or specific indices. 
             # Here we assume it matches the robot dof if it's a full array.
             # If it contains Nones, np.array might be object type.
             pass 
        
        # If we have access to robot limits, we should use them.
        # For now, let's just check for extreme values that indicate explosion.
        if np.any(np.abs(joint_positions) > 100.0): # Arbitrary large number
             carb.log_error(f"Safety Violation: Extreme joint positions detected: {joint_positions}")
             raise RuntimeError(f"Controller Safety Violation: Extreme joint positions detected: {joint_positions}")

    def _is_grasped(self, current_joint_positions: np.ndarray) -> bool:
        """Returns True if the gripper is holding something (not fully closed)."""
        # Gripper joints are the last 6 joints (indices 6-11)
        # Observed fully closed values from logs: ~[-0.782, 0.781]
        # This corresponds roughly to +/- 45 degrees (0.785 rad)
        
        gripper_joints = current_joint_positions[6:8] # Check first 2 gripper joints
        
        # Updated closed position based on simulation logs
        # Index 6 is negative, Index 7 is positive
        closed_pos = np.array([-0.785, 0.785])
        
        # Check difference
        diff = np.abs(gripper_joints - closed_pos)
        
        # If any joint is significantly far from closed (e.g. > 0.05 rad), we are holding something
        # If we are holding something, the gripper stops EARLY, so the value is smaller in magnitude.
        # e.g. if holding, joints might be [-0.5, 0.5].
        # diff = abs(-0.5 - (-0.785)) = 0.285 > 0.05 -> True.
        # If empty, joints are [-0.785, 0.785].
        # diff = 0 < 0.05 -> False.
        
        is_holding = np.any(diff > 0.05)
        
        # Debug print
        # carb.log_warn(f"Grasp Check: Joints={gripper_joints}, Closed={closed_pos}, Diff={diff}, Holding={is_holding}")
        print(f"DEBUG: Grasp Check: Joints={gripper_joints}, Diff={diff}, Holding={is_holding}")
        
        return is_holding
    
    def get_grasp(self):
        return self._grasp

    def get_total_time(self):
        return self._total_time
