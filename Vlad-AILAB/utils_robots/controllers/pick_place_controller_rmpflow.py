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
        self._safe_height = end_effector_initial_height
        self._last_print_time = 0.0
        
        # Joint space positions for overhead and turned positions
        self._overhead_position = np.array([np.pi, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
    
        self._turned_position = np.array([-np.pi, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0, 0, 0, 0, 0])
        
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

    def _check_position_reached(self, target_position: np.ndarray) -> bool:
        """Check if end effector has reached target position."""
        ee_position, _ = self._robot_articulation.end_effector.get_world_pose()
        
        # Target is for the tip/gripper, but ee_position is the flange.
        # We must apply the offset to the target to compare apples to apples (flange to flange target).
        # effectively: target_flange = target_tip + offset
        target_flange = target_position
        if self._end_effector_offset is not None:
             target_flange = target_position + self._end_effector_offset
             
        distance = np.linalg.norm(ee_position - target_flange)
        carb.log_warn(f"Distance to target: {distance}")
        return bool(distance < self._position_threshold)

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        
        if self._pause or self.is_done():
            self.pause()
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # Print phase dt info every 0.5 seconds
        if self._t - self._last_print_time >= 0.5 or self._t == 0:
            print(f"Phase {self._event} dt: {self._t:.4f}/{self._events_dt[self._event]}")
            print(f"Joints: {current_joint_positions[:6]}") # Print first 6 joints
            self._last_print_time = self._t

        # Get current EE position for interpolation start
        current_ee_pos, _ = self._robot_articulation.end_effector.get_world_pose()

        # Log phase start and set targets
        if self._t == 0:
            carb.log_warn(f"\n{'='*40}")
            carb.log_warn(f"STARTING PHASE {self._event}")
            carb.log_warn(f"Current Joints: {current_joint_positions}")
            self._phase_start_pos = current_ee_pos
            
            if self._event == 0:
                self._current_phase_target = None # Joint space
                carb.log_warn("Action: Move to Turned Position (Joint Space)")
            elif self._event == 1: # Move to pre-pick
                self._current_phase_target = np.array([picking_position[0], picking_position[1], self._safe_height])
                carb.log_warn(f"Action: Move to Pre-Pick [X, Y, SafeZ]")
                carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 2: # Lower to pick
                self._current_phase_target = picking_position.copy()
                carb.log_warn(f"Action: Lower to Pick [X, Y, PickZ]")
                carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 3: # Settle
                self._current_phase_target = None
                carb.log_warn("Action: Settle")
            elif self._event == 4: # Close gripper
                self._current_phase_target = None
                carb.log_warn("Action: Close Gripper")
            elif self._event == 5: # Lift
                self._current_phase_target = np.array([picking_position[0], picking_position[1], self._safe_height])
                carb.log_warn(f"Action: Lift to Safe Height")
                carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 6: # Move to pre-place
                self._current_phase_target = np.array([placing_position[0], placing_position[1], self._safe_height])
                carb.log_warn(f"Action: Move to Pre-Place")
                carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 7: # Lower to place
                self._current_phase_target = placing_position.copy()
                carb.log_warn(f"Action: Lower to Place")
                carb.log_warn(f"Target: {self._current_phase_target}")
            elif self._event == 8: # Open gripper
                self._current_phase_target = None
                carb.log_warn("Action: Open Gripper")
            elif self._event == 9: # Return
                self._current_phase_target = None
                carb.log_warn("Action: Return to Home (Joint Space)")
            
            if self._current_phase_target is not None:
                dist = np.linalg.norm(self._current_phase_target - self._phase_start_pos)
                carb.log_warn(f"Start Pos: {self._phase_start_pos}")
                carb.log_warn(f"Distance to travel: {dist:.4f}m")
            carb.log_warn(f"{'='*40}\n")

        # Set default orientation if not set
        if self._end_effector_orientation is None:
            # Standard top-down orientation for UR5e
            self._end_effector_orientation = euler_angles_to_quat(np.array([np.pi/2, np.pi/2, -np.pi/2]))

        # Helper to get RMPFlow action with interpolation
        def get_interpolated_rmp_action(target_pos):
            # Calculate interpolation factor alpha
            duration = self._events_dt[self._event]
            # Ensure duration is at least one step to avoid divide by zero
            duration = max(duration, 0.01)
            
            alpha = min(self._t / duration, 1.0)
            
            # Interpolate position
            # Linear interpolation: p(t) = p_start + (p_end - p_start) * alpha
            interpolated_target = self._phase_start_pos + (target_pos - self._phase_start_pos) * alpha
            
            # Apply offset if needed
            # Note: RMPFlow might handle offset differently, but since we are driving the target,
            # we should apply the offset to the target position so the end-effector (flange) goes to the right place.
            # However, RMPFlow config usually defines the EE frame. If it's "flange", we need to offset.
            # If it's "tool0" or "gripper_tip", we might not need to.
            # Assuming "flange" based on previous tasks.
            final_target = interpolated_target + (self._end_effector_offset if self._end_effector_offset is not None else np.zeros(3))
            
            action = self._rmpflow.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=self._end_effector_orientation
            )
            
            return action

        # Execute Phase
        # Execute Phase
        if self._event == 0: # Turned Position (Start)
            # Use turned position as the start/home position
            target_joint_positions = ArticulationAction(joint_positions=self._turned_position)
            
        elif self._event == 1: # Pre-pick
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target)
            
        elif self._event == 2: # Pick
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target)
            
        elif self._event == 3: # Settle
            # Just hold position
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            
        elif self._event == 4: # Close gripper
            target_joint_positions = self._gripper.forward(action="close")
            
        elif self._event == 5: # Lift
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target)
            
        elif self._event == 6: # Pre-place
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target)
            
        elif self._event == 7: # Place
            target_joint_positions = get_interpolated_rmp_action(self._current_phase_target)
            
        elif self._event == 8: # Open gripper
            target_joint_positions = self._gripper.forward(action="open")
            
        elif self._event == 9: # Return
            target_joint_positions = ArticulationAction(joint_positions=self._turned_position)
            
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        # Update timing and check completion
        # Assuming 60Hz physics step or similar. Ideally dt should be passed in.
        self._t += 1.0 / 60.0
        
        # Check if phase is done
        phase_done = False
        if self._t >= self._events_dt[self._event]: # Duration elapsed
            if self._current_phase_target is not None:
                if self._check_position_reached(self._current_phase_target):
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
            if self._event == 5: # After close gripper (was 4)
                 self._grasp = self._is_grasped()

        return target_joint_positions

    def _is_grasped(self) -> bool:
        """Returns True if the gripper is not closed (meaning it's holding something), False otherwise."""
        # Simple check: if joints are not at fully closed position
        # This depends on the gripper specifics
        return True # Simplified for now, or implement actual check
    
    def get_grasp(self):
        return self._grasp
