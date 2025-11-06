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
        end_effector_offset: Optional[np.ndarray] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [0.01, 0.01, 1, 0.01, 0.1, 0.01, 0.005, 1, 0.01, 0.1]
        
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
            end_effector_initial_height=0.55,
            events_dt=events_dt,
        )
        self._grasp = False
        self._current_target_x = None
        self._current_target_y = None
        self._end_effector_offset = end_effector_offset
        self._robot_articulation = robot_articulation
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
        # DEBUG: Log input parameters and current end effector position
        carb.log_warn(f"\n{'='*70}")
        carb.log_warn(f"PICK_PLACE_CONTROLLER - Event {self._event}, Time {self._t:.4f}")
        carb.log_warn(f"{'='*70}")
        carb.log_warn(f"Input picking_position: [{picking_position[0]:.3f}, {picking_position[1]:.3f}, {picking_position[2]:.3f}]")
        carb.log_warn(f"Input placing_position:  [{placing_position[0]:.3f}, {placing_position[1]:.3f}, {placing_position[2]:.3f}]")
        carb.log_warn(f"Current joint positions: {current_joint_positions}")
        carb.log_warn(f"Current joint positions shape: {current_joint_positions.shape}")
        
        # Get current end effector position for comparison
        try:
            ee_pos, ee_ori = self._robot_articulation.end_effector.get_world_pose()
            carb.log_warn(f"Current end effector world position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        except Exception as e:
            carb.log_warn(f"Could not get end effector position: {e}")
        
        # Use stored offset or parameter (like push controller)
        if end_effector_offset is None:
            end_effector_offset = self._end_effector_offset if self._end_effector_offset is not None else np.array([0, 0, 0])
        carb.log_warn(f"End effector offset: [{end_effector_offset[0]:.3f}, {end_effector_offset[1]:.3f}, {end_effector_offset[2]:.3f}]")
        
        # Use same orientation as push controller which works correctly
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([np.pi/2, np.pi/2, -np.pi/2]))
        carb.log_warn(f"End effector orientation: {end_effector_orientation}")
        
        if self._pause or self.is_done():
            carb.log_warn("Controller paused or done, returning None positions")
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        # Calculate position target with offset (like push controller)
        def get_target(base_pos):
            result = base_pos + (end_effector_offset if end_effector_offset is not None else np.zeros(3))
            carb.log_warn(f"  get_target: base=[{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}] -> result=[{result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f}]")
            return result
        
        if self._event == 0:
            # Phase 0: Move above picking position (lift height above object)
            lift_height = 0.15  # 15cm above object, similar to push controller
            target_pos = np.array([picking_position[0], picking_position[1], picking_position[2] + lift_height])
            carb.log_warn(f"PHASE 0: Move above picking position")
            carb.log_warn(f"  Target position (before offset): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            final_target = get_target(target_pos)
            carb.log_warn(f"  Calling cspace_controller with position: [{final_target[0]:.3f}, {final_target[1]:.3f}, {final_target[2]:.3f}]")
            if end_effector_orientation is not None:
                carb.log_warn(f"  Orientation quaternion: [{end_effector_orientation[0]:.3f}, {end_effector_orientation[1]:.3f}, {end_effector_orientation[2]:.3f}, {end_effector_orientation[3]:.3f}]")
            else:
                carb.log_warn(f"  Orientation quaternion: None")
            try:
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=final_target,
                    target_end_effector_orientation=end_effector_orientation,
                )
                if target_joint_positions.joint_positions is not None:
                    carb.log_warn(f"  Returned joint positions: {target_joint_positions.joint_positions}")
                    carb.log_warn(f"  Joint positions shape: {target_joint_positions.joint_positions.shape}")
                    # Check if any joint positions are None (shouldn't happen for cspace controller)
                    none_joints = [i for i, pos in enumerate(target_joint_positions.joint_positions) if pos is None]
                    if none_joints:
                        carb.log_error(f"  ERROR: Some joint positions are None at indices: {none_joints}")
                else:
                    carb.log_error(f"  ERROR: Returned joint positions is None!")
            except Exception as e:
                carb.log_error(f"  EXCEPTION in cspace_controller.forward: {e}")
                import traceback
                carb.log_error(traceback.format_exc())
                # Return safe default
                target_joint_positions = ArticulationAction(
                    joint_positions=[None] * current_joint_positions.shape[0]
                )
        elif self._event == 1:
            # Phase 1: Lower to picking position
            a = self._mix_sin(max(0, self._t))
            lift_height = 0.15
            start_height = picking_position[2] + lift_height
            target_height = self._combine_convex(start_height, picking_position[2], a)
            target_pos = np.array([picking_position[0], picking_position[1], target_height])
            carb.log_warn(f"PHASE 1: Lower to picking position (alpha={a:.3f})")
            carb.log_warn(f"  Start height: {start_height:.3f}, Target height: {target_height:.3f}, Picking Z: {picking_position[2]:.3f}")
            carb.log_warn(f"  Target position (before offset): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            final_target = get_target(target_pos)
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Returned joint positions: {target_joint_positions.joint_positions}")
            else:
                carb.log_warn(f"  WARNING: Returned joint positions is None!")
        elif self._event == 2:
            # Phase 2: Wait for robot to settle
            carb.log_warn(f"PHASE 2: Wait for robot to settle (returning None positions)")
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
            carb.log_warn(f"  Returned {len(target_joint_positions.joint_positions)} None positions")
        elif self._event == 3:
            # Phase 3: Close gripper
            carb.log_warn(f"PHASE 3: Close gripper")
            target_joint_positions = self._gripper.forward(action="close")
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Gripper joint positions: {target_joint_positions.joint_positions}")
                carb.log_warn(f"  Joint positions shape: {target_joint_positions.joint_positions.shape}")
            else:
                carb.log_warn(f"  WARNING: Gripper returned None positions!")
        elif self._event == 4:
            # Phase 4: Lift up while keeping grip (like push controller phase 4)
            lift_height = 0.15
            target_pos = np.array([picking_position[0], picking_position[1], picking_position[2] + lift_height])
            carb.log_warn(f"PHASE 4: Lift up while keeping grip")
            carb.log_warn(f"  Target position (before offset): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            final_target = get_target(target_pos)
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Returned joint positions: {target_joint_positions.joint_positions}")
            else:
                carb.log_warn(f"  WARNING: Returned joint positions is None!")
        elif self._event == 5:
            # Phase 5: Move horizontally toward placing position (keep height constant)
            lift_height = 0.15
            a = self._mix_sin(self._t)
            target_x = (1 - a) * picking_position[0] + a * placing_position[0]
            target_y = (1 - a) * picking_position[1] + a * placing_position[1]
            # Use max height to ensure we're above both positions
            target_height = max(picking_position[2], placing_position[2]) + lift_height
            target_pos = np.array([target_x, target_y, target_height])
            carb.log_warn(f"PHASE 5: Move horizontally (alpha={a:.3f})")
            carb.log_warn(f"  Interpolated XY: [{target_x:.3f}, {target_y:.3f}], Height: {target_height:.3f}")
            carb.log_warn(f"  Target position (before offset): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            final_target = get_target(target_pos)
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Returned joint positions: {target_joint_positions.joint_positions}")
            else:
                carb.log_warn(f"  WARNING: Returned joint positions is None!")
        elif self._event == 6:
            # Phase 6: Lower to placing height
            lift_height = 0.15
            a = self._mix_sin(self._t)
            # Start from the lift height (we're at placing position xy from phase 5)
            start_height = max(picking_position[2], placing_position[2]) + lift_height
            target_height = self._combine_convex(start_height, placing_position[2], a)
            target_pos = np.array([placing_position[0], placing_position[1], target_height])
            carb.log_warn(f"PHASE 6: Lower to placing height (alpha={a:.3f})")
            carb.log_warn(f"  Start height: {start_height:.3f}, Target height: {target_height:.3f}, Placing Z: {placing_position[2]:.3f}")
            carb.log_warn(f"  Target position (before offset): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            final_target = get_target(target_pos)
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Returned joint positions: {target_joint_positions.joint_positions}")
            else:
                carb.log_warn(f"  WARNING: Returned joint positions is None!")
        elif self._event == 7:
            # Phase 7: Open gripper
            carb.log_warn(f"PHASE 7: Open gripper")
            target_joint_positions = self._gripper.forward(action="open")
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Gripper joint positions: {target_joint_positions.joint_positions}")
            else:
                carb.log_warn(f"  WARNING: Gripper returned None positions!")
        elif self._event == 8:
            # Phase 8: Lift up from placing position
            lift_height = 0.15
            target_pos = np.array([placing_position[0], placing_position[1], placing_position[2] + lift_height])
            carb.log_warn(f"PHASE 8: Lift up from placing position")
            carb.log_warn(f"  Target position (before offset): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            final_target = get_target(target_pos)
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Returned joint positions: {target_joint_positions.joint_positions}")
            else:
                carb.log_warn(f"  WARNING: Returned joint positions is None!")
        elif self._event == 9:
            # Phase 9: Return to picking position xy (at height)
            lift_height = 0.15
            a = self._mix_sin(self._t)
            target_x = (1 - a) * placing_position[0] + a * picking_position[0]
            target_y = (1 - a) * placing_position[1] + a * picking_position[1]
            target_pos = np.array([target_x, target_y, picking_position[2] + lift_height])
            carb.log_warn(f"PHASE 9: Return to picking position (alpha={a:.3f})")
            carb.log_warn(f"  Interpolated XY: [{target_x:.3f}, {target_y:.3f}], Height: {picking_position[2] + lift_height:.3f}")
            carb.log_warn(f"  Target position (before offset): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            final_target = get_target(target_pos)
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=final_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            if target_joint_positions.joint_positions is not None:
                carb.log_warn(f"  Returned joint positions: {target_joint_positions.joint_positions}")
            else:
                carb.log_warn(f"  WARNING: Returned joint positions is None!")
        else:
            carb.log_warn(f"UNKNOWN EVENT: {self._event}, returning None positions")
            target_joint_positions = ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )
        
        # DEBUG: Log before time update
        old_event = self._event
        old_t = self._t
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
            carb.log_warn(f"  EVENT ADVANCED: {old_event} -> {self._event} (t: {old_t:.4f} -> {self._t:.4f})")
            if self._event == 6:
                self._grasp = self._is_grasped()
                carb.log_warn(f"  Grasp status checked: {self._grasp}")
        else:
            carb.log_warn(f"  Time updated: t = {self._t:.4f} (dt = {self._events_dt[self._event]:.4f})")
        
        # DEBUG: Final check on return value
        if target_joint_positions is None:
            carb.log_error(f"  ERROR: target_joint_positions is None!")
        elif hasattr(target_joint_positions, 'joint_positions'):
            if target_joint_positions.joint_positions is None:
                carb.log_warn(f"  WARNING: target_joint_positions.joint_positions is None")
            else:
                # Check for invalid joint values
                joint_pos = target_joint_positions.joint_positions
                if isinstance(joint_pos, (list, np.ndarray)):
                    invalid_joints = [i for i, pos in enumerate(joint_pos) if pos is not None and (np.isnan(pos) or np.isinf(pos))]
                    if invalid_joints:
                        carb.log_error(f"  ERROR: Invalid joint values (NaN/Inf) at indices: {invalid_joints}")
                    # Check for extreme values
                    extreme_joints = [i for i, pos in enumerate(joint_pos) if pos is not None and abs(pos) > 10.0]
                    if extreme_joints:
                        carb.log_warn(f"  WARNING: Extreme joint values (>10 rad) at indices: {extreme_joints}")
        
        carb.log_warn(f"{'='*70}\n")
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
        self._current_target_x = None
        self._current_target_y = None
        self._grasp = False
        return

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
