# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import Articulation


class RMPFlowController(mg.MotionPolicyController):
    """[summary]

    Args:
        name (str): [description]
        robot_articulation (Articulation): [description]
        physics_dt (float, optional): [description]. Defaults to 1.0/60.0.
        attach_gripper (bool, optional): [description]. Defaults to False.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
        usd_path: Optional[str] = None,
    ) -> None:
        import carb
        import os
        
        if attach_gripper:
            # CRITICAL FIX: Generate RMPFlow config from the actual USD to ensure kinematic match
            # The built-in Isaac Sim UR5e config doesn't match our custom USD with gripper
            carb.log_warn(f"\n{'='*70}")
            carb.log_warn(f"RMPFlow: Generating config from USD for kinematic accuracy")
            carb.log_warn(f"{'='*70}")
            
            # Get USD file path - try multiple sources
            robot_usd_path = None
            if usd_path and os.path.isfile(usd_path):
                robot_usd_path = usd_path
                carb.log_warn(f"Using provided USD path: {robot_usd_path}")
            else:
                # Try to infer from robot_articulation attributes
                if hasattr(robot_articulation, '_usd_path'):
                    robot_usd_path = robot_articulation._usd_path
                    carb.log_warn(f"Using robot._usd_path: {robot_usd_path}")
                else:
                    # Fallback: assume standard location
                    working_dir = os.path.dirname(os.path.realpath(__file__))
                    robot_usd_path = os.path.join(working_dir, "..", "tasks", "ur5e_handeye_gripper.usd")
                    robot_usd_path = os.path.normpath(robot_usd_path)
                    if os.path.isfile(robot_usd_path):
                        carb.log_warn(f"Using inferred USD path: {robot_usd_path}")
                    else:
                        robot_usd_path = None
                        carb.log_warn(f"Could not find USD file, will use standard config")
            
            if robot_usd_path and os.path.isfile(robot_usd_path):
                try:
                    # Try to import robot directly from USD using Lula
                    from isaacsim.robot_motion.motion_generation import lula
                    
                    robot_prim_path = robot_articulation.prim_path
                    carb.log_warn(f"Robot prim path: {robot_prim_path}")
                    carb.log_warn(f"Importing from USD file: {robot_usd_path}")
                    
                    # Import robot description from USD stage using prim path
                    robot_description = lula.import_robot_from_usd.create_robot_from_usd_stage(
                        prim_path=robot_prim_path
                    )
                    
                    # Use the standard UR5e RMPFlow config but with our robot description
                    self.rmp_flow_config = (
                        mg.interface_config_loader.load_supported_motion_policy_config(
                            "UR5e", "RMPflow"
                        )
                    )
                    
                    # Override to use flange (which exists in our USD)
                    self.rmp_flow_config["end_effector_frame_name"] = "flange"
                    carb.log_warn(f"✓ Successfully loaded robot description from USD stage")
                    carb.log_warn(f"  End effector frame: flange")
                    carb.log_warn(f"{'='*70}\n")
                    
                    # Create RMPFlow with USD-derived robot description
                    self.rmp_flow = mg.lula.motion_policies.RmpFlow(
                        robot_description=robot_description,
                        rmpflow_config_path=self.rmp_flow_config["rmpflow_config_path"],
                        end_effector_frame_name="flange",
                        maximum_substep_size=0.00334
                    )
                    
                except Exception as e:
                    carb.log_error(f"✗ Failed to create RMPFlow from USD: {e}")
                    carb.log_warn(f"  Falling back to standard UR5e config (may have kinematic mismatch)")
                    carb.log_warn(f"{'='*70}\n")
                    
                    # Fallback to standard config
                    self.rmp_flow_config = (
                        mg.interface_config_loader.load_supported_motion_policy_config(
                            "UR5e", "RMPflow"
                        )
                    )
                    self.rmp_flow_config["end_effector_frame_name"] = "flange"
                    self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
            else:
                carb.log_warn(f"✗ USD file not found, using standard UR5e config")
                carb.log_warn(f"  This may cause kinematic mismatch if USD has gripper/modifications")
                carb.log_warn(f"{'='*70}\n")
                
                # Use standard config
                self.rmp_flow_config = (
                    mg.interface_config_loader.load_supported_motion_policy_config(
                        "UR5e", "RMPflow"
                    )
                )
                self.rmp_flow_config["end_effector_frame_name"] = "flange"
                self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        else:
            self.rmp_flow_config = (
                mg.interface_config_loader.load_supported_motion_policy_config(
                    "UR10", "RMPflow"
                )
            )
            self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmp_flow, physics_dt
        )

        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp
        )
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
        
        # DIAGNOSTIC: Log RMPFlow configuration
        import carb
        carb.log_warn(f"\n{'='*70}")
        carb.log_warn(f"RMPFlow Controller Initialization")
        carb.log_warn(f"{'='*70}")
        carb.log_warn(f"Robot base position set to: [{self._default_position[0]:.3f}, {self._default_position[1]:.3f}, {self._default_position[2]:.3f}]")
        carb.log_warn(f"Robot base orientation: [{self._default_orientation[0]:.3f}, {self._default_orientation[1]:.3f}, {self._default_orientation[2]:.3f}, {self._default_orientation[3]:.3f}]")
        carb.log_warn(f"attach_gripper parameter: {attach_gripper}")
        if attach_gripper:
            carb.log_warn(f"Using UR5e RMPFlow config")
        else:
            carb.log_warn(f"Using UR10 RMPFlow config")
        carb.log_warn(f"{'='*70}\n")
        
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
