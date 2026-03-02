# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded Differential IK Action for workspace-constrained manipulation."""

import torch
from dataclasses import MISSING

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


@configclass
class BoundedDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for Bounded Differential IK Action.
    
    This action clips the target end-effector position to a user-defined 3D bounding box
    before solving for joint angles. This prevents:
    - Arms crashing through the table (Z > Z_min)
    - Arms overlapping (separate left/right workspace boxes)
    - Wasted exploration outside task-relevant areas
    """
    
    task_space_bounds: tuple[float, float, float, float, float, float] = MISSING
    """3D bounding box for the end-effector workspace: (x_min, y_min, z_min, x_max, y_max, z_max).
    
    The target position will be clamped to stay within this box before IK solving.
    Example: (-0.6, 0.2, 0.05, -0.05, 0.8, 0.4) restricts to left side of table.
    """


class BoundedDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Differential IK action that clamps end-effector targets to a bounding box.
    
    This action inherits from DifferentialInverseKinematicsAction but adds workspace
    bounds checking. After computing the target pose from the action deltas, the target
    position is clamped to stay within the configured 3D box before the IK solver
    computes joint commands.
    
    The rotation target is not bounded, allowing free orientation within the box.
    """
    
    cfg: BoundedDifferentialIKActionCfg
    
    def __init__(self, cfg: BoundedDifferentialIKActionCfg, env):
        """Initialize the bounded IK action.
        
        Args:
            cfg: Configuration for the bounded IK action.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # Convert bounds to tensors for efficient GPU clamping
        # Shape: (3,) for [x_min, y_min, z_min]
        self.bounds_min = torch.tensor(
            cfg.task_space_bounds[:3], 
            device=self.device, 
            dtype=torch.float32
        )
        # Shape: (3,) for [x_max, y_max, z_max]
        self.bounds_max = torch.tensor(
            cfg.task_space_bounds[3:], 
            device=self.device, 
            dtype=torch.float32
        )
    
    def process_actions(self, actions: torch.Tensor):
        """Process actions and clamp target positions to workspace bounds.
        
        Args:
            actions: Input actions from the policy. Shape: (num_envs, action_dim)
        """
        # 1. Standard differential IK processing
        # This scales actions, computes current EE pose, and calls
        # self._ik_controller.set_command() which populates:
        #   - self._ik_controller.ee_pos_des (desired position)
        #   - self._ik_controller.ee_quat_des (desired orientation)
        super().process_actions(actions)
        
        # 2. BOUNDED WORKSPACE FIX: Clamp the controller's desired position to the box
        # The IK controller stores the target in ee_pos_des (shape: num_envs x 3)
        # Clamp each dimension (X, Y, Z) independently to [bounds_min, bounds_max]
        self._ik_controller.ee_pos_des[:] = torch.clamp(
            self._ik_controller.ee_pos_des,
            min=self.bounds_min,
            max=self.bounds_max
        )
        
        # Note: We don't clamp rotation (ee_quat_des)
        # This allows the hand to rotate freely within the bounded workspace
