"""cuRobo inverse kinematics action term for Isaac Lab.

Calls cuRobo IKSolver directly with frame conversion from env-local to the
UR5e arm base frame (*_base_link_inertia).  The 6-D action is a relative delta
[dx,dy,dz,droll,dpitch,dyaw]; apply_actions() accumulates it onto the current
EE pose, converts to the arm base frame, and calls solve_batch().

CUDA graphs enabled for ~10x speedup; LBFGS tuned to 30/10 iterations.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets.articulation import Articulation

logger = logging.getLogger(__name__)

_CUROBO_AVAILABLE = False
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, load_yaml
    _CUROBO_AVAILABLE = True
except ImportError:
    pass


def _rpy_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor):
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qw, qx, qy, qz], dim=-1)


class CuroboInverseKinematicsAction(ActionTerm):
    """cuRobo IK action — calls solve_batch() each step with frame conversion.

    Action format (6-D per env): (dx, dy, dz, droll, dpitch, dyaw) relative delta.
    The delta is accumulated onto the current EE pose, then the absolute target
    is converted from env-local to the arm base frame before calling cuRobo.
    """

    cfg: CuroboIKActionCfg
    _asset: Articulation

    def __init__(self, cfg: CuroboIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not _CUROBO_AVAILABLE:
            raise ImportError("cuRobo not installed. pip install curobo")

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one body for '{self.cfg.body_name}', found {len(body_ids)}: {body_names}"
            )
        self._body_idx = body_ids[0]

        # Find arm base body for frame conversion
        _side = self._joint_names[0].split("_")[0]
        _arm_base_name = f"{_side}_base_link_inertia"
        try:
            _base_ids, __ = self._asset.find_bodies(_arm_base_name)
        except ValueError:
            _base_ids = []
        self._arm_base_idx: int | None = _base_ids[0] if len(_base_ids) > 0 else None

        tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)
        _custom_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ur5e_arm.yml")
        _content_dir = os.path.dirname(os.path.dirname(get_robot_configs_path()))
        _urdf_path = os.path.join(_content_dir, "assets", "robot", "ur_description", "ur5e.urdf")
        ur_yaml = load_yaml(_custom_yaml)
        ur_yaml["robot_cfg"]["kinematics"]["urdf_path"] = _urdf_path
        ur_yaml["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.dirname(_urdf_path)
        robot_cfg = RobotConfig.from_dict(ur_yaml["robot_cfg"], tensor_args)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_model=None,
            tensor_args=tensor_args,
            num_seeds=self.cfg.num_seeds,
            position_threshold=self.cfg.position_threshold,
            rotation_threshold=self.cfg.rotation_threshold,
            use_cuda_graph=True,
        )
        ik_cfg.solver.newton_optimizer.n_iters = 30
        ik_cfg.solver.newton_optimizer.inner_iters = 10
        self._solver = IKSolver(ik_cfg)

        # Warm-up: capture CUDA graph
        logger.info("[cuRobo] Warming up CUDA graph...")
        _wup_pos = torch.zeros(self.num_envs, 3, device=self.device)
        _wup_quat = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=torch.float32
        ).expand(self.num_envs, 4)
        self._solver.solve_batch(
            Pose(position=_wup_pos.unsqueeze(1), quaternion=_wup_quat.unsqueeze(1)),
            seed_config=torch.zeros(self.num_envs, 1, 6, device=self.device),
            retract_config=torch.zeros(self.num_envs, 6, device=self.device),
            use_nn_seed=False,
            num_seeds=1,
            newton_iters=30,
        )
        logger.info("[cuRobo] Warm-up done.")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        from isaaclab.utils.math import quat_apply, quat_mul

        cur_joints = self._asset.data.joint_pos[:, self._joint_ids]
        N = cur_joints.shape[0]

        # Relative delta from action
        delta_pos = self._raw_actions[:, :3]
        delta_roll = self._raw_actions[:, 3]
        delta_pitch = self._raw_actions[:, 4]
        delta_yaw = self._raw_actions[:, 5]

        # Current EE pose in world frame
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]

        # Target absolute pose in world frame = current + delta
        target_pos_w = ee_pos_w + delta_pos
        delta_quat = _rpy_to_quat(delta_roll, delta_pitch, delta_yaw)
        target_quat_w = quat_mul(delta_quat, ee_quat_w)

        if self._arm_base_idx is not None:
            base_pos_w = self._asset.data.body_pos_w[:, self._arm_base_idx]
            base_quat_w = self._asset.data.body_quat_w[:, self._arm_base_idx]
            base_quat_inv = base_quat_w.clone()
            base_quat_inv[:, 1:] *= -1.0

            rel = target_pos_w - base_pos_w
            pos = quat_apply(base_quat_inv, rel)
            quat = quat_mul(base_quat_inv, target_quat_w)
        else:
            pos = target_pos_w
            quat = target_quat_w

        goal_pose = Pose(
            position=pos.unsqueeze(1),
            quaternion=quat.unsqueeze(1),
        )

        result = self._solver.solve_batch(
            goal_pose,
            seed_config=cur_joints.unsqueeze(1),
            retract_config=cur_joints,
            use_nn_seed=False,
            num_seeds=1,
            newton_iters=30,
        )

        success = result.success.any(dim=1)
        solution = result.solution
        if solution.ndim == 3:
            solution = solution.squeeze(1)

        joint_pos_des = cur_joints.clone()
        for i in range(N):
            if success[i]:
                joint_pos_des[i] = solution[i, :self._num_joints]

        self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class CuroboIKActionCfg(ActionTermCfg):
    """Configuration for cuRobo inverse kinematics action term.

    The 6-D action is a relative delta (dx, dy, dz, droll, dpitch, dyaw).
    """

    class_type: type[ActionTerm] = CuroboInverseKinematicsAction

    asset_name: str = ""
    joint_names: list[str] | None = None
    body_name: str = ""
    position_threshold: float = 0.01
    rotation_threshold: float = 0.05
    num_seeds: int = 10
