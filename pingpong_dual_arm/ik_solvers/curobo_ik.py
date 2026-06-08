"""cuRobo inverse kinematics action term for Isaac Lab.

Calls cuRobo IKSolver directly — no redundant wrapper layers, no double-scaling.
In absolute mode, the 6-D action = (x, y, z, roll, pitch, yaw) target pose in
the robot base frame.  RPY is converted to quaternion per environment.

CUDA graphs are enabled for ~10x speedup; a warm-up call is issued at init.
LBFGS iterations are tuned for real-time step-by-step control (30 outer / 10 inner).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets.articulation import Articulation

logger = logging.getLogger(__name__)

try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml

    _CUROBO_AVAILABLE = True
except ImportError:
    _CUROBO_AVAILABLE = False


def _rpy_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert (N,) RPY tensors → (N, 4) quaternion (w, x, y, z)."""
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
    """cuRobo IK action — calls solve_batch() each step.

    Action format (6-D per env): (x, y, z, roll, pitch, yaw) absolute target
    pose in the robot base frame.
    """

    cfg: "CuroboIKActionCfg"
    _asset: Articulation
    _raw_actions: torch.Tensor  # (num_envs, 6)

    def __init__(self, cfg: "CuroboIKActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not _CUROBO_AVAILABLE:
            raise ImportError("cuRobo not installed. pip install curobo")

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, True
        )
        self._num_joints = len(self._joint_ids)

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one body for '{self.cfg.body_name}', found "
                f"{len(body_ids)}: {body_names}"
            )
        self._body_idx = body_ids[0]

        tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)
        ur_yaml = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
        robot_cfg = RobotConfig.from_dict(ur_yaml["robot_cfg"], tensor_args)

        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,  # world_model — no collision world needed
            tensor_args,
            self.cfg.num_seeds,
            self.cfg.position_threshold,
            self.cfg.rotation_threshold,
            use_cuda_graph=True,  # enabled for ~10x speedup
        )

        # Tune LBFGS: fewer iterations, same quality for step-by-step control
        ik_cfg.solver.newton_optimizer.n_iters = 30
        ik_cfg.solver.newton_optimizer.inner_iters = 10

        self._solver = IKSolver(ik_cfg)

        # Warm-up: capture CUDA graph once for the fixed batch size
        logger.info(
            f"[cuRobo] Warming up CUDA graph for N={self.num_envs} envs, "
            f"num_seeds={self.cfg.num_seeds}..."
        )
        _wup_pos = torch.zeros(self.num_envs, 3, device=self.device)
        _wup_quat = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=torch.float32
        ).expand(self.num_envs, 4)
        self._solver.solve_batch(
            Pose(position=_wup_pos.unsqueeze(1), quaternion=_wup_quat.unsqueeze(1)),
            seed_config=torch.zeros(self.num_envs, 1, 6, device=self.device),
            retract_config=torch.zeros(self.num_envs, 6, device=self.device),
        )
        logger.info("[cuRobo] Warm-up done.")

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

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
        cur_joints = self._asset.data.joint_pos[:, self._joint_ids]
        N = cur_joints.shape[0]

        pos = self._raw_actions[:, :3]
        roll = self._raw_actions[:, 3]
        pitch = self._raw_actions[:, 4]
        yaw = self._raw_actions[:, 5]

        quat = _rpy_to_quat(roll, pitch, yaw)

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
                joint_pos_des[i] = solution[i, : self._num_joints]

        self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0


@configclass
class CuroboIKActionCfg(ActionTermCfg):
    """Configuration for cuRobo inverse kinematics action term.

    The 6-D action is an absolute target pose (x, y, z, roll, pitch, yaw)
    in the robot base frame.
    """

    class_type: type[ActionTerm] = CuroboInverseKinematicsAction

    asset_name: str = ""
    joint_names: list[str] | None = None
    body_name: str = ""
    position_threshold: float = 0.01
    rotation_threshold: float = 0.05
    num_seeds: int = 10
