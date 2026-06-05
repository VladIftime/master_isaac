"""cuRobo inverse kinematics action term for Isaac Lab.

Calls cuRobo IKSolver directly.  The 6-D action is a relative delta
[dx,dy,dz,droll,dpitch,dyaw]; apply_actions() accumulates it onto the
current EE pose to get an absolute target in world frame, then converts
to the arm base frame before calling solve_batch().

Frame conversion uses a hardcoded arm base pose computed from the robot's
known initial position (STAND_A_POS/STAND_B_POS) and the URDF transforms
through fixed joints (world_to_body_joint + {side}_base_joint +
{side}_base_link-base_link_inertia).  Verified against implementation.md:
net rotation from world to arm base is 180° Z.

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
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
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


# Arm base poses in world frame, computed from STAND positions + URDF fixed-joint
# chain (world_to_body_joint Rz(90°) → {side}_base_joint Rz(-90°) →
# {side}_base_link-base_link_inertia Rz(180°) → net Rz(180°) from world).
# Robot A: body at (0, -2.7, 0.6), init rot = identity
# Robot B: body at (0, +2.7, 0.6), init rot = Rz(180°) = quat(0,0,0,1)
#
# Right arm offset from body_base_link: pos=(0.05, -0.255, 0.03) rot=Rz(90°)
# Left  arm offset from body_base_link: pos=(0.05, +0.25,  0.03) rot=Rz(90°)
def _arm_base_pose(asset_name: str, side: str, device: str, dtype=torch.float32):
    """Return (pos, quat) tensors (shape (1,3), (1,4)) for the arm base."""
    import math
    # Root = body_base_link world pose
    if "robot_A" in asset_name:
        root_pos = (0.0, -2.7, 0.6)
        root_yaw_deg = 0.0
    else:
        root_pos = (0.0, 2.7, 0.6)
        root_yaw_deg = 180.0  # init_state.rot = (0,0,0,1)

    # world_to_body_joint adds 90° Z
    root_yaw_rad = math.radians(root_yaw_deg + 90.0)
    root_cos = math.cos(root_yaw_rad * 0.5)
    root_sin = math.sin(root_yaw_rad * 0.5)
    root_quat = (root_cos, 0.0, 0.0, root_sin)

    # offset from root to arm base (right_base_joint + base_link_inertia joints)
    # offset_pos in root frame, offset_rot = Rz(90°)
    offset_yaw = math.radians(90.0)
    off_cos = math.cos(offset_yaw * 0.5)
    off_sin = math.sin(offset_yaw * 0.5)
    if side == "left":
        off_pos = (0.05, 0.25, 0.03)
    else:
        off_pos = (0.05, -0.255, 0.03)

    # Compose: arm_base_world = root_world * offset
    # pos: root_pos + R_root_to_world * off_pos
    ry = math.radians(root_yaw_deg + 90.0)
    cr, sr = math.cos(ry), math.sin(ry)
    # Rz(root_yaw) * off_pos
    wx = cr * off_pos[0] - sr * off_pos[1]
    wy = sr * off_pos[0] + cr * off_pos[1]
    base_pos = (root_pos[0] + wx, root_pos[1] + wy, root_pos[2] + off_pos[2])

    # quat: root_quat * offset_quat
    rw, rz = root_cos, root_sin
    ow, oz = off_cos, off_sin
    bqw = rw * ow - rz * oz  # Z * Z: w1*w2 - z1*z2
    bqz = rw * oz + rz * ow  # Z * Z: w1*z2 + z1*w2
    # x, y = 0 for pure-Z rotations
    base_quat = (bqw, 0.0, 0.0, bqz)

    pos = torch.tensor(base_pos, device=device, dtype=dtype).unsqueeze(0)
    quat = torch.tensor(base_quat, device=device, dtype=dtype).unsqueeze(0)
    return pos, quat


class CuroboInverseKinematicsAction(ActionTerm):
    """cuRobo IK action — calls solve_batch() each step with frame conversion.

    Action format (6-D per env): (dx, dy, dz, droll, dpitch, dyaw) relative delta.
    The delta is accumulated onto the current EE pose, then the absolute world-frame
    target is converted to the arm base frame before calling cuRobo.
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

        _side = self._joint_names[0].split("_")[0]
        self._arm_base_pos, self._arm_base_quat = _arm_base_pose(
            self.cfg.asset_name, _side, self.device
        )
        # Expand to num_envs
        N = self.num_envs
        self._arm_base_pos = self._arm_base_pos.expand(N, 3).contiguous()
        self._arm_base_quat = self._arm_base_quat.expand(N, 4).contiguous()

        logger.info(
            "[cuRobo] arm_base (world) pos=(%.3f,%.3f,%.3f) quat=(%.3f,%.3f,%.3f,%.3f)",
            self._arm_base_pos[0, 0].item(),
            self._arm_base_pos[0, 1].item(),
            self._arm_base_pos[0, 2].item(),
            self._arm_base_quat[0, 0].item(),
            self._arm_base_quat[0, 1].item(),
            self._arm_base_quat[0, 2].item(),
            self._arm_base_quat[0, 3].item(),
        )

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
        _wup_pos = torch.zeros(self.num_envs, 3, device=self.device)
        _wup_quat = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=torch.float32
        ).expand(self.num_envs, 4)
        self._solver.solve_batch(
            Pose(position=_wup_pos.unsqueeze(1), quaternion=_wup_quat.unsqueeze(1)),
            seed_config=torch.zeros(self.num_envs, 1, 6, device=self.device),
            retract_config=torch.zeros(self.num_envs, 6, device=self.device),
        )

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

        # Convert target from world frame to arm base frame
        base_quat_inv = self._arm_base_quat.clone()
        base_quat_inv[:, 1:] *= -1.0
        rel = target_pos_w - self._arm_base_pos
        pos = quat_apply(base_quat_inv, rel)
        quat = quat_mul(base_quat_inv, target_quat_w)

        goal_pose = Pose(
            position=pos.unsqueeze(1),
            quaternion=quat.unsqueeze(1),
        )

        result = self._solver.solve_batch(
            goal_pose,
            seed_config=cur_joints.unsqueeze(1),
            retract_config=cur_joints,
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

        # --- periodic debug logging ---
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1
        if self._step_count <= 3 or self._step_count % 600 == 0:
            n_ok = success.sum().item()
            err_norm = torch.norm(joint_pos_des - cur_joints, dim=-1)
            logger.info(
                "[cuRobo] step=%d  IK_ok=%d/%d  max_joint_delta=%.4f rad  "
                "cur_joints[0]=%s",
                self._step_count,
                n_ok,
                N,
                err_norm.max().item(),
                [f"{v:.3f}" for v in cur_joints[0].tolist()],
            )

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
