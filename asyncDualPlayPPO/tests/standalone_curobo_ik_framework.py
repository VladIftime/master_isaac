"""
Standalone CuRobo IK Framework Example
======================================
Demonstrates integrating CuRobo as a drop-in KinematicsSolver replacement for
the built-in LulaKinematicsSolver, using the clean World/Task paradigm.

The interactive red sphere target can be dragged with the viewport gizmo.
CuRobo GPU-accelerated IK drives the arm so the Robotiq gripper reaches the sphere.

Key fixes vs naive integration:
  - ArticulationAction(joint_indices=...) targets only arm joints 0-5 (robot has 12 total)
  - js_solution.position[0,0] extracts the true 1D solution (shape was (1,1,dof) not (dof,))
  - TCP offset from CuRobo tool0 (wrist_3_link) to actual EE (robotiq base) is computed via
    CuRobo FK and updated each step so the sphere stays between the gripper fingers
    as the wrist rotates.
"""

import sys
import torch
import numpy as np
import argparse

# CuRobo MUST be imported before AppLauncher to prevent library conflicts.
try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
except ModuleNotFoundError:
    print("\n[ERROR] CuRobo not found. Make sure you are using the .master_venv python.")
    sys.exit(1)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="CuRobo IK Standalone Example")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import isaacsim.core.api.tasks as tasks
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualSphere
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _quat_rotate(v: torch.Tensor, q_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (local-to-world: v_world = q * v * q⁻¹)."""
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    u = torch.stack([x, y, z])
    return v + 2.0 * w * torch.cross(u, v) + 2.0 * torch.cross(u, torch.cross(u, v))


def _quat_inv_rotate(v: torch.Tensor, q_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by q⁻¹ (world-to-local: v_local = q⁻¹ * v * q)."""
    q_conj = torch.stack([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]])
    return _quat_rotate(v, q_conj)


# ── FollowTarget task with sphere target ───────────────────────────────────────

class FollowTarget(tasks.FollowTarget):
    """FollowTarget task that uses a VisualSphere instead of the default VisualCuboid."""

    def __init__(self, name: str = "ur10e_follow_target", target_position=None, **kwargs):
        super().__init__(name=name, target_position=target_position, **kwargs)

    def set_robot(self) -> SingleManipulator:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Could not find Isaac Sim assets folder")
        asset_path = (
            assets_root_path
            + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
        )
        add_reference_to_stage(usd_path=asset_path, prim_path="/ur")
        gripper = ParallelGripper(
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0]),
            joint_closed_positions=np.array([40]),
            action_deltas=np.array([-40]),
            use_mimic_joints=True,
        )
        return SingleManipulator(
            prim_path="/ur",
            name="ur10_robot",
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            gripper=gripper,
        )

    def set_params(
        self,
        target_prim_path=None,
        target_name=None,
        target_position=None,
        target_orientation=None,
    ):
        """Override to spawn a red VisualSphere instead of the default VisualCuboid."""
        if target_prim_path is not None:
            if self._target is not None:
                del self._task_objects[self._target.name]
            if is_prim_path_valid(target_prim_path):
                self._target = self.scene.add(
                    SingleXFormPrim(
                        prim_path=target_prim_path,
                        position=target_position,
                        orientation=target_orientation,
                        name=target_name,
                    )
                )
            else:
                self._target = self.scene.add(
                    VisualSphere(
                        prim_path=target_prim_path,
                        name=target_name,
                        position=target_position,
                        orientation=target_orientation,
                        color=np.array([1.0, 0.0, 0.0]),
                        radius=0.04,
                    )
                )
            self._task_objects[self._target.name] = self._target
            self._target_visual_material = self._target.get_applied_visual_material()
            if self._target_visual_material is not None and hasattr(
                self._target_visual_material, "set_color"
            ):
                self._target_visual_material.set_color(np.array([1.0, 0.0, 0.0]))
        else:
            self._target.set_local_pose(
                position=target_position, orientation=target_orientation
            )


# ── CuRobo kinematic solver ────────────────────────────────────────────────────

class CuRoboKinematicsSolver:
    """
    CuRobo IKSolver wrapped as a kinematic solver (ArticulationKinematicsSolver interface).

    TCP offset compensation
    -----------------------
    CuRobo's ur10e.yml ee_link="tool0" sits at wrist_3_link (zero XYZ offset in CuRobo's
    URDF). The actual Robotiq gripper base (robotiq_arg2f_base_link) is 150–200 mm further
    along the tool axis.  Without compensation, CuRobo places the wrist AT the sphere and
    the gripper overshoots it.

    We measure the offset from tool0 to the actual EE once at reset, store it in tool0's
    local frame, then re-rotate it to world frame each step using CuRobo FK.  This keeps
    the sphere between the gripper fingers as the wrist rotates.

    Call setup_tcp_offset() after the first world.reset() to initialise.
    """

    MAX_JOINT_DELTA = 0.3  # rad/step; rejects arm flips

    def __init__(
        self,
        robot_articulation: SingleManipulator,
        robot_yaml_name: str = "ur10e.yml",
        device: str = "cuda:0",
    ):
        self._robot = robot_articulation
        self.device = torch.device(device)
        tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)
        ur_yaml = load_yaml(join_path(get_robot_configs_path(), robot_yaml_name))
        robot_cfg = RobotConfig.from_dict(ur_yaml["robot_cfg"], tensor_args)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg, world_model=None, tensor_args=tensor_args
        )
        self._ik_solver = IKSolver(ik_cfg)
        self._last_succ_joints: torch.Tensor | None = None
        self._tcp_offset_local: torch.Tensor | None = None  # offset in tool0 frame

    # Extra reach from robotiq_arg2f_base_link to grasp centre, along tool Z axis.
    # Applied in tool0 LOCAL frame so it stays correct as the wrist rotates.
    FINGER_REACH = 0.15  # metres

    def setup_tcp_offset(self, ee_pos_world: np.ndarray) -> None:
        """Compute and cache the tool0→grasp-centre offset in tool0's local frame.

        Call once after world.reset() + a few sim steps so transforms are settled.

        Args:
            ee_pos_world: world position of robotiq_arg2f_base_link.
        """
        cur_joints = torch.tensor(
            self._robot.get_joint_positions()[:6], device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 6)
        fk = self._ik_solver.fk(cur_joints)
        tool0_pos = fk.ee_position[0]       # (3,) in robot-base frame (= world, robot at origin)
        tool0_quat = fk.ee_quaternion[0]    # (4,) (w,x,y,z)

        ee_pos = torch.tensor(ee_pos_world, device=self.device, dtype=torch.float32)
        # Offset from tool0 to robotiq_arg2f_base_link, expressed in tool0 local frame.
        offset_local = _quat_inv_rotate(ee_pos - tool0_pos, tool0_quat)
        # Extend along tool Z (positive = toward fingertips) to reach grasp centre.
        offset_local[2] += self.FINGER_REACH
        self._tcp_offset_local = offset_local
        world_check = _quat_rotate(offset_local, tool0_quat) + tool0_pos
        print(f"  TCP offset (local): {offset_local.cpu().numpy().round(4)}")
        print(f"  Grasp centre (world at reset): {world_check.cpu().numpy().round(4)}")

    def _current_tcp_offset_world(self) -> torch.Tensor:
        """Rotate the stored local TCP offset to world frame using current FK."""
        if self._tcp_offset_local is None:
            return torch.zeros(3, device=self.device)
        cur_joints = torch.tensor(
            self._robot.get_joint_positions()[:6], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        fk = self._ik_solver.fk(cur_joints)
        return _quat_rotate(self._tcp_offset_local, fk.ee_quaternion[0])

    def compute_inverse_kinematics(
        self, target_position: np.ndarray, target_orientation: np.ndarray
    ):
        """Compute IK so the gripper TCP lands at target_position.

        Args:
            target_position:    world position of the desired TCP (sphere centre).
            target_orientation: world orientation (w,x,y,z) for the EE.

        Returns:
            ArticulationAction: arm joint targets (joint_indices=0..5).
            bool: True if IK succeeded.
        """
        # Adjust: CuRobo places tool0 at ik_target; we want TCP at target_position.
        tcp_offset_world = self._current_tcp_offset_world()
        ik_target = torch.tensor(target_position, device=self.device, dtype=torch.float32) \
                    - tcp_offset_world

        cur_joints = torch.tensor(
            self._robot.get_joint_positions()[:6], device=self.device, dtype=torch.float32
        )
        seed = (
            self._last_succ_joints.unsqueeze(0).unsqueeze(0)
            if self._last_succ_joints is not None
            else cur_joints.unsqueeze(0).unsqueeze(0)
        )
        goal = Pose(
            position=ik_target.unsqueeze(0),
            quaternion=torch.tensor(
                target_orientation, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
        )
        result = self._ik_solver.solve_single(
            goal,
            seed_config=seed,
            retract_config=cur_joints.unsqueeze(0),
        )

        if not result.success.any():
            return ArticulationAction(), False

        # js_solution.position shape: (batch=1, return_seeds=1, dof) → [0,0] for 1D
        solution = result.js_solution.position[0, 0]

        if self._last_succ_joints is not None:
            if torch.max(torch.abs(solution - self._last_succ_joints)).item() > self.MAX_JOINT_DELTA:
                return ArticulationAction(), False

        self._last_succ_joints = solution.clone()
        joints_np = solution.cpu().numpy().flatten()[:6]
        # joint_indices restricts to arm joints 0-5 only (robot has 12 total: 6 arm + 6 gripper)
        return ArticulationAction(joint_positions=joints_np, joint_indices=np.arange(6)), True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    my_world = World(stage_units_in_meters=1.0)

    # Workspace centre: robot at origin, table at Y=0.5, objects at Y=0.5–0.7 (from AsyncDualPlay.yaml)
    my_task = FollowTarget(name="follow_target_task", target_position=[0.0, 0.50, 0.15])
    my_world.add_task(my_task)
    my_world.reset()

    task_params = my_world.get_task("follow_target_task").get_params()
    ur10_name = task_params["robot_name"]["value"]
    target_name = task_params["target_name"]["value"]

    my_ur10 = my_world.scene.get_object(ur10_name)
    articulation_controller = my_ur10.get_articulation_controller()

    print("\n[INFO] Initializing CuRoboKinematicsSolver...")
    my_controller = CuRoboKinematicsSolver(my_ur10, robot_yaml_name="ur10e.yml")

    # Warm up: step a few frames so PhysX transforms settle before measuring offset.
    for _ in range(10):
        my_world.step(render=False)

    ee_pos_world, _ = my_ur10.end_effector.get_world_pose()
    my_controller.setup_tcp_offset(ee_pos_world)
    print("[INFO] CuRobo Ready! Starting simulation loop.\n")

    reset_needed = False
    while simulation_app.is_running():
        my_world.step(render=True)

        if my_world.is_stopped() and not reset_needed:
            reset_needed = True

        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller._last_succ_joints = None
                # Remeasure TCP offset after reset
                for _ in range(10):
                    my_world.step(render=False)
                ee_pos_world, _ = my_ur10.end_effector.get_world_pose()
                my_controller.setup_tcp_offset(ee_pos_world)
                reset_needed = False

            observations = my_world.get_observations()
            actions, succ = my_controller.compute_inverse_kinematics(
                target_position=observations[target_name]["position"],
                target_orientation=observations[target_name]["orientation"],
            )
            if succ:
                articulation_controller.apply_action(actions)
            else:
                carb.log_warn("CuRobo IK did not converge. Holding pose.")

    simulation_app.close()


if __name__ == "__main__":
    main()
