"""Gazebo-style throw primitive for Isaac Lab.

Encapsulates the full throw sequence including IK-based grasping:
  1. SETTLE   — spawn drink on table, let physics settle
  2. APPROACH — IK move EE above drink (XY at drink, Z at crane height)
  3. DESCEND  — IK lower EE to grasp height above drink
  4. GRASP    — gradually close gripper
  5. LIFT     — IK return to crane pose
  6. GO_TO_INIT — move arm to Gazebo init_joints_pose
  7. GO_TO_INITIAL — move to initial_joints_pose (shoulder_pan = initial_joint_value)
  8. THROW    — interpolate all 6 joints from initial to end over `duration`
  9. RELEASE  — open gripper at releasing_time fraction
  10. FLIGHT  — wait for object to settle, measure distance

The 4 learnable macro parameters:
  - initial_joint_value: shoulder_pan angle for the wind-up pose
  - final_joint_value: shoulder_pan angle for the throw end pose
  - releasing_time: fraction of duration at which to release [0.05, 1.0]
  - duration: trajectory execution time in seconds [0.1, 1.0]

Reference joint presets (from gazebo_impl/new_impl.cpp, RIGHT arm):
  init_joints_pose    = [1.6, -1.7236, 2.3313, -2.0629, -1.5987, 0.0]
  end_joints_pose     = [final, -1.2774, 0.8647, -2.1966, -1.5744, 0.0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

RIGHT_INIT_JOINTS = [1.6, -1.7235609493651332, 2.3313358465777796, -2.0628696880736292, -1.5987361113177698, 0.0]
RIGHT_END_JOINTS = [1.6, -1.2773774427226563, 0.8647106329547327, -2.1965824566283167, -1.5743544737445276, 0.0]

LEFT_INIT_JOINTS = [-1.6, -1.435, -2.3313358465777796, -1.0, 1.5987361113177698, 0.0]
LEFT_END_JOINTS = [-1.6, -1.881, -0.8647, -0.9, 1.5743544737445276, 0.0]

PHASE_SETTLE = 60
PHASE_APPROACH = 60
PHASE_DESCEND = 100
PHASE_GRASP = 20
PHASE_LIFT = 60
PHASE_GO_TO_INIT = 40
PHASE_GO_TO_INITIAL = 40
PHASE_THROW_MAX = 120
PHASE_FLIGHT = 300

GRASP_Z_OFFSET = 0.30

DROP_DISTANCE_THRESHOLD = 0.25
DROP_PENALTY_DISTANCE = 10.0
DRINK_BELOW_TABLE_Z = 0.45

DRINK_WORLD_X = 0.65
DRINK_WORLD_Y = 0.50
DRINK_WORLD_Z = 0.62

EE_YAW_OFFSET = math.pi / 2

TOTAL_PRIMITIVE_STEPS = (
    PHASE_SETTLE + PHASE_APPROACH + PHASE_DESCEND + PHASE_GRASP + PHASE_LIFT
    + PHASE_GO_TO_INIT + PHASE_GO_TO_INITIAL + PHASE_THROW_MAX + PHASE_FLIGHT
)


@dataclass
class ThrowPrimitiveParams:
    initial_joint_value: float
    final_joint_value: float
    releasing_time: float
    duration: float


def map_action_to_params(action: torch.Tensor, side: str = "right") -> torch.Tensor:
    """Map normalized action [4] to throw primitive parameters.

    Action space (matching Gazebo RL script):
      action[0]: initial_joint_value in [-1, 1]
      action[1]: final_joint_value in [-1, 1]
      action[2]: releasing_time in [0.05, 1.0]
      action[3]: duration in [0.1, 1.0]

    Returns tensor (..., 4) with: [initial_joint_rad, final_joint_rad, releasing_frac, duration_s]
    """
    if side == "left":
        initial_jv = -((0.5 * (1.0 + action[..., 0]) * 2.4) + 0.001)
        final_jv = -((0.5 * (1.0 + action[..., 1]) * 2.4) + 0.001)
    else:
        initial_jv = (0.5 * (1.0 + action[..., 0]) * 2.4) + 0.001
        final_jv = (0.5 * (1.0 + action[..., 1]) * 2.4) + 0.001

    releasing_time = action[..., 2]
    duration = action[..., 3]

    return torch.stack([initial_jv, final_jv, releasing_time, duration], dim=-1)


def compute_phase_boundaries(duration: torch.Tensor, releasing_time: torch.Tensor, sim_dt: float = 1.0 / 120.0):
    """Compute sub-step boundaries for the throw phase only.

    Args:
        duration: (N,) trajectory duration in seconds
        releasing_time: (N,) fraction of duration at which to release

    Returns:
        dict with throw_steps and release_step tensors
    """
    throw_steps = (duration / sim_dt).clamp(min=12, max=PHASE_THROW_MAX).long()
    release_step = (releasing_time * throw_steps.float()).long().clamp(min=1)

    return {
        "throw_steps": throw_steps,
        "release_step": release_step,
    }


def build_joint_targets(
    init_joints: torch.Tensor,
    initial_joint_value: torch.Tensor,
    end_joints: torch.Tensor,
    final_joint_value: torch.Tensor,
    side: str = "right",
    joint_pos_limits: torch.Tensor | None = None,
):
    """Build the initial_joints_pose and end_joints_pose per-env.

    Matches Gazebo new_impl.cpp: when initial_joint_value is set (non-zero),
    the left arm also overrides joint[1] to -1.3 (hardcoded in Gazebo).

    Joint targets are clamped to hardware limits (matching Gazebo's implicit
    clamping by the ROS trajectory controller).

    Args:
        init_joints: (6,) base init joints pose
        initial_joint_value: (N,) shoulder_pan override for initial pose
        end_joints: (6,) base end joints pose
        final_joint_value: (N,) shoulder_pan override for end pose
        side: "right" or "left"
        joint_pos_limits: (N, 6, 2) or (6, 2) joint position limits [lower, upper].
            If provided, targets are clamped to these limits.

    Returns:
        initial_joints_pose: (N, 6) - init with shoulder_pan overridden
        end_joints_pose: (N, 6) - end with shoulder_pan overridden
    """
    N = initial_joint_value.shape[0]
    device = initial_joint_value.device

    initial_joints_pose = init_joints.unsqueeze(0).expand(N, -1).clone().to(device)
    initial_joints_pose[:, 0] = initial_joint_value
    if side == "left":
        initial_joints_pose[:, 1] = -1.3

    end_joints_pose = end_joints.unsqueeze(0).expand(N, -1).clone().to(device)
    end_joints_pose[:, 0] = final_joint_value

    if joint_pos_limits is not None:
        limits = joint_pos_limits.to(device)
        if limits.dim() == 2:
            lower = limits[:, 0].unsqueeze(0).expand(N, -1)
            upper = limits[:, 1].unsqueeze(0).expand(N, -1)
        else:
            lower = limits[..., 0]
            upper = limits[..., 1]
        initial_joints_pose = torch.clamp(initial_joints_pose, min=lower, max=upper)
        end_joints_pose = torch.clamp(end_joints_pose, min=lower, max=upper)

    return initial_joints_pose, end_joints_pose


class ThrowPrimitiveExecutor:
    """Executes the Gazebo-style throw primitive with IK-based grasping.

    Requires a ThrowingEnv (ManagerBasedRLEnv) with DiffIK action terms.
    The IK phases use env.step(action) with position error as the action.
    """

    def __init__(
        self,
        robot: "Articulation",
        milk: "RigidObject",
        arm_joint_ids: list[int],
        gripper_set_fn,
        ee_body_name: str = "right_wrist_3_link",
        side: str = "right",
        sim_dt: float = 1.0 / 120.0,
        device: str = "cuda:0",
    ):
        self.robot = robot
        self.milk = milk
        self.arm_joint_ids = arm_joint_ids
        self.gripper_set_fn = gripper_set_fn
        self.ee_body_name = ee_body_name
        self.side = side
        self.sim_dt = sim_dt
        self.device = device

        self._ee_body_ids, _ = robot.find_bodies([ee_body_name])

        if side == "right":
            self.init_joints = torch.tensor(RIGHT_INIT_JOINTS, device=device)
            self.end_joints_base = torch.tensor(RIGHT_END_JOINTS, device=device)
        else:
            self.init_joints = torch.tensor(LEFT_INIT_JOINTS, device=device)
            self.end_joints_base = torch.tensor(LEFT_END_JOINTS, device=device)

    def _ee_state(self):
        pos = self.robot.data.body_pos_w[:, self._ee_body_ids[0]]
        quat = self.robot.data.body_quat_w[:, self._ee_body_ids[0]]
        return pos, quat

    def _compute_ik_action(self, env, target_pos_local, env_origin):
        """Compute IK action (position error) toward a local-frame target."""
        from isaaclab.utils.math import compute_pose_error
        ee_pos, ee_quat = self._ee_state()
        target_world = target_pos_local + env_origin.unsqueeze(0)
        pos_err, _ = compute_pose_error(
            ee_pos, ee_quat, target_world, ee_quat.clone(), rot_error_type="axis_angle"
        )
        action = torch.cat([pos_err[0], torch.zeros(3, device=self.device)], dim=-1).unsqueeze(0)
        return action

    def execute_single(
        self,
        env,
        params: ThrowPrimitiveParams,
        env_id: int = 0,
        headless: bool = False,
        drink_x: float = DRINK_WORLD_X,
        drink_y: float = DRINK_WORLD_Y,
        drink_z: float = DRINK_WORLD_Z,
        verbose: bool = False,
    ) -> float:
        """Execute full throw primitive with IK grasping. Returns landing distance.

        Phases: SETTLE → APPROACH → DESCEND → GRASP → LIFT →
                GO_TO_INIT → GO_TO_INITIAL → THROW → FLIGHT

        If verbose=True, logs EE pos/quat, drink pos, and distance to target every 5 steps.
        """
        device = self.device
        env_ids = torch.tensor([env_id], device=device)
        origin = env.scene.env_origins[env_id].to(device)
        target_obj = env.scene["target"]
        _step = [0]

        def _log(phase):
            _step[0] += 1
            if not verbose or _step[0] % 5 != 0:
                return
            ee_p, ee_q = self._ee_state()
            ee_local = ee_p[env_id] - origin
            milk_local = self.milk.data.root_pos_w[env_id, :3] - origin
            tgt_local = target_obj.data.root_pos_w[env_id, :3] - origin
            dist = torch.norm(self.milk.data.root_pos_w[env_id, :3] - target_obj.data.root_pos_w[env_id, :3]).item()
            print(
                f"  {_step[0]:>5} {phase:>10}  "
                f"ee=({ee_local[0]:+.3f},{ee_local[1]:+.3f},{ee_local[2]:+.3f}) "
                f"q=({ee_q[env_id,0]:.2f},{ee_q[env_id,1]:.2f},{ee_q[env_id,2]:.2f},{ee_q[env_id,3]:.2f}) "
                f"obj=({milk_local[0]:+.3f},{milk_local[1]:+.3f},{milk_local[2]:+.3f}) "
                f"dist={dist:.3f}",
                flush=True,
            )

        if verbose:
            print(
                f"  {'step':>5} {'phase':>10}  "
                f"{'ee_pos':^21} {'ee_quat':^23} {'obj_pos':^21} {'dist':>6}",
                flush=True,
            )

        # ── SETTLE: spawn drink on table ────────────────────────────────
        drink_pos_w = torch.tensor(
            [[drink_x, drink_y, drink_z]], device=device
        ) + origin.unsqueeze(0)
        drink_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        self.milk.write_root_pose_to_sim(
            torch.cat([drink_pos_w, drink_quat], dim=-1), env_ids=env_ids
        )
        self.milk.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
        self.gripper_set_fn(self.robot, 0.0, env_ids)
        for _ in range(PHASE_SETTLE):
            env.step(torch.zeros(1, 6, device=device))
            _log("SETTLE")

        drink_actual_w = self.milk.data.root_pos_w[env_id, :3].clone()
        drink_local = drink_actual_w - origin
        dx, dy, dz = drink_local[0].item(), drink_local[1].item(), drink_local[2].item()

        crane_pos, _ = self._ee_state()
        crane_local = crane_pos[0] - origin
        cx, cy, cz = crane_local[0].item(), crane_local[1].item(), crane_local[2].item()

        # ── APPROACH: IK move EE above drink at crane Z ─────────────────
        target_approach = torch.tensor([[dx, dy, cz]], device=device)
        for _ in range(PHASE_APPROACH):
            action = self._compute_ik_action(env, target_approach, origin)
            env.step(action)
            _log("APPROACH")

        # ── DESCEND: IK lower to grasp height ───────────────────────────
        target_descend = torch.tensor([[dx, dy, dz + GRASP_Z_OFFSET]], device=device)
        for _ in range(PHASE_DESCEND):
            action = self._compute_ik_action(env, target_descend, origin)
            env.step(action)
            _log("DESCEND")

        # ── GRASP: close gripper gradually ──────────────────────────────
        for i in range(PHASE_GRASP):
            progress = i / (PHASE_GRASP - 1) if PHASE_GRASP > 1 else 1.0
            self.gripper_set_fn(self.robot, 0.48 * progress, env_ids)
            env.step(torch.zeros(1, 6, device=device))
            _log("GRASP")

        # ── Kinematic hold from here until release ──────────────────────
        # Record drink world pos relative to EE at grasp moment
        ee_p_g, _ = self._ee_state()
        milk_p_g = self.milk.data.root_pos_w[env_id, :3].clone()
        _grasp_offset = milk_p_g - ee_p_g[env_id]
        _grasp_quat = self.milk.data.root_quat_w[env_id].clone()

        def _hold():
            ee_p, _ = self._ee_state()
            pos = (ee_p[env_id] + _grasp_offset).unsqueeze(0)
            pose = torch.cat([pos, _grasp_quat.unsqueeze(0)], dim=-1)
            self.milk.write_root_pose_to_sim(pose, env_ids=env_ids)
            self.milk.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

        # ── LIFT: IK return to crane pose ───────────────────────────────
        target_lift = torch.tensor([[cx, cy, cz]], device=device)
        for _ in range(PHASE_LIFT):
            action = self._compute_ik_action(env, target_lift, origin)
            env.step(action)
            _hold()
            _log("LIFT")

        # ── GO_TO_INIT ──────────────────────────────────────────────────
        init_joints_rotated = self.init_joints.clone()
        init_joints_rotated[5] += EE_YAW_OFFSET
        self.robot.set_joint_position_target(
            init_joints_rotated.unsqueeze(0), joint_ids=self.arm_joint_ids, env_ids=env_ids
        )
        for _ in range(PHASE_GO_TO_INIT):
            self.gripper_set_fn(self.robot, 0.7, env_ids)
            self.robot.write_data_to_sim()
            env.sim.step(render=not headless)
            env.scene.update(self.sim_dt)
            _hold()
            _log("GO_TO_INIT")

        # ── GO_TO_INITIAL ───────────────────────────────────────────────
        initial_jv = torch.tensor([params.initial_joint_value], device=device)
        final_jv = torch.tensor([params.final_joint_value], device=device)
        arm_limits = self.robot.data.joint_pos_limits[0, self.arm_joint_ids, :]
        initial_joints_pose, end_joints_pose = build_joint_targets(
            self.init_joints, initial_jv, self.end_joints_base, final_jv,
            side=self.side, joint_pos_limits=arm_limits,
        )
        initial_joints_pose[:, 5] += EE_YAW_OFFSET
        end_joints_pose[:, 5] += EE_YAW_OFFSET
        self.robot.set_joint_position_target(
            initial_joints_pose, joint_ids=self.arm_joint_ids, env_ids=env_ids
        )
        for _ in range(PHASE_GO_TO_INITIAL):
            self.gripper_set_fn(self.robot, 0.7, env_ids)
            self.robot.write_data_to_sim()
            env.sim.step(render=not headless)
            env.scene.update(self.sim_dt)
            _hold()
            _log("GO_INITIAL")

        # ── THROW: interpolate from initial to end joints ───────────────
        duration_t = torch.tensor([params.duration], device=device)
        releasing_t = torch.tensor([params.releasing_time], device=device)
        boundaries = compute_phase_boundaries(duration_t, releasing_t, self.sim_dt)
        throw_steps = boundaries["throw_steps"][0].item()
        release_step_in_throw = boundaries["release_step"][0].item()
        released = False

        for i in range(throw_steps):
            t = min(float(i) / max(throw_steps - 1, 1), 1.0)
            target = initial_joints_pose * (1.0 - t) + end_joints_pose * t
            self.robot.set_joint_position_target(
                target, joint_ids=self.arm_joint_ids, env_ids=env_ids
            )
            if i >= release_step_in_throw and not released:
                self.gripper_set_fn(self.robot, 0.0, env_ids)
                released = True
            elif not released:
                self.gripper_set_fn(self.robot, 0.7, env_ids)
            self.robot.write_data_to_sim()
            env.sim.step(render=not headless)
            env.scene.update(self.sim_dt)
            if not released:
                _hold()
            _log("THROW")

        if not released:
            self.gripper_set_fn(self.robot, 0.0, env_ids)

        # ── FLIGHT: wait for settle ─────────────────────────────────────
        settle_count = 0
        for _ in range(PHASE_FLIGHT):
            env.sim.step(render=not headless)
            env.scene.update(self.sim_dt)
            _log("FLIGHT")
            vel = self.milk.data.root_lin_vel_w[env_id]
            if torch.norm(vel).item() < 0.05:
                settle_count += 1
                if settle_count >= 30:
                    break
            else:
                settle_count = 0

        # ── Measure distance ────────────────────────────────────────────
        milk_pos = self.milk.data.root_pos_w[env_id, :3]
        target_obj = env.scene["target"]
        target_pos = target_obj.data.root_pos_w[env_id, :3]
        distance = torch.norm(milk_pos - target_pos).item()

        # ── RETURN to crane pose ────────────────────────────────────────
        for _ in range(40):
            action = self._compute_ik_action(env, target_lift, origin)
            env.step(action)

        return distance


def execute_primitive_batched(
    env,
    actions: torch.Tensor,
    arm_joint_ids: list[int],
    ee_body_name: str,
    gripper_set_fn,
    side: str = "right",
    drink_x: float = DRINK_WORLD_X,
    drink_y: float = DRINK_WORLD_Y,
    drink_z: float = DRINK_WORLD_Z,
) -> torch.Tensor:
    """Execute the throw primitive for all N envs in parallel. Returns distances (N,).

    Uses env.step(actions) for IK phases (all envs execute same phase simultaneously).
    Uses direct joint control for throw phase.

    Args:
        env: ThrowingEnv (ManagerBasedRLEnv with DiffIK action term)
        actions: (N, 4) raw actions [initial_jv, final_jv, releasing_time, duration]
        arm_joint_ids: joint indices for the throwing arm
        ee_body_name: end-effector body name
        gripper_set_fn: function to set gripper state
        side: "right" or "left"
    """
    render = env.sim.has_gui()
    from isaaclab.utils.math import compute_pose_error

    device = env.device
    N = env.num_envs
    all_ids = torch.arange(N, device=device)
    origins = env.scene.env_origins.to(device)
    robot = env.scene["robot"]
    milk = env.scene["milk"]
    sim_dt = 1.0 / 120.0

    ee_body_ids, _ = robot.find_bodies([ee_body_name])

    if side == "right":
        init_joints = torch.tensor(RIGHT_INIT_JOINTS, device=device)
        end_joints_base = torch.tensor(RIGHT_END_JOINTS, device=device)
    else:
        init_joints = torch.tensor(LEFT_INIT_JOINTS, device=device)
        end_joints_base = torch.tensor(LEFT_END_JOINTS, device=device)

    params = map_action_to_params(actions, side=side)
    initial_jv = params[:, 0]
    final_jv = params[:, 1]
    releasing_time = params[:, 2]
    duration = params[:, 3]

    boundaries = compute_phase_boundaries(duration, releasing_time, sim_dt)
    arm_limits = robot.data.joint_pos_limits[0, arm_joint_ids, :]
    initial_joints_pose, end_joints_pose = build_joint_targets(
        init_joints, initial_jv, end_joints_base, final_jv,
        side=side, joint_pos_limits=arm_limits,
    )
    initial_joints_pose[:, 5] += EE_YAW_OFFSET
    end_joints_pose[:, 5] += EE_YAW_OFFSET
    throw_steps_per_env = boundaries["throw_steps"]
    release_step_per_env = boundaries["release_step"]
    max_throw_steps = throw_steps_per_env.max().item()

    def _ee_state():
        return robot.data.body_pos_w[:, ee_body_ids[0]], robot.data.body_quat_w[:, ee_body_ids[0]]

    def _ik_action_toward(target_local):
        ee_pos, ee_quat = _ee_state()
        target_world = target_local + origins
        pos_err, _ = compute_pose_error(
            ee_pos, ee_quat, target_world, ee_quat.clone(), rot_error_type="axis_angle"
        )
        return torch.cat([pos_err, torch.zeros(N, 3, device=device)], dim=-1)

    # ── SETTLE: spawn drinks on table ───────────────────────────────────
    drink_pos_local = torch.tensor([[drink_x, drink_y, drink_z]], device=device).expand(N, -1)
    drink_pos_w = drink_pos_local + origins
    drink_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, -1)
    milk.write_root_pose_to_sim(torch.cat([drink_pos_w, drink_quat], dim=-1), env_ids=all_ids)
    milk.write_root_velocity_to_sim(torch.zeros(N, 6, device=device), env_ids=all_ids)
    gripper_set_fn(robot, 0.0, all_ids)
    for _ in range(PHASE_SETTLE):
        env.step(torch.zeros(N, 6, device=device))

    drink_actual_local = milk.data.root_pos_w[:, :3] - origins
    crane_pos, _ = _ee_state()
    crane_local = crane_pos - origins

    # ── APPROACH: IK move EE above drink at crane Z ─────────────────────
    target_approach = torch.stack([
        drink_actual_local[:, 0], drink_actual_local[:, 1], crane_local[:, 2]
    ], dim=-1)
    for _ in range(PHASE_APPROACH):
        env.step(_ik_action_toward(target_approach))

    # ── DESCEND: IK lower to grasp height ───────────────────────────────
    target_descend = torch.stack([
        drink_actual_local[:, 0], drink_actual_local[:, 1],
        drink_actual_local[:, 2] + GRASP_Z_OFFSET
    ], dim=-1)
    for _ in range(PHASE_DESCEND):
        env.step(_ik_action_toward(target_descend))

    # ── GRASP: close gripper gradually ──────────────────────────────────
    for i in range(PHASE_GRASP):
        progress = i / (PHASE_GRASP - 1) if PHASE_GRASP > 1 else 1.0
        gripper_set_fn(robot, 0.48 * progress, all_ids)
        env.step(torch.zeros(N, 6, device=device))

    # ── Kinematic hold from here until release ─────────────────────────
    ee_p_g, _ = _ee_state()
    milk_p_g = milk.data.root_pos_w[:, :3].clone()
    _grasp_offset = milk_p_g - ee_p_g
    _grasp_quat = milk.data.root_quat_w.clone()

    def _hold_batched(env_ids_hold=None):
        if env_ids_hold is None:
            env_ids_hold = all_ids
        ee_p, _ = _ee_state()
        pos = ee_p[env_ids_hold] + _grasp_offset[env_ids_hold]
        pose = torch.cat([pos, _grasp_quat[env_ids_hold]], dim=-1)
        milk.write_root_pose_to_sim(pose, env_ids=env_ids_hold)
        milk.write_root_velocity_to_sim(
            torch.zeros(len(env_ids_hold), 6, device=device), env_ids=env_ids_hold)

    # ── LIFT: IK return to crane pose ───────────────────────────────────
    for _ in range(PHASE_LIFT):
        env.step(_ik_action_toward(crane_local))
        _hold_batched()

    # ── GO_TO_INIT ──────────────────────────────────────────────────────
    init_joints_rotated = init_joints.clone()
    init_joints_rotated[5] += EE_YAW_OFFSET
    robot.set_joint_position_target(
        init_joints_rotated.unsqueeze(0).expand(N, -1), joint_ids=arm_joint_ids
    )
    for _ in range(PHASE_GO_TO_INIT):
        gripper_set_fn(robot, 0.7, all_ids)
        robot.write_data_to_sim()
        env.sim.step(render=render)
        env.scene.update(sim_dt)
        _hold_batched()

    # ── GO_TO_INITIAL ───────────────────────────────────────────────────
    robot.set_joint_position_target(initial_joints_pose, joint_ids=arm_joint_ids)
    for _ in range(PHASE_GO_TO_INITIAL):
        gripper_set_fn(robot, 0.7, all_ids)
        robot.write_data_to_sim()
        env.sim.step(render=render)
        env.scene.update(sim_dt)
        _hold_batched()

    dropped = torch.zeros(N, device=device, dtype=torch.bool)

    # ── THROW: interpolate from initial to end joints ───────────────────
    released = torch.zeros(N, device=device, dtype=torch.bool)
    for i in range(max_throw_steps):
        active = i < throw_steps_per_env
        t_frac = (torch.tensor(i, device=device).float() / (throw_steps_per_env.float() - 1).clamp(min=1)).clamp(0, 1)
        targets = initial_joints_pose * (1.0 - t_frac.unsqueeze(-1)) + end_joints_pose * t_frac.unsqueeze(-1)
        robot.set_joint_position_target(targets, joint_ids=arm_joint_ids)

        should_release = active & (i >= release_step_per_env) & ~released
        if should_release.any():
            release_ids = should_release.nonzero(as_tuple=True)[0]
            gripper_set_fn(robot, 0.0, release_ids)
            released[should_release] = True

        still_closed = ~released
        if still_closed.any():
            gripper_set_fn(robot, 0.7, still_closed.nonzero(as_tuple=True)[0])

        robot.write_data_to_sim()
        env.sim.step(render=render)
        env.scene.update(sim_dt)

        if still_closed.any():
            _hold_batched(still_closed.nonzero(as_tuple=True)[0])

    still_holding = ~released
    if still_holding.any():
        gripper_set_fn(robot, 0.0, still_holding.nonzero(as_tuple=True)[0])

    # ── FLIGHT: wait for settle ─────────────────────────────────────────
    settle_counts = torch.zeros(N, device=device, dtype=torch.long)
    settled = torch.zeros(N, device=device, dtype=torch.bool)
    for _ in range(PHASE_FLIGHT):
        env.sim.step(render=render)
        env.scene.update(sim_dt)
        vel_norm = torch.norm(milk.data.root_lin_vel_w[:, :3], dim=-1)
        low_vel = vel_norm < 0.05
        settle_counts[low_vel & ~settled] += 1
        settle_counts[~low_vel] = 0
        settled |= settle_counts >= 30
        if settled.all():
            break

    # ── Measure distances ───────────────────────────────────────────────
    target_obj = env.scene["target"]
    milk_final_pos = milk.data.root_pos_w[:, :3].clone()
    target_final_pos = target_obj.data.root_pos_w[:, :3].clone()
    distances = torch.norm(milk_final_pos - target_final_pos, dim=-1)
    distances[dropped] = DROP_PENALTY_DISTANCE

    return {
        "distances": distances,
        "dropped": dropped,
        "milk_final_pos": milk_final_pos - origins,
        "target_pos": target_final_pos - origins,
    }
