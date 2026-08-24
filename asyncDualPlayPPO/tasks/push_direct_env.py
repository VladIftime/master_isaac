"""PushDirectEnv — DirectRLEnv for push primitive with cuRobo IK + HER.

One outer step() = one complete push macro-action (72 physics substeps).
No ManagerBasedRLEnv overhead.  State-machine _apply_action() executes cuRobo IK
per waypoint substep.  Returns dict observations for HER (SB3 HerReplayBuffer).
"""

from __future__ import annotations

import math
import torch

import gymnasium as gym
import numpy as np

from isaaclab.envs import DirectRLEnv

from .push_direct_env_cfg import (
    PushDirectEnvCfg,
    PHASE_APPROACH, PHASE_DESCEND, PHASE_PUSH, PHASE_RETRACT, PHASE_RETURN,
    TOTAL_DECIMATION, PUSH_APPROACH_HEIGHT, OBS_ROBOT_DIM, OBS_OBJ_STATE_DIM,
    OBS_GOAL_DIM, OBS_DIST_DIM, OBS_REL_DIM,
)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_WS_X = (-0.50, 0.50)
_WS_Y = (0.25, 0.70)
_WS_Z = (0.25, 0.55)

_QUAT_TOOL_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)


def _quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    roll = torch.atan2(2.0 * (quat[..., 0] * quat[..., 1] + quat[..., 2] * quat[..., 3]),
                        1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2))
    pitch = torch.asin(torch.clamp(2.0 * (quat[..., 0] * quat[..., 2] - quat[..., 3] * quat[..., 1]),
                                   -1.0, 1.0))
    yaw = torch.atan2(2.0 * (quat[..., 0] * quat[..., 3] + quat[..., 1] * quat[..., 2]),
                       1.0 - 2.0 * (quat[..., 2] ** 2 + quat[..., 3] ** 2))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _yaw_distance_rad(euler_a: torch.Tensor, euler_b: torch.Tensor) -> torch.Tensor:
    diff = (euler_a[..., 2] - euler_b[..., 2]) % (2.0 * torch.pi)
    return torch.where(diff > torch.pi, 2.0 * torch.pi - diff, diff)


def _euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


class PushDirectEnv(DirectRLEnv):
    cfg: PushDirectEnvCfg

    def __init__(self, cfg: PushDirectEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_envs = cfg.scene.num_envs
        self._robot = self.scene["robot"]
        self._target_object = self.scene["target_object"]
        self._table = self.scene["table"]

        self._arm_jids, _ = self._robot.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
        lf_ids, _ = self._robot.find_bodies("left_inner_finger")
        rf_ids, _ = self._robot.find_bodies("right_inner_finger")
        self._lf_id = lf_ids[0]
        self._rf_id = rf_ids[0]

        arm_patterns = ["shoulder_.*", "elbow_.*", "wrist_.*"]
        self._arm_ids, _ = self._robot.find_joints(arm_patterns)

        self._quat_tool_down = _QUAT_TOOL_DOWN.to(self.device)

        self._init_ik_solver()
        self._calibrate_ik_error()

        N = self._num_envs
        self._step_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        self._waypoints_pos = torch.zeros(N, TOTAL_DECIMATION, 3, device=self.device)
        self._waypoints_quat = torch.zeros(N, TOTAL_DECIMATION, 4, device=self.device)
        self._joint_cmd = torch.zeros(N, 6, device=self.device)

        self.push_count = torch.zeros(N, dtype=torch.long, device=self.device)
        self.prev_obj_pos = torch.zeros(N, 3, device=self.device)
        self.prev_obj_euler = torch.zeros(N, 3, device=self.device)
        self.at_goal_pos = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.at_goal_both = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.goal_pos_euler = torch.zeros(N, 6, device=self.device)
        self._gave_completion = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._gave_rot_bonus = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._ep_started = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.ep_start_pos = torch.zeros(N, 3, device=self.device)
        self.ep_start_euler = torch.zeros(N, 3, device=self.device)
        self._episode_reward = torch.zeros(N, device=self.device)
        self._terminated_this_step = torch.zeros(N, dtype=torch.bool, device=self.device)

        self._total_ik_fails = 0
        self._total_ik_steps = 0

        self._obs_dim = OBS_ROBOT_DIM + OBS_OBJ_STATE_DIM + OBS_GOAL_DIM + OBS_DIST_DIM

        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        act_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )
        self.single_observation_space = obs_space
        self.single_action_space = act_space
        self.observation_space = obs_space
        self.action_space = act_space

        self._nan_warned = False
        self._episode_log = []

    def _init_ik_solver(self):
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.types.robot import RobotConfig
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml as curobo_load_yaml

        tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)
        ur5e_yaml = curobo_load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))
        robot_cfg = RobotConfig.from_dict(ur5e_yaml["robot_cfg"], tensor_args)
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg, world_model=None, tensor_args=tensor_args,
        )
        ik_config.solver.newton_optimizer.n_iters = self.cfg.ik_n_iters
        ik_config.solver.newton_optimizer.inner_iters = self.cfg.ik_inner_iters
        self._ik_solver = IKSolver(ik_config)

        N = self._num_envs
        wup_pos = torch.zeros(N, 3, device=self.device)
        wup_quat = _QUAT_TOOL_DOWN.to(self.device).expand(N, 4)
        from curobo.types.math import Pose as CuroboPose
        self._ik_solver.solve_batch(
            CuroboPose(position=wup_pos, quaternion=wup_quat),
            seed_config=torch.zeros(N, 1, 6, device=self.device),
            retract_config=torch.zeros(N, 6, device=self.device),
        )

    def _calibrate_ik_error(self):
        from curobo.types.math import Pose as CuroboPose

        N = self._num_envs
        calib_pos = torch.zeros(N, 3, device=self.device)
        calib_pos[:, 1] = 0.60
        calib_pos[:, 2] = 0.25
        calib_cur = self._robot.data.joint_pos[:, self._arm_jids]
        calib_res = self._ik_solver.solve_batch(
            CuroboPose(position=calib_pos,
                       quaternion=self._quat_tool_down.expand(N, 4)),
            seed_config=calib_cur.unsqueeze(1),
            retract_config=calib_cur,
        )
        calib_cmd = calib_res.solution.view(N, 6)
        for _ in range(30):
            self._robot.set_joint_position_target(calib_cmd, joint_ids=self._arm_ids)
            self._robot.write_joint_position_to_sim(
                self._robot.data.joint_pos[:, self._arm_ids], joint_ids=self._arm_ids)
            self.sim.step(render=False)
        finger_after = self._tcp_pos_local()
        self._ik_error = (finger_after - calib_pos).clone()

    def _tcp_pos_local(self) -> torch.Tensor:
        lf_w = self._robot.data.body_pos_w[:, self._lf_id]
        rf_w = self._robot.data.body_pos_w[:, self._rf_id]
        return ((lf_w + rf_w) / 2.0 - self.scene.env_origins).clone()

    def _setup_scene(self):
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor):
        from asyncDualPlayPPO.tasks.utils.action_push_continuous import (
            decode_push_action_continuous,
            decode_push_action_relative_continuous,
        )
        from asyncDualPlayPPO.tasks.utils.action_push import compute_push_waypoints

        actions = actions.clamp(-1.0, 1.0)
        actions = torch.nan_to_num(actions, nan=0.0)
        self._step_idx.zero_()
        self._terminated_this_step.zero_()

        ee_pos_local = self._tcp_pos_local()
        ee_quat_w = self._quat_tool_down.expand(self._num_envs, 4).clone()

        if self.cfg.rel_act:
            obj_pos_local = self._get_obj_pos_local()
            obj_xy = obj_pos_local[:, :2]
            obj_euler = _quat_to_euler_xyz(self._target_object.data.root_quat_w)
            obj_yaw = obj_euler[:, 2]
            Xs, Ys, length, theta = decode_push_action_relative_continuous(
                actions, obj_xy, obj_yaw,
                min_r=self.cfg.action_min_r,
                max_r=self.cfg.action_max_r,
                max_len=self.cfg.action_max_len,
            )
        else:
            Xs, Ys, length, theta = decode_push_action_continuous(
                actions, max_len=self.cfg.action_max_len,
            )

        Xf = Xs + length * torch.cos(theta)
        Yf = Ys + length * torch.sin(theta)

        self._Xs = Xs
        self._Ys = Ys
        self._Xf = Xf
        self._Yf = Yf
        self._length = length
        self._theta = theta

        waypoints = compute_push_waypoints(
            Xs=Xs, Ys=Ys, length=length, theta=theta,
            current_ee_pos=ee_pos_local,
            current_ee_quat=ee_quat_w,
            device=self.device,
        )

        for i, (wp_pos, wp_quat, _wp_grip) in enumerate(waypoints):
            self._waypoints_pos[:, i] = wp_pos
            self._waypoints_quat[:, i] = wp_quat

        obj_pos_raw = self._get_obj_pos_local()
        self.prev_obj_pos[:] = obj_pos_raw
        self.prev_obj_euler[:] = _quat_to_euler_xyz(self._target_object.data.root_quat_w)

        new_ep = ~self._ep_started
        if new_ep.any():
            self.ep_start_pos[new_ep] = self.prev_obj_pos[new_ep].clone()
            self.ep_start_euler[new_ep] = self.prev_obj_euler[new_ep].clone()
            self._ep_started[new_ep] = True

    def _apply_action(self):
        from curobo.types.math import Pose as CuroboPose

        step = self._step_idx
        N = self._num_envs
        if (step >= TOTAL_DECIMATION).any():
            self._step_idx += 1
            return

        idx = torch.arange(N, device=self.device)
        step_clamped = step.clamp(0, TOTAL_DECIMATION - 1)
        wp_pos = self._waypoints_pos[idx, step_clamped]
        wp_quat = self._waypoints_quat[idx, step_clamped]

        ik_target = wp_pos - self._ik_error
        ik_target[:, 0].clamp_(*_WS_X)
        ik_target[:, 1].clamp_(*_WS_Y)
        ik_target[:, 2].clamp_(*_WS_Z)

        result = self._ik_solver.solve_batch(
            CuroboPose(position=ik_target, quaternion=wp_quat),
            seed_config=self._joint_cmd.unsqueeze(1),
            retract_config=self._joint_cmd,
        )

        ik_ok = result.success.squeeze(-1)
        cur_joints = self._robot.data.joint_pos[:, self._arm_jids]

        self._total_ik_steps += N
        self._total_ik_fails += int((~ik_ok).sum().item())

        solved = result.solution.view(N, 6)
        elbow_bad = solved[:, 2] < 0.0
        if elbow_bad.any():
            ik_ok[elbow_bad] = False

        raw_cmd = torch.where(ik_ok.unsqueeze(-1), solved, self._joint_cmd)
        if self._terminated_this_step.any():
            raw_cmd[self._terminated_this_step] = cur_joints[self._terminated_this_step]
        self._joint_cmd = raw_cmd.detach().clone()

        self._robot.set_joint_position_target(raw_cmd, joint_ids=self._arm_ids)
        self._robot.write_joint_position_to_sim(
            self._robot.data.joint_pos[:, self._arm_ids], joint_ids=self._arm_ids)

        self._step_idx += 1

    def _get_observations(self) -> dict:
        ee_pos = self._tcp_pos_local()
        ee_quat_raw = self._robot.data.body_quat_w[:, self._lf_id]
        ee_euler = _quat_to_euler_xyz(ee_quat_raw)

        obj_pos = self._get_obj_pos_local()
        obj_quat_raw = self._target_object.data.root_quat_w
        obj_euler = _quat_to_euler_xyz(obj_quat_raw)

        obj_linvel = self._target_object.data.root_lin_vel_w[:, :3].clamp(-5.0, 5.0)
        obj_angvel = self._target_object.data.root_ang_vel_w[:, :3].clamp(-5.0, 5.0)

        dist_to_ee = torch.norm(obj_pos - ee_pos, dim=-1, keepdim=True)

        ee_x = ee_pos[:, 0:1]
        palm_euler = _quat_to_euler_xyz(ee_quat_raw)
        ee_z = ee_pos[:, 2:3]
        contact = (
            (obj_pos[:, 0:1].abs() < 0.5)
            & (obj_pos[:, 1:2] > 0.1)
            & (obj_pos[:, 1:2] < 0.8)
            & (obj_pos[:, 2:3] < 0.15)
        ).float()
        contact_1d = contact * 1.0

        goal_pos = self.goal_pos_euler[:, :3]
        goal_euler = self.goal_pos_euler[:, 3:6]

        pos_dist = torch.norm(obj_pos[:, :2] - goal_pos[:, :2], dim=-1, keepdim=True)
        rot_dist = _yaw_distance_rad(obj_euler, goal_euler).unsqueeze(-1)

        observation = torch.cat([
            ee_pos, ee_euler,
            obj_pos, obj_euler,
            obj_linvel, obj_angvel,
            dist_to_ee, contact_1d,
            torch.zeros(self._num_envs, 2, device=self.device),
        ], dim=-1)

        if self.cfg.rel_obs:
            rel_dx = (goal_pos[:, 0:1] - obj_pos[:, 0:1])
            rel_dy = (goal_pos[:, 1:2] - obj_pos[:, 1:2])
            observation = torch.cat([
                ee_pos, ee_euler,
                obj_pos, obj_euler,
                obj_linvel, obj_angvel,
                dist_to_ee, contact_1d,
                rel_dx, rel_dy,
            ], dim=-1)

        achieved_goal = torch.cat([
            obj_pos[:, :2], obj_euler[:, 2:3],
        ], dim=-1)  # (N, 3)

        desired_goal = torch.cat([
            goal_pos[:, :2], goal_euler[:, 2:3],
        ], dim=-1)  # (N, 3)

        return {
            "policy": observation,
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._get_obj_pos_local()
        obj_euler = _quat_to_euler_xyz(self._target_object.data.root_quat_w)
        goal_pos = self.goal_pos_euler[:, :3]
        goal_euler = self.goal_pos_euler[:, 3:6]

        d_prev = (self.prev_obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1)
        d_now = (obj_pos[:, :2] - goal_pos[:, :2]).norm(dim=-1)
        y_prev = _yaw_distance_rad(self.prev_obj_euler, goal_euler)
        y_now = _yaw_distance_rad(obj_euler, goal_euler)

        alpha = self.cfg.dense_alpha
        pos_imp = alpha * (d_prev - d_now) / d_prev.clamp(min=0.01)
        rot_imp = alpha * (y_prev - y_now) / y_prev.clamp(min=0.01)
        penalty = -self.cfg.dense_beta * d_now
        rot_penalty = -self.cfg.dense_rot_beta * y_now

        pos_imp = torch.clamp(pos_imp, -5.0, 5.0)
        rot_imp = torch.clamp(rot_imp, -4.0, 4.0)
        penalty = torch.clamp(penalty, -2.0, 0.0)
        rot_penalty = torch.clamp(rot_penalty, -1.0, 0.0)
        reward = pos_imp + rot_imp + penalty + rot_penalty

        pos_ok = d_now < self.cfg.push_success_threshold_pos
        rot_ok = y_now < self.cfg.push_success_threshold_rot
        both_ok = pos_ok & rot_ok

        new_completion = pos_ok & ~self._gave_completion
        completion = torch.where(new_completion,
                                 torch.tensor(self.cfg.completion_bonus, device=self.device),
                                 torch.zeros_like(reward))
        reward = reward + completion

        new_rot_bonus = both_ok & (~self._gave_completion | ~self._gave_rot_bonus)
        rot_bonus = torch.where(new_rot_bonus,
                                torch.tensor(self.cfg.rotation_sub_bonus, device=self.device),
                                torch.zeros_like(reward))
        reward = reward + rot_bonus

        tipped = (obj_euler[:, 0].abs() > self.cfg.tip_over_threshold) | \
                 (obj_euler[:, 1].abs() > self.cfg.tip_over_threshold)
        tip_pen = torch.where(tipped,
                              torch.tensor(self.cfg.tip_penalty, device=self.device),
                              torch.zeros_like(reward))
        reward = reward + tip_pen

        self._gave_completion[self._gave_completion | new_completion] = True
        self._gave_rot_bonus[self._gave_rot_bonus | new_rot_bonus] = True
        self.at_goal_pos = pos_ok
        self.at_goal_both = both_ok

        self._last_pos_err = d_now
        self._last_rot_err = y_now
        self._last_pos_imp = pos_imp
        self._last_rot_imp = rot_imp
        self._last_penalty = penalty
        self._last_completion = completion + rot_bonus + tip_pen

        self.push_count += 1

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self._get_obj_pos_local()
        obj_euler = _quat_to_euler_xyz(self._target_object.data.root_quat_w)

        max_pushes = self.push_count >= self.cfg.max_pushes_per_episode

        obj_z = obj_pos[:, 2]
        launched = obj_z > 0.15

        tipped = (obj_euler[:, 0].abs() > self.cfg.tip_over_threshold) | \
                 (obj_euler[:, 1].abs() > self.cfg.tip_over_threshold)

        oob_x = (obj_pos[:, 0] < self.cfg.ws_x[0]) | (obj_pos[:, 0] > self.cfg.ws_x[1])
        oob_y = (obj_pos[:, 1] < self.cfg.ws_y[0]) | (obj_pos[:, 1] > self.cfg.ws_y[1])
        oob = oob_x | oob_y

        at_goal = self.at_goal_pos

        terminated = max_pushes | at_goal | launched | tipped | oob
        self._terminated_this_step = terminated.clone()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return

        N = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

        ox = torch.empty(N, device=self.device).uniform_(*self.cfg.spawn_x_range)
        oy = torch.empty(N, device=self.device).uniform_(*self.cfg.spawn_y_range)
        oz = torch.full((N,), self.cfg.spawn_z, device=self.device)
        oyaw = torch.empty(N, device=self.device).uniform_(0, 2 * torch.pi)
        oeuler = torch.zeros(N, 3, device=self.device)
        oeuler[:, 2] = oyaw

        pos_local = torch.stack([ox, oy, oz], dim=-1)
        pos_world = pos_local + origins
        quat = _euler_to_quat(oeuler)
        pose_7d = torch.cat([pos_world, quat], dim=-1)
        self._target_object.write_root_pose_to_sim(pose_7d, env_ids=env_ids)

        obj_pos_local = pos_local.clone()
        obj_xy = obj_pos_local[:, :2]

        gx = torch.empty(N, device=self.device).uniform_(*self.cfg.goal_x_range)
        gy = torch.empty(N, device=self.device).uniform_(*self.cfg.goal_y_range)
        gz = torch.full((N,), self.cfg.goal_z, device=self.device)
        geuler = torch.zeros(N, 3, device=self.device)
        geuler[:, 2] = torch.empty(N, device=self.device).uniform_(0, 2 * torch.pi)

        for _ in range(10):
            goal_xy = torch.stack([gx, gy], dim=-1)
            dist = (goal_xy - obj_xy).norm(dim=-1)
            bad = (dist < self.cfg.goal_min_dist) | (dist > self.cfg.goal_max_dist)
            if not bad.any():
                break
            bc = int(bad.sum().item())
            gx[bad] = torch.empty(bc, device=self.device).uniform_(*self.cfg.goal_x_range)
            gy[bad] = torch.empty(bc, device=self.device).uniform_(*self.cfg.goal_y_range)

        self.goal_pos_euler[env_ids] = torch.cat([
            gx.unsqueeze(-1), gy.unsqueeze(-1), gz.unsqueeze(-1), geuler,
        ], dim=-1)

        self.push_count[env_ids] = 0
        self.at_goal_pos[env_ids] = False
        self.at_goal_both[env_ids] = False
        self._gave_completion[env_ids] = False
        self._gave_rot_bonus[env_ids] = False
        self._ep_started[env_ids] = False
        self._episode_reward[env_ids] = 0.0
        self._step_idx[env_ids] = 0
        self._joint_cmd[env_ids] = self._robot.data.joint_pos[:, self._arm_ids][env_ids]
        self._terminated_this_step[env_ids] = False

    def _get_obj_pos_local(self) -> torch.Tensor:
        return self._target_object.data.root_pos_w[:, :3] - self.scene.env_origins

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        infos: list[dict],
    ) -> np.ndarray:
        N = len(achieved_goal)
        result = np.zeros(N, dtype=np.float32)

        for i in range(N):
            ag = achieved_goal[i]
            dg = desired_goal[i]
            info = infos[i]

            d_now = np.sqrt((ag[0] - dg[0]) ** 2 + (ag[1] - dg[1]) ** 2)
            y_now = abs(ag[2] - dg[2])
            while y_now > math.pi:
                y_now = abs(2 * math.pi - y_now)

            prev_ag = info.get("prev_achieved_goal", ag)
            d_prev = np.sqrt((prev_ag[0] - dg[0]) ** 2 + (prev_ag[1] - dg[1]) ** 2)
            y_prev = abs(prev_ag[2] - dg[2])
            while y_prev > math.pi:
                y_prev = abs(2 * math.pi - y_prev)

            alpha = self.cfg.dense_alpha
            pos_imp = alpha * (d_prev - d_now) / max(d_prev, 0.01)
            rot_imp = alpha * (y_prev - y_now) / max(y_prev, 0.01)
            penalty = -self.cfg.dense_beta * d_now
            rot_penalty = -self.cfg.dense_rot_beta * y_now

            r = pos_imp + rot_imp + penalty + rot_penalty

            if d_now < self.cfg.push_success_threshold_pos:
                r += self.cfg.completion_bonus
            if d_now < self.cfg.push_success_threshold_pos and y_now < self.cfg.push_success_threshold_rot:
                r += self.cfg.rotation_sub_bonus

            result[i] = float(r)

        return result

    def close(self):
        super().close()
