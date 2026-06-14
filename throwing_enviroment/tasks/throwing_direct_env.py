"""DirectRLEnv for throw primitive — no ManagerBasedRLEnv overhead.

State-machine inside _apply_action() executes the full throw primitive:
  STABILIZE → GO_TO_INIT → GO_TO_INITIAL → THROW → FLIGHT

One outer step() = one complete throw. The drink is kinematically held
from start until release (no IK needed — uses cached crane pose + offsets).
"""

from __future__ import annotations

import math
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate

from .throwing_direct_env_cfg import (
    ThrowingDirectEnvCfg,
    PHASE_STABILIZE,
    PHASE_GO_TO_INIT,
    PHASE_GO_TO_INITIAL,
    PHASE_THROW_MAX,
    PHASE_FLIGHT,
    TOTAL_DECIMATION,
    DRINK_HOLD_Z_OFFSET,
    DRINK_BELOW_TABLE_Z,
    TABLE_Z,
)
from .throw_primitive import (
    RIGHT_INIT_JOINTS,
    RIGHT_END_JOINTS,
    LEFT_INIT_JOINTS,
    LEFT_END_JOINTS,
    map_action_to_params,
    build_joint_targets,
    compute_phase_boundaries,
    DROP_PENALTY_DISTANCE,
)
from .events import _set_gripper_state

OBS_MAX_NORM = 3.0
EE_YAW_OFFSET = math.pi / 2


class ThrowingDirectEnv(DirectRLEnv):
    """High-performance DirectRLEnv for throw primitive training."""

    cfg: ThrowingDirectEnvCfg

    def __init__(self, cfg: ThrowingDirectEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._side = cfg.playing_arm_side
        self._ee_body_name = f"{self._side}_wrist_3_link"

        if self._side == "right":
            arm_patterns = ["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"]
            self._init_joints = torch.tensor(RIGHT_INIT_JOINTS, device=self.device)
            self._end_joints_base = torch.tensor(RIGHT_END_JOINTS, device=self.device)
            self._gripper_patterns = [
                "rgripper_finger_joint",
                "rgripper_.*_knuckle_joint$",
                "rgripper_.*_inner_finger_joint$",
            ]
        else:
            arm_patterns = ["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"]
            self._init_joints = torch.tensor(LEFT_INIT_JOINTS, device=self.device)
            self._end_joints_base = torch.tensor(LEFT_END_JOINTS, device=self.device)
            self._gripper_patterns = [
                "lgripper_finger_joint",
                "lgripper_.*_knuckle_joint$",
                "lgripper_.*_inner_finger_joint$",
            ]

        robot = self.scene["robot"]
        self._arm_ids, _ = robot.find_joints(arm_patterns)
        self._ee_body_ids, _ = robot.find_bodies([self._ee_body_name])
        self._gripper_ids, _ = robot.find_joints(self._gripper_patterns)

        self._init_joints_rotated = self._init_joints.clone()
        self._init_joints_rotated[5] += EE_YAW_OFFSET

        N = self.num_envs
        self._sub_step = torch.zeros(N, device=self.device, dtype=torch.long)
        self._initial_joints_pose = torch.zeros(N, 6, device=self.device)
        self._end_joints_pose = torch.zeros(N, 6, device=self.device)
        self._throw_steps = torch.zeros(N, device=self.device, dtype=torch.long)
        self._release_step = torch.zeros(N, device=self.device, dtype=torch.long)
        self._released = torch.zeros(N, device=self.device, dtype=torch.bool)
        self._dropped = torch.zeros(N, device=self.device, dtype=torch.bool)
        self._grasp_offset = torch.zeros(N, 3, device=self.device)
        self._grasp_quat = torch.zeros(N, 4, device=self.device)

        self._phase_go_init_start = PHASE_STABILIZE
        self._phase_go_initial_start = PHASE_STABILIZE + PHASE_GO_TO_INIT
        self._phase_throw_start = PHASE_STABILIZE + PHASE_GO_TO_INIT + PHASE_GO_TO_INITIAL

        self._robot_obs_val = -1.0 if self._side == "left" else 1.0
        self._episode_count = 0
        self._last_distances = torch.zeros(N, device=self.device)
        self._last_milk_pos = torch.zeros(N, 3, device=self.device)
        self._last_target_pos = torch.zeros(N, 3, device=self.device)

    def _setup_scene(self):
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._sub_step[:] = 0
        self._released[:] = False
        self._dropped[:] = False

        actions = actions.clamp(-1.0, 1.0)
        actions = torch.nan_to_num(actions, nan=0.0)

        params = map_action_to_params(actions, side=self._side)
        initial_jv = params[:, 0]
        final_jv = params[:, 1]
        releasing_time = params[:, 2].clamp(0.05, 1.0)
        duration = params[:, 3].clamp(0.1, 1.0)

        boundaries = compute_phase_boundaries(duration, releasing_time, self.physics_dt)
        self._throw_steps = boundaries["throw_steps"]
        self._release_step = boundaries["release_step"]

        self._initial_joints_pose, self._end_joints_pose = build_joint_targets(
            self._init_joints, initial_jv, self._end_joints_base, final_jv
        )
        self._initial_joints_pose[:, 5] += EE_YAW_OFFSET
        self._end_joints_pose[:, 5] += EE_YAW_OFFSET

    def _apply_action(self):
        robot = self.scene["robot"]
        milk = self.scene["milk"]
        N = self.num_envs
        step = self._sub_step
        all_ids = torch.arange(N, device=self.device)

        in_stabilize = step < self._phase_go_init_start
        in_go_init = (step >= self._phase_go_init_start) & (step < self._phase_go_initial_start)
        in_go_initial = (step >= self._phase_go_initial_start) & (step < self._phase_throw_start)
        in_throw = (step >= self._phase_throw_start) & (step < self._phase_throw_start + self._throw_steps)
        in_flight = step >= (self._phase_throw_start + self._throw_steps)

        targets = torch.zeros(N, 6, device=self.device)

        crane_joints = robot.data.default_joint_pos[:, self._arm_ids]

        if in_stabilize.any():
            targets[in_stabilize] = crane_joints[in_stabilize]

        if in_go_init.any():
            targets[in_go_init] = self._init_joints_rotated.unsqueeze(0).expand(in_go_init.sum(), -1)

        if in_go_initial.any():
            targets[in_go_initial] = self._initial_joints_pose[in_go_initial]

        if in_throw.any():
            throw_idx = in_throw.nonzero(as_tuple=True)[0]
            local_step = step[throw_idx] - self._phase_throw_start
            throw_len = self._throw_steps[throw_idx].float().clamp(min=1)
            t = (local_step.float() / (throw_len - 1).clamp(min=1)).clamp(0, 1).unsqueeze(-1)
            targets[throw_idx] = (
                self._initial_joints_pose[throw_idx] * (1.0 - t)
                + self._end_joints_pose[throw_idx] * t
            )

            should_release = in_throw & (step >= self._phase_throw_start + self._release_step) & ~self._released
            if should_release.any():
                release_ids = should_release.nonzero(as_tuple=True)[0]
                _set_gripper_state(robot, 0.0, release_ids)
                self._released[should_release] = True
                ee_lin_vel = robot.data.body_lin_vel_w[release_ids, self._ee_body_ids[0]]
                ee_ang_vel = robot.data.body_ang_vel_w[release_ids, self._ee_body_ids[0]]
                ee_lin_vel = torch.nan_to_num(ee_lin_vel, nan=0.0).clamp(-20.0, 20.0)
                ee_ang_vel = torch.nan_to_num(ee_ang_vel, nan=0.0).clamp(-50.0, 50.0)
                release_vel = torch.cat([ee_lin_vel, ee_ang_vel], dim=-1)
                vel_norm = torch.norm(ee_lin_vel, dim=-1, keepdim=True).clamp(min=0.1)
                vel_dir = ee_lin_vel / vel_norm
                release_pos = milk.data.root_pos_w[release_ids, :3] + vel_dir * 0.10
                release_pos = torch.nan_to_num(release_pos, nan=0.0)
                release_quat = milk.data.root_quat_w[release_ids]
                milk.write_root_pose_to_sim(
                    torch.cat([release_pos, release_quat], dim=-1), env_ids=release_ids
                )
                milk.write_root_velocity_to_sim(release_vel, env_ids=release_ids)

        if in_flight.any():
            targets[in_flight] = self._end_joints_pose[in_flight]

        robot.set_joint_position_target(targets, joint_ids=self._arm_ids)

        not_released = ~self._released
        if not_released.any():
            nr_ids = not_released.nonzero(as_tuple=True)[0]
            _set_gripper_state(robot, self.cfg.grasp_strength, nr_ids)

        if not_released.any() & (~self._dropped).any():
            hold_mask = not_released & ~self._dropped
            if hold_mask.any():
                hold_ids = hold_mask.nonzero(as_tuple=True)[0]
                ee_pos = robot.data.body_pos_w[hold_ids, self._ee_body_ids[0]]
                if not torch.isnan(ee_pos).any():
                    pos = ee_pos + self._grasp_offset[hold_ids]
                    pose = torch.cat([pos, self._grasp_quat[hold_ids]], dim=-1)
                    milk.write_root_pose_to_sim(pose, env_ids=hold_ids)
                    milk.write_root_velocity_to_sim(
                        torch.zeros(len(hold_ids), 6, device=self.device), env_ids=hold_ids
                    )

        milk_z = milk.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        new_drop = (milk_z < DRINK_BELOW_TABLE_Z) & ~self._dropped & self._released
        self._dropped |= new_drop

        self._sub_step += 1

    def _get_observations(self) -> dict:
        milk = self.scene["milk"]
        target = self.scene["target"]
        origins = self.scene.env_origins

        milk_pos = milk.data.root_pos_w[:, :3] - origins
        target_pos = target.data.root_pos_w[:, :3] - origins

        dist_vec = milk_pos - target_pos
        dist = torch.norm(dist_vec, dim=-1, keepdim=True)
        dist_x = torch.abs(dist_vec[:, 0:1])
        dist_y = torch.abs(dist_vec[:, 1:2])

        robot_ind = torch.full((self.num_envs, 1), self._robot_obs_val, device=self.device)

        obs = torch.cat([
            robot_ind,
            target_pos[:, 0:1] / OBS_MAX_NORM,
            target_pos[:, 1:2] / OBS_MAX_NORM,
            milk_pos[:, 0:1] / OBS_MAX_NORM,
            milk_pos[:, 1:2] / OBS_MAX_NORM,
            dist / OBS_MAX_NORM,
            dist_x / OBS_MAX_NORM,
            dist_y / OBS_MAX_NORM,
        ], dim=-1)

        return {"policy": obs, "critic": obs}

    def _get_rewards(self) -> torch.Tensor:
        milk = self.scene["milk"]
        target = self.scene["target"]

        milk_pos = milk.data.root_pos_w[:, :3]
        target_pos = target.data.root_pos_w[:, :3]
        dist = torch.norm(milk_pos - target_pos, dim=-1)

        nan_mask = torch.isnan(dist)
        dist = torch.nan_to_num(dist, nan=10.0)

        origins = self.scene.env_origins
        self._last_distances = dist.clone()
        self._last_milk_pos = torch.nan_to_num(milk_pos - origins, nan=0.0).clone()
        self._last_target_pos = torch.nan_to_num(target_pos - origins, nan=0.0).clone()

        alpha = 0.9
        reward = (
            alpha * torch.exp(-(dist ** 2) / 0.1)
            + (1.0 - alpha) * torch.exp(-(dist ** 2) / 0.5)
            + 0.5 * torch.clamp(1.0 - dist, min=0.0)
        )
        reward[dist < self.cfg.success_threshold] = 2.0
        reward[self._dropped] = 0.0
        reward[nan_mask] = 0.0

        self._episode_count += self.num_envs
        if self._episode_count % (self.num_envs * 10) == 0:
            mean_d = dist[~self._dropped].mean().item() if (~self._dropped).any() else float("inf")
            n_suc = (dist < self.cfg.success_threshold).sum().item()
            n_drop = self._dropped.sum().item()
            print(
                f"[Ep {self._episode_count:>7}] "
                f"reward={reward.mean().item():.4f}  dist={mean_d:.3f}m  "
                f"success={n_suc}/{self.num_envs}  dropped={n_drop}/{self.num_envs}"
            )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        time_outs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return terminated, time_outs

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        robot = self.scene["robot"]
        milk = self.scene["milk"]
        target = self.scene["target"]
        N = len(env_ids)

        robot.write_joint_state_to_sim(
            position=robot.data.default_joint_pos[env_ids],
            velocity=robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
        robot.set_joint_position_target(
            robot.data.default_joint_pos[env_ids], env_ids=env_ids,
        )

        origins = self.scene.env_origins[env_ids]
        tx = torch.empty(N, device=self.device).uniform_(*self.cfg.target_x_range)
        ty = torch.empty(N, device=self.device).uniform_(*self.cfg.target_y_range)
        tz = torch.full((N,), self.cfg.target_z, device=self.device)
        target_pos = torch.stack([tx + origins[:, 0], ty + origins[:, 1], tz + origins[:, 2]], dim=-1)
        target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(N, -1)
        target.write_root_pose_to_sim(torch.cat([target_pos, target_quat], dim=-1), env_ids=env_ids)

        ee_pos = robot.data.body_pos_w[env_ids, self._ee_body_ids[0]]
        ee_quat = robot.data.body_quat_w[env_ids, self._ee_body_ids[0]]
        drink_pos = ee_pos.clone()
        drink_pos[:, 2] += DRINK_HOLD_Z_OFFSET
        drink_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(N, -1)
        milk.write_root_pose_to_sim(torch.cat([drink_pos, drink_quat], dim=-1), env_ids=env_ids)
        milk.write_root_velocity_to_sim(torch.zeros(N, 6, device=self.device), env_ids=env_ids)

        self._grasp_offset[env_ids] = drink_pos - ee_pos
        self._grasp_quat[env_ids] = drink_quat

        self._sub_step[env_ids] = 0
        self._released[env_ids] = False
        self._dropped[env_ids] = False
