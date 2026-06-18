"""
Push-primitive ASP environment wrapper.

Combines the Asymmetric Self-Play (ASP) two-phase structure (Alice proposes
goals, Bob solves them) with push-primitive macro-actions.  Unlike the per-step
EE-delta variant (wrapper.py), this wrapper treats each push as a discrete
macro-action.  Alice gets `alice_pushes` pushes to construct a goal by moving
objects; Bob gets `bob_pushes` pushes to match the objects to their goal
positions and orientations.

SPARSE REWARDS ONLY (integer-valued, per ASP paper philosophy):
  - Alice: +1 valid goal, -1 shallow goal, -3 invalid/off-table, +5 Bob fails,
           -1 Bob succeeds.  All paid at Alice phase end.
  - Bob:   +1 per object entering goal threshold, -1 leaving, +5 completion,
           + phase-end progress reward.  Paid per push (sparse gates, no dense
           distance penalty).

Key differences from the per-step ASP wrapper (wrapper.py):
  - Actions are push parameters (6D), not EE deltas
  - Phase length is measured in pushes, not physics steps
  - Bob's per-step sparse reward is computed per PUSH (not per physics step)
  - Observations use the push_task_curobo layout (29D base, sliced to 21D/29D)
"""

import torch
import numpy as np
import math
from typing import Optional, Dict, Any, Tuple, Union
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

from ...utils.episode_manager import EpisodeManager, Phase
from ...utils.goal_validator import validate_goal
from . import rewards as reward_utils
from .events import reset_objects_to_random_safe_pose, reset_robot_joints
from .reward_pbrs import compute_dpose

# Observation layout indices (matches push_task_curobo.py, Fix P39: no gripper):
# [ee_pose(6) | obj_state(14) | goal_pose(6) | goal_dist(2)] = 28D total
# When rel_obs=True, goal_dist(2) is replaced with [rel_dx, rel_dy] (same dimension)
_OBS_ROBOT_DIM = 6
_OBS_OBJ_STATE_DIM = 14
_OBS_GOAL_DIM = 6
_OBS_DIST_DIM = 2
_OBS_DIM = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + _OBS_GOAL_DIM + _OBS_DIST_DIM  # 28

# Success thresholds (matching wrapper.py and train_curobo.py)
POS_THRESHOLD = 0.05
ROT_THRESHOLD = 0.2


def _rot_distance_rad(euler_a: torch.Tensor, euler_b: torch.Tensor) -> torch.Tensor:
    """Maximum absolute Euler-angle difference with wraparound (range [0, pi])."""
    diff = (euler_a - euler_b) % (2.0 * torch.pi)
    diff = torch.where(diff > torch.pi, 2.0 * torch.pi - diff, diff)
    return diff.max(dim=-1)[0]


def _yaw_distance_rad(euler_a: torch.Tensor, euler_b: torch.Tensor) -> torch.Tensor:
    """Yaw-only Euler-angle difference with wraparound (range [0, pi])."""
    yaw_a = euler_a[..., 2]
    yaw_b = euler_b[..., 2]
    diff = (yaw_a - yaw_b) % (2.0 * torch.pi)
    diff = torch.where(diff > torch.pi, 2.0 * torch.pi - diff, diff)
    return diff


class PushASPEnvWrapper:
    """
    Wrapper for push-primitive Asymmetric Self-Play.

    Manages:
      - Asymmetric observation spaces (Alice: 21D, Bob: 29D)
      - Two-phase ASP structure (Alice proposes, Bob solves)
      - Sparse reward computation (per push for Bob, per phase for Alice)
      - Goal state capture and validation
      - Phase transitions
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        alice_pushes: int = 5,
        bob_pushes: int = 10,
        max_goals_per_episode: int = 3,
        num_objects: int = 1,
        rel_obs: bool = False,
        dpose_obs: bool = False,
        char_length: float = 0.0,
        dpose_threshold: float = 0.06,
        obj_spawn_z: float = 0.05,
        obj_settled_z: float = 0.023,
        device: str = "cuda",
    ):
        self.env = env
        self.device = device
        self.num_objects = num_objects
        self.alice_pushes = alice_pushes
        self.bob_pushes = bob_pushes
        self.max_goals_per_episode = max_goals_per_episode
        self.rel_obs = rel_obs
        self.dpose_obs = dpose_obs
        self.char_length = char_length
        self.dpose_threshold = dpose_threshold
        self.obj_spawn_z = obj_spawn_z
        self.obj_settled_z = obj_settled_z

        self.robot_dim = _OBS_ROBOT_DIM
        self.obj_state_dim = _OBS_OBJ_STATE_DIM
        self.goal_dim = _OBS_GOAL_DIM
        self.dist_dim = _OBS_DIST_DIM

        # Alice obs: robot(6) + obj_state(14) = 20D (no goal info)
        self.alice_obs_dim = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM
        # Bob obs: 28D — when rel_obs, goal_dist(2) holds rel_dx/rel_dy instead
        self.bob_obs_dim = _OBS_DIM
        self.push_obs_dim = self.bob_obs_dim

        self.alice_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.alice_obs_dim,), dtype=np.float32,
        )
        self.bob_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.bob_obs_dim,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.bob_obs_dim,), dtype=np.float32,
        )
        self.state_space = self.observation_space

        if len(env.action_space.shape) > 1:
            self.action_space = gym.spaces.Box(
                low=env.action_space.low[0],
                high=env.action_space.high[0],
                shape=env.action_space.shape[1:],
                dtype=env.action_space.dtype,
            )
        else:
            self.action_space = env.action_space

        # Episode manager tracks phases, goals, success per env
        self.episode_manager = EpisodeManager(
            num_envs=env.num_envs,
            device=device,
            alice_timesteps=alice_pushes,
            bob_timesteps=bob_pushes,
            max_goals_per_episode=max_goals_per_episode,
        )
        self.env.episode_manager = self.episode_manager

        # Table bounds for goal validation
        self.table_bounds = {
            "x_range": (-0.70, 0.70),
            "y_range": (-0.10, 0.90),
            "z_min": -0.2,
            "z_max": 0.15,
        }
        self.placement_bounds = {
            "x_range": (-0.50, 0.50),
            "y_range": (0.25, 0.70),
        }

        # Alice reward accumulator (paid at phase end)
        self.delayed_alice_reward = torch.zeros(env.num_envs, device=device)

        # Bob phase-end progress reward state
        self.bob_init_pos_err = torch.zeros(env.num_envs, device=device)
        self.bob_init_rot_err = torch.zeros(env.num_envs, device=device)
        self.bob_init_dpose = torch.zeros(env.num_envs, device=device)
        self._bob_progress_captured = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device,
        )

        # Bob pre-push state snapshots for sparse-reward gating
        self._bob_pre_push_pos_err = torch.zeros(env.num_envs, device=device)
        self._bob_pre_push_rot_err = torch.zeros(env.num_envs, device=device)
        self._bob_at_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self._bob_gave_completion = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device,
        )

        # Safe reset state (local frame)
        _id_euler = torch.tensor([0.0, 0.0, 0.0], device=device)
        _t_pos = torch.tensor([0.0, 0.5, obj_settled_z], device=device)
        if num_objects == 1:
            self._safe_reset_state = torch.cat([_t_pos, _id_euler])
        else:
            _t2_pos = torch.tensor([-0.25, 0.7, 0.023], device=device)
            self._safe_reset_state = torch.cat([_t_pos, _id_euler, _t2_pos, _id_euler])

        # Iteration stats
        self._iter_stats = self._make_iter_stats()

        # Push counter (per env, counts within current phase)
        self.push_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        # Object pose snapshots for reward computation
        self.prev_obj_pos = torch.zeros(env.num_envs, 3, device=device)
        self.prev_obj_euler = torch.zeros(env.num_envs, 3, device=device)

        print(f"[PushASP Wrapper] Initialized: Alice {alice_pushes} pushes, "
              f"Bob {bob_pushes} pushes, max_goals={max_goals_per_episode}")
        _obs_tag = " (dpose L={:.3f} thr={:.3f})".format(char_length, dpose_threshold) if dpose_obs else \
                   " (rel_obs)" if rel_obs else ""
        print(f"  Alice obs: {self.alice_obs_dim}D, Bob obs: {self.bob_obs_dim}D{_obs_tag}")

        # Tight spawn bounds (10cm × 10cm box) + random yaw to limit OOB
        self.spawn_x_range = (-0.04, 0.04)
        self.spawn_y_range = (0.4, 0.45)

    def _rand_reset_objs(self, env_ids: torch.Tensor) -> dict:
        """Reset objects with tight spawn bounds and random yaw."""
        return reset_objects_to_random_safe_pose(
            self.env, env_ids,
            x_range=self.spawn_x_range,
            y_range=self.spawn_y_range,
            random_yaw=True,
            spawn_z=self.obj_spawn_z,
            settled_z=self.obj_settled_z,
        )

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def _make_iter_stats(self) -> dict:
        return {
            "invalid_goals": 0,
            "valid_goals": 0,
            "bob_successes": 0,
            "bob_failures": 0,
            "terminations": {},
            "alice_total": 0,
            "alice_disp_3d_sum": 0.0,
            "alice_not_moved": 0,
        }

    def reset_iter_stats(self):
        self._iter_stats = self._make_iter_stats()

    def get_iter_stats(self) -> dict:
        return dict(self._iter_stats)

    @property
    def num_envs(self):
        return self.env.num_envs

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_push_obs(self) -> torch.Tensor:
        """Get the full push-policy observation. When dpose_obs=True, replaces
        goal_dist(2) with [d_pose, bearing]. When rel_obs=True (and not dpose),
        replaces goal_dist(2) with [rel_dx, rel_dy]."""
        self._update_goal_in_extras()
        obs_dict = self.env.observation_manager.compute()
        obs = obs_dict["push_policy"]
        dist_idx = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + _OBS_GOAL_DIM
        if self.dpose_obs:
            obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
            obj_yaw = obs[:, _OBS_ROBOT_DIM + 5]
            goal_pos = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM:
                           _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
            goal_yaw = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 5]
            d_pose = compute_dpose(obj_pos, goal_pos, obj_yaw, goal_yaw, self.char_length)
            dx = goal_pos[:, 0] - obj_pos[:, 0]
            dy = goal_pos[:, 1] - obj_pos[:, 1]
            bearing = torch.atan2(dy, dx)
            obs[:, dist_idx] = d_pose
            obs[:, dist_idx + 1] = bearing
        elif self.rel_obs:
            obj_pos = obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]
            goal_pos = obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM:
                           _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]
            rel_dx = goal_pos[:, 0:1] - obj_pos[:, 0:1]
            rel_dy = goal_pos[:, 1:2] - obj_pos[:, 1:2]
            obs[:, dist_idx:dist_idx + 1] = rel_dx
            obs[:, dist_idx + 1:dist_idx + 2] = rel_dy
        return obs

    def _get_alice_obs(self, push_obs: torch.Tensor) -> torch.Tensor:
        """Alice sees robot state + object state only (no goal info)."""
        return push_obs[:, :self.alice_obs_dim]

    def _get_bob_obs(self, push_obs: torch.Tensor) -> torch.Tensor:
        """Bob sees the full observation including goal info."""
        return push_obs

    def _get_obj_pos(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract first object's position from push observation."""
        return obs[:, _OBS_ROBOT_DIM: _OBS_ROBOT_DIM + 3]

    def _get_obj_euler(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract first object's Euler angles from push observation."""
        return obs[:, _OBS_ROBOT_DIM + 3: _OBS_ROBOT_DIM + 6]

    def _get_goal_pos(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract first object's goal position."""
        return obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM:
                   _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]

    def _get_goal_euler(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract first object's goal Euler angles."""
        return obs[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3:
                   _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 6]

    def capture_pre_push(self, obs: torch.Tensor):
        """Snapshot object pose before a push for improvement/reward computation."""
        self.prev_obj_pos = self._get_obj_pos(obs).clone()
        self.prev_obj_euler = self._get_obj_euler(obs).clone()

    def _build_full_push_obs(self, obs_dict: dict) -> torch.Tensor:
        """Build the full push observation and update goal distances."""
        self._update_goal_in_extras()
        obs_dict = self.env.observation_manager.compute()
        return obs_dict["push_policy"]

    # ------------------------------------------------------------------
    # Environment interaction
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Reset all environments and episode manager."""
        obs_dict, info = self.env.reset(seed=seed, options=options)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.episode_manager.reset_episode(env_ids, reason="Global Manual Reset")
        self.delayed_alice_reward[env_ids] = 0.0
        self.push_count[env_ids] = 0
        self._bob_progress_captured[env_ids] = False
        self._bob_at_goal[env_ids] = False
        self._bob_gave_completion[env_ids] = False

        spawn_info = self._rand_reset_objs(env_ids)
        reset_robot_joints(self.env, env_ids)
        self.env.scene.write_data_to_sim()
        self.episode_manager.initial_states = self._initial_states_from_spawn(
            spawn_info, self.num_envs,
        )

        self.set_table_color(env_ids, (0.8, 0.1, 0.1))  # Red = Alice

        obs = self._get_push_obs()
        self.capture_pre_push(obs)
        return obs

    def step(self, action: torch.Tensor):
        """
        Single physics step (for compatibility).  Push reward is computed
        separately via compute_bob_push_reward() after the full trajectory.
        Returns zero reward by default.
        """
        obs_dict, _, terminated, truncated, _ = self.env.step(action)
        obs = obs_dict["push_policy"]
        return obs, torch.zeros(self.num_envs, device=self.device), terminated, truncated, {}

    def _initial_states_from_spawn(self, spawn_info: dict, n: int) -> torch.Tensor:
        """Build initial_states tensor from _rand_reset_objs output."""
        t_local = spawn_info.get("target_local", torch.zeros(n, 3, device=self.device))
        t_yaw = spawn_info.get("target_yaw", torch.zeros(n, device=self.device))
        _id_euler = torch.zeros(n, 3, device=self.device)
        _id_euler[:, 2] = t_yaw
        if self.num_objects == 1:
            return torch.cat([t_local, _id_euler], dim=-1)
        t2_local = spawn_info.get("cube_local")
        if t2_local is None:
            t2_local = t_local.clone()
            t2_local[:, 0] -= 0.10
        return torch.cat([t_local, _id_euler, t2_local, _id_euler], dim=-1)

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def _update_goal_in_extras(self):
        """Write goal tensor into env.extras so observation functions can read it."""
        if not hasattr(self.env, "extras"):
            self.env.extras = {}
        gs = self.episode_manager.goal_states
        if gs is not None:
            self.env.extras["goal_state"] = gs
        else:
            self.env.extras["goal_state"] = torch.zeros(
                self.num_envs, self.num_objects * 6, device=self.device,
            )

    def set_table_color(self, env_ids: torch.Tensor, color: Tuple[float, float, float]):
        """Update table diffuse color for specific environments (non-headless only)."""
        if not self.env.sim.has_gui():
            return
        import omni.usd
        from pxr import Gf, Usd, UsdShade
        stage = omni.usd.get_context().get_stage()
        color_vec = Gf.Vec3f(color[0], color[1], color[2])
        for i in env_ids.tolist():
            shader_path = f"/World/envs/env_{i}/Table/VisualMaterial/Shader"
            prim = stage.GetPrimAtPath(shader_path)
            if prim.IsValid():
                attr = prim.GetAttribute("inputs:diffuseColor")
                if attr.IsValid():
                    attr.Set(color_vec)
                    continue
            fallback_path = f"/World/envs/env_{i}/Table/Visuals/mesh/material/Shader"
            prim = stage.GetPrimAtPath(fallback_path)
            if prim.IsValid():
                attr = prim.GetAttribute("inputs:diffuseColor")
                if attr.IsValid():
                    attr.Set(color_vec)
                    continue
            table_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Table")
            if table_prim.IsValid():
                for p in Usd.PrimRange(table_prim):
                    if p.IsA(UsdShade.Shader):
                        attr = p.GetAttribute("inputs:diffuseColor")
                        if attr.IsValid():
                            attr.Set(Gf.Vec3f(color[0], color[1], color[2]))
                            break

    def hide_goal_ghost(self, env_ids: torch.Tensor):
        """Hide goal marker by moving it under the table."""
        if "goal_marker" in self.env.scene.rigid_objects:
            N = len(env_ids)
            hide_pos = torch.cat([
                self.env.scene.env_origins[env_ids] +
                torch.tensor([0.0, 0.0, -1.0], device=self.device),
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(N, -1),
            ], dim=-1)
            self.env.scene["goal_marker"].write_root_pose_to_sim(hide_pos, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Push reward computation for Bob (sparse, per push)
    # ------------------------------------------------------------------

    def compute_bob_push_reward(self, push_obs: torch.Tensor) -> torch.Tensor:
        """
        Sparse reward after a Bob push macro-action completes.

        Reward gates (per ASP paper):
          +1: object enters goal threshold (position < 0.05m AND rotation < 0.2rad)
          -1: object leaves goal threshold
          +5: completion bonus (first time all objects at goal simultaneously)

        Returns (num_envs,) reward tensor.
        """
        rewards = torch.zeros(self.num_envs, device=self.device)

        cur_obj_pos = self._get_obj_pos(push_obs)
        cur_obj_euler = self._get_obj_euler(push_obs)
        goal_pos = self._get_goal_pos(push_obs)
        goal_euler = self._get_goal_euler(push_obs)

        pos_err = (cur_obj_pos - goal_pos).norm(dim=-1)
        rot_err = _rot_distance_rad(cur_obj_euler, goal_euler)

        at_goal = (pos_err < POS_THRESHOLD) & (rot_err < ROT_THRESHOLD)

        newly_entered = at_goal & ~self._bob_at_goal
        newly_left = ~at_goal & self._bob_at_goal

        rewards += newly_entered.float() * 1.0   # +1 entering goal
        rewards -= newly_left.float() * 1.0       # -1 leaving goal

        new_completion = at_goal & ~self._bob_gave_completion
        completion_bonus = new_completion.float() * 5.0
        rewards += completion_bonus

        self._bob_gave_completion |= new_completion
        self._bob_at_goal = at_goal
        self._bob_pre_push_pos_err = pos_err
        self._bob_pre_push_rot_err = rot_err

        return rewards

    def _capture_bob_init_errors(self, env_ids: torch.Tensor):
        """Capture Bob's initial position/rotation errors for phase-end progress."""
        if len(env_ids) == 0:
            return
        obs = self._get_push_obs()
        cur_pos = self._get_obj_pos(obs)[env_ids]
        cur_euler = self._get_obj_euler(obs)[env_ids]
        goal_pos = self._get_goal_pos(obs)[env_ids]
        goal_euler = self._get_goal_euler(obs)[env_ids]

        pos_err = (cur_pos - goal_pos).norm(dim=-1)
        rot_err = _rot_distance_rad(cur_euler, goal_euler)

        self.bob_init_pos_err[env_ids] = pos_err
        self.bob_init_rot_err[env_ids] = rot_err
        if self.dpose_obs:
            self.bob_init_dpose[env_ids] = compute_dpose(
                cur_pos, goal_pos, cur_euler[:, 2], goal_euler[:, 2], self.char_length,
            )
        self._bob_progress_captured[env_ids] = True

    def compute_bob_progress_reward(self, bob_done_ids: torch.Tensor) -> torch.Tensor:
        """
        Phase-end progress reward for Bob envs that just completed.
        When dpose_obs=True, uses single d_pose metric.
        Otherwise: r_progress = clamp(w_pos * delta_pos/init_pos + w_rot * delta_rot/init_rot, -1, +1)
        """
        prog_rew = torch.zeros(self.num_envs, device=self.device)

        if len(bob_done_ids) == 0:
            return prog_rew

        obs = self._get_push_obs()
        cur_pos = self._get_obj_pos(obs)[bob_done_ids]
        cur_euler = self._get_obj_euler(obs)[bob_done_ids]
        goal_pos = self._get_goal_pos(obs)[bob_done_ids]
        goal_euler = self._get_goal_euler(obs)[bob_done_ids]

        if self.dpose_obs:
            init_dp = self.bob_init_dpose[bob_done_ids]
            final_dp = compute_dpose(
                cur_pos, goal_pos, cur_euler[:, 2], goal_euler[:, 2], self.char_length,
            )
            r_prog = ((init_dp - final_dp) / (init_dp + 1e-6)).clamp(-1.0, 1.0)
        else:
            w_pos, w_rot = 0.6, 0.4
            init_pos = self.bob_init_pos_err[bob_done_ids]
            init_rot = self.bob_init_rot_err[bob_done_ids]
            final_pos = (cur_pos - goal_pos).norm(dim=-1)
            final_rot = _rot_distance_rad(cur_euler, goal_euler)
            pos_progress = (init_pos - final_pos) / (init_pos + 1e-6)
            rot_progress = (init_rot - final_rot) / (init_rot + 1e-6)
            r_prog = (w_pos * pos_progress + w_rot * rot_progress).clamp(-1.0, 1.0)

        prog_rew[bob_done_ids] = r_prog
        self._bob_progress_captured[bob_done_ids] = False

        return prog_rew

    # ── Dense push improvement reward for Bob (Fix P53 + Fix P63) ──────────
    # Normalised fractional reward: a push that halves the remaining error
    # earns α×0.5 regardless of domain.  One coefficient instead of two.
    _PUSH_DENSE_ALPHA = 3.0     # unitless fractional improvement gain
    _PUSH_DENSE_BETA = 0.5      # distance penalty (urgency)
    _PUSH_DENSE_ROT_BETA = 0.25 # continuous yaw penalty (mirror of positional urgency)

    def compute_bob_dense_push_reward(self, push_obs: torch.Tensor) -> torch.Tensor:
        """
        Dense per-push improvement reward for Bob only.
        R = α·(d_prev−d_now)/d_prev           position improvement (fractional)
          + α·(y_prev−y_now)/y_prev           rotation improvement (fractional)
          − β·d_now                            distance penalty
          − β_rot·y_now                        continuous yaw penalty

        Denominators clamped at 0.01 to avoid division by zero.
        All components clamped to guard against PhysX glitches (Fix P30).
        No completion bonus — that comes from the sparse reward.
        """
        cur_obj_pos = self._get_obj_pos(push_obs)
        cur_obj_euler = self._get_obj_euler(push_obs)
        goal_pos = self._get_goal_pos(push_obs)
        goal_euler = self._get_goal_euler(push_obs)

        d_prev = (self.prev_obj_pos - goal_pos).norm(dim=-1)
        d_now = (cur_obj_pos - goal_pos).norm(dim=-1)
        y_prev = _yaw_distance_rad(self.prev_obj_euler, goal_euler)
        y_now = _yaw_distance_rad(cur_obj_euler, goal_euler)

        pos_imp = (self._PUSH_DENSE_ALPHA * (d_prev - d_now) / d_prev.clamp(min=0.01)).clamp(-5.0, 5.0)
        rot_imp = (self._PUSH_DENSE_ALPHA * (y_prev - y_now) / y_prev.clamp(min=0.01)).clamp(-4.0, 4.0)
        penalty = (-self._PUSH_DENSE_BETA * d_now).clamp(-2.0, 0.0)
        rot_penalty = (-self._PUSH_DENSE_ROT_BETA * y_now).clamp(-1.0, 0.0)

        return pos_imp + rot_imp + penalty + rot_penalty

    # ------------------------------------------------------------------
    # Phase transitions & goal validation
    # ------------------------------------------------------------------

    def _extract_object_states(self, push_obs: torch.Tensor) -> torch.Tensor:
        """Extract object poses from push observation (Euler, local frame).
        Returns [pos(3)+euler(3)] = 6D for 1 object."""
        obj_pos = self._get_obj_pos(push_obs)
        obj_euler = self._get_obj_euler(push_obs)
        return torch.cat([obj_pos, obj_euler], dim=-1)

    def handle_alice_phase_end(
        self,
        env_ids: torch.Tensor,
        push_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle completion of Alice's push phase.
        Returns (valid_ids, invalid_ids)."""
        goal_state = self._extract_object_states(push_obs)
        initial_state = self.episode_manager.initial_states

        active_goal = goal_state[env_ids]
        active_initial = initial_state[env_ids]

        valid, val_reward, reasons = validate_goal(
            active_initial, active_goal,
            self.table_bounds, self.placement_bounds,
            pos_threshold=0.05, rot_threshold=0.25,
            min_meaningful_disp=0.10,
            require_all_moved=(self.num_objects >= 2),
        )

        self.delayed_alice_reward[env_ids] = val_reward

        n = len(env_ids)
        self._iter_stats["valid_goals"] += int(valid.sum().item())
        self._iter_stats["invalid_goals"] += int((~valid).sum().item())
        self._iter_stats["alice_total"] += n
        dist_3d = torch.norm(active_goal[:, 0:3] - active_initial[:, 0:3], dim=-1)
        self._iter_stats["alice_disp_3d_sum"] += dist_3d.sum().item()
        self._iter_stats["alice_not_moved"] += int((dist_3d <= 0.05).sum().item())

        self.episode_manager.store_goal_state(active_goal, env_ids)
        self.episode_manager.mark_goal_valid(env_ids, valid)
        self.episode_manager.mark_alice_base_reward(env_ids, val_reward)

        valid_env_ids = env_ids[valid]
        invalid_env_ids = env_ids[~valid]

        if len(valid_env_ids) > 0:
            self._transition_to_bob(valid_env_ids, push_obs)

        if len(invalid_env_ids) > 0:
            self.episode_manager.reset_episode(invalid_env_ids, reason="Alice Invalid Goal")
            self.hide_goal_ghost(invalid_env_ids)
            _sp = self._rand_reset_objs( invalid_env_ids)
            reset_robot_joints(self.env, invalid_env_ids)
            self.env.scene.write_data_to_sim()
            self.episode_manager.initial_states[invalid_env_ids] = (
                self._initial_states_from_spawn(_sp, len(invalid_env_ids))
            )
            self.set_table_color(invalid_env_ids, (0.8, 0.1, 0.1))  # Red = Alice

        return valid_env_ids, invalid_env_ids

    def _transition_to_bob(self, env_ids: torch.Tensor, push_obs: torch.Tensor):
        """Transition envs from Alice to Bob phase — reset objects, set goal."""
        self.episode_manager.transition_to_bob(env_ids)
        start_states = self.episode_manager.initial_states[env_ids]
        origins = self.env.scene.env_origins[env_ids]

        goal_state = self._extract_object_states(push_obs)

        # Reset target object to start position
        from .observations import _euler_xyz_to_quat
        t_pos_local = start_states[:, 0:3]
        t_quat = _euler_xyz_to_quat(start_states[:, 3:6])
        pos1_global = t_pos_local + origins
        self.env.scene["target_object"].write_root_pose_to_sim(
            torch.cat([pos1_global, t_quat], dim=-1), env_ids=env_ids,
        )

        # Trivial goal check: reject goals where Bob starts within success threshold
        _goal_pos = goal_state[env_ids, 0:3]
        _goal_euler = goal_state[env_ids, 3:6]
        _start_pos = start_states[:, 0:3]
        _start_euler = start_states[:, 3:6]
        if self.dpose_obs:
            _dp = compute_dpose(_start_pos, _goal_pos, _start_euler[:, 2], _goal_euler[:, 2], self.char_length)
            too_easy = _dp < self.dpose_threshold
        else:
            _pos_dist = (_start_pos - _goal_pos).norm(dim=-1)
            _rot_dist = _rot_distance_rad(_start_euler, _goal_euler)
            too_easy = (_pos_dist < POS_THRESHOLD) & (_rot_dist < ROT_THRESHOLD)

        too_easy_ids = env_ids[too_easy]
        valid_env_ids = env_ids[~too_easy]

        if len(too_easy_ids) > 0:
            self.episode_manager.reset_episode(too_easy_ids, reason="Alice Too-Easy Goal")
            self.hide_goal_ghost(too_easy_ids)
            _sp = self._rand_reset_objs( too_easy_ids)
            reset_robot_joints(self.env, too_easy_ids)
            self.env.scene.write_data_to_sim()
            self.episode_manager.initial_states[too_easy_ids] = (
                self._initial_states_from_spawn(_sp, len(too_easy_ids))
            )
            self._iter_stats["valid_goals"] -= len(too_easy_ids)
            self._iter_stats["invalid_goals"] += len(too_easy_ids)
            self.delayed_alice_reward[too_easy_ids] = -3.0
            self.set_table_color(too_easy_ids, (0.8, 0.1, 0.1))  # Red = Alice

        if len(valid_env_ids) > 0:
            # Move goal marker to goal position
            if "goal_marker" in self.env.scene.rigid_objects:
                goal_pos_local = goal_state[valid_env_ids, 0:3].clone()
                goal_pos_local[:, 2] = -0.001
                goal_euler = goal_state[valid_env_ids, 3:6].clone()
                goal_euler[:, 0] = 0.0
                goal_euler[:, 1] = 0.0
                goal_quat = _euler_xyz_to_quat(goal_euler)
                marker_pos_global = goal_pos_local + origins[valid_env_ids]
                self.env.scene["goal_marker"].write_root_pose_to_sim(
                    torch.cat([marker_pos_global, goal_quat], dim=-1),
                    env_ids=valid_env_ids,
                )

            reset_robot_joints(self.env, valid_env_ids)
            self.push_count[valid_env_ids] = 0
            self._bob_at_goal[valid_env_ids] = False
            self._bob_gave_completion[valid_env_ids] = False
            self._capture_bob_init_errors(valid_env_ids)
            self.set_table_color(valid_env_ids, (0.1, 0.1, 0.8))  # Blue = Bob

        self.env.scene.write_data_to_sim()

    def handle_bob_phase_end(
        self,
        env_ids: torch.Tensor,
        push_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle completion of Bob's push phase.
        Returns (success, pos_err, rot_err, bob_done, progress_reward)."""
        goal_state = self.episode_manager.goal_states
        current_state = self._extract_object_states(push_obs)

        cur_pos = self._get_obj_pos(push_obs)[env_ids]
        cur_euler = self._get_obj_euler(push_obs)[env_ids]
        goal_pos = self._get_goal_pos(push_obs)[env_ids]
        goal_euler = self._get_goal_euler(push_obs)[env_ids]

        pos_err = (cur_pos - goal_pos).norm(dim=-1)
        rot_err = _rot_distance_rad(cur_euler, goal_euler)

        if self.dpose_obs:
            dp = compute_dpose(cur_pos, goal_pos, cur_euler[:, 2], goal_euler[:, 2], self.char_length)
            success = dp < self.dpose_threshold
        else:
            success = (pos_err < POS_THRESHOLD) & (rot_err < ROT_THRESHOLD)
        self.episode_manager.mark_bob_success(env_ids, success)

        n_success = int(success.sum().item())
        n_failure = len(env_ids) - n_success
        self._iter_stats["bob_successes"] += n_success
        self._iter_stats["bob_failures"] += n_failure

        # Alice outcome reward
        outcome_rewards = torch.where(
            success,
            torch.tensor(reward_utils.ALICE_BOB_SUCCESS_REWARD, device=self.device),
            torch.tensor(reward_utils.ALICE_BOB_FAIL_REWARD, device=self.device),
        )
        self.delayed_alice_reward[env_ids] += outcome_rewards

        # Position/rotation error for logging
        step_pos_err = torch.zeros(self.num_envs, device=self.device)
        step_rot_err = torch.zeros(self.num_envs, device=self.device)
        step_pos_err[env_ids] = pos_err
        step_rot_err[env_ids] = rot_err

        step_bob_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        step_bob_done[env_ids] = True

        step_bob_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        step_bob_success[env_ids] = success

        # Phase-end progress reward
        prog_rew = self.compute_bob_progress_reward(env_ids)

        can_continue = (
            self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
        )
        continue_ids = env_ids[can_continue]
        reset_ids = env_ids[~can_continue]

        if len(continue_ids) > 0:
            self.episode_manager.transition_to_alice(continue_ids)
            self.hide_goal_ghost(continue_ids)
            _sp = self._rand_reset_objs( continue_ids)
            reset_robot_joints(self.env, continue_ids)
            self.env.scene.write_data_to_sim()
            self.episode_manager.initial_states[continue_ids] = (
                self._initial_states_from_spawn(_sp, len(continue_ids))
            )
            self.push_count[continue_ids] = 0
            self.set_table_color(continue_ids, (0.8, 0.1, 0.1))  # Red = Alice

        if len(reset_ids) > 0:
            reason = "Bob Succeeded" if success[~can_continue].any() else "Bob Failed"
            self.episode_manager.reset_episode(reset_ids, reason=reason)
            self.hide_goal_ghost(reset_ids)
            _sp = self._rand_reset_objs( reset_ids)
            reset_robot_joints(self.env, reset_ids)
            self.env.scene.write_data_to_sim()
            self.episode_manager.initial_states[reset_ids] = (
                self._initial_states_from_spawn(_sp, len(reset_ids))
            )
            self.set_table_color(reset_ids, (0.8, 0.1, 0.1))  # Red = Alice

        self.env.scene.write_data_to_sim()

        return step_bob_success, step_pos_err, step_rot_err, step_bob_done, prog_rew

    def handle_bob_early_success(
        self,
        env_ids: torch.Tensor,
        push_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Handle Bob achieving completion mid-phase (all objects at goal).
        Returns a (num_envs,) tensor of progress rewards for these envs."""
        self.episode_manager.bob_success[env_ids] = True
        self.episode_manager.completion_given[env_ids] = True

        cur_pos = self._get_obj_pos(push_obs)[env_ids]
        cur_euler = self._get_obj_euler(push_obs)[env_ids]
        goal_pos = self._get_goal_pos(push_obs)[env_ids]
        goal_euler = self._get_goal_euler(push_obs)[env_ids]

        if self.dpose_obs:
            init_dp = self.bob_init_dpose[env_ids]
            final_dp = compute_dpose(
                cur_pos, goal_pos, cur_euler[:, 2], goal_euler[:, 2], self.char_length,
            )
            r_prog = ((init_dp - final_dp) / (init_dp + 1e-6)).clamp(-1.0, 1.0)
        else:
            w_pos, w_rot = 0.6, 0.4
            init_pos = self.bob_init_pos_err[env_ids]
            init_rot = self.bob_init_rot_err[env_ids]
            final_pos = (cur_pos - goal_pos).norm(dim=-1)
            final_rot = _rot_distance_rad(cur_euler, goal_euler)
            pos_prog = (init_pos - final_pos) / (init_pos + 1e-6)
            rot_prog = (init_rot - final_rot) / (init_rot + 1e-6)
            r_prog = (w_pos * pos_prog + w_rot * rot_prog).clamp(-1.0, 1.0)
        self._bob_progress_captured[env_ids] = False

        prog_rew = torch.zeros(self.num_envs, device=self.device)
        prog_rew[env_ids] = r_prog

        self._iter_stats["bob_successes"] += len(env_ids)

        alice_success_penalty = torch.full(
            (len(env_ids),), reward_utils.ALICE_BOB_SUCCESS_REWARD, device=self.device,
        )
        self.delayed_alice_reward[env_ids] += alice_success_penalty

        can_continue = (
            self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
        )
        continue_ids = env_ids[can_continue]
        reset_ids = env_ids[~can_continue]

        if len(continue_ids) > 0:
            self.episode_manager.transition_to_alice(continue_ids)
            self.hide_goal_ghost(continue_ids)
            _sp = self._rand_reset_objs( continue_ids)
            reset_robot_joints(self.env, continue_ids)
            self.env.scene.write_data_to_sim()
            self.episode_manager.initial_states[continue_ids] = (
                self._initial_states_from_spawn(_sp, len(continue_ids))
            )
            self.push_count[continue_ids] = 0
            self.set_table_color(continue_ids, (0.8, 0.1, 0.1))  # Red = Alice

        if len(reset_ids) > 0:
            self.episode_manager.reset_episode(reset_ids, reason="Episode Complete")
            self.hide_goal_ghost(reset_ids)
            _sp = self._rand_reset_objs( reset_ids)
            reset_robot_joints(self.env, reset_ids)
            self.env.scene.write_data_to_sim()
            self.episode_manager.initial_states[reset_ids] = (
                self._initial_states_from_spawn(_sp, len(reset_ids))
            )
            self.set_table_color(reset_ids, (0.8, 0.1, 0.1))  # Red = Alice

        self.env.scene.write_data_to_sim()
        return prog_rew

    def is_alice_phase(self) -> torch.Tensor:
        return self.episode_manager.is_alice_phase()

    def is_bob_phase(self) -> torch.Tensor:
        return self.episode_manager.is_bob_phase()

    def construct_bob_observation(
        self, alice_obs: torch.Tensor, goal_states: torch.Tensor,
    ) -> torch.Tensor:
        """Construct Bob's full observation from Alice's obs + goal states.
        Alice obs: [robot(7) | obj_state(14)] = 21D
        Bob obs:   [robot(7) | obj_state(14) | goal_pose(6) | goal_dist(2)] = 29D
        """
        if alice_obs.dim() == 1:
            alice_obs = alice_obs.unsqueeze(0)
        if goal_states.dim() == 1:
            goal_states = goal_states.unsqueeze(0)

        robot = alice_obs[:, :self.robot_dim]
        obj_state = alice_obs[:, self.robot_dim:self.robot_dim + self.obj_state_dim]

        goal_pose = goal_states[:, :6]
        goal_pos = goal_pose[:, :3]
        goal_euler = goal_pose[:, 3:6]

        obj_pos = obj_state[:, :3]
        obj_euler = obj_state[:, 3:6]

        pos_dist = (obj_pos - goal_pos).norm(dim=-1, keepdim=True)
        rot_dist = _rot_distance_rad(obj_euler, goal_euler).unsqueeze(-1)

        return torch.cat([robot, obj_state, goal_pose, pos_dist, rot_dist], dim=-1)

    def close(self):
        self.env.close()
