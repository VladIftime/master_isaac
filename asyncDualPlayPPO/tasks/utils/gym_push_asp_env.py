"""
Gym-pusht ASP environment for PBRS Model C (Alice / Bob asymmetric self-play).

Gym-native port of PushASPEnvWrapper: reuses EpisodeManager + validate_goal +
PBRS verbatim, and reimplements only the physics touchpoints (object read/write,
observation build, push execution) against gym-pusht (pymunk) instead of Isaac.

ASP orchestration is centralized (batched EpisodeManager) — exactly as in
train_c_pbrs_asp.py — because Alice's outcome reward is delayed across the
Bob phase, which does not fit per-worker AsyncVectorEnv autoreset.  The N
gym envs are stepped synchronously here; train_c_gym keeps the rest of the
train_c loop unchanged.

No Isaac / cuRobo dependency — runs in .master_venv.
"""

import math

import numpy as np
import torch
import gymnasium as gym

from gym_pusht.envs.pusht import PushTEnv

from asyncDualPlayPPO.utils.episode_manager import EpisodeManager
from asyncDualPlayPPO.utils.goal_validator import validate_goal
from asyncDualPlayPPO.tasks.utils.gym_push_primitive_env import (
    meter_to_pixel, pixel_to_meter, _wrap_pi, SCALE, ARENA, TABLE_CENTER_Y,
    APPROACH_STEPS, PUSH_STEPS,
)

_OBS_ROBOT_DIM = 6
_OBS_OBJ_STATE_DIM = 14
_OBS_GOAL_DIM = 6
_OBS_DIST_DIM = 2
_OBS_DIM = 28
POS_THRESHOLD = 0.05
ROT_THRESHOLD = 0.2
ALICE_BOB_FAIL_REWARD = 5.0
ALICE_BOB_SUCCESS_REWARD = -1.0
SETTLED_Z = 0.05


def _rot_distance_rad(euler_a, euler_b):
    diff = (euler_a - euler_b).abs()
    diff = torch.min(diff, 2.0 * math.pi - diff)
    return diff.max(dim=-1)[0]


class GymPushASPEnv:
    """Synchronous N-env gym-pusht ASP wrapper exposing the train_c interface."""

    def __init__(self, num_envs=16, alice_pushes=5, bob_pushes=10,
                 max_goals_per_episode=3, num_objects=1, rel_obs=False,
                 device="cuda", seed=0):
        self._n = int(num_envs)
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.num_objects = 1
        self.alice_pushes = alice_pushes
        self.bob_pushes = bob_pushes
        self.rel_obs = rel_obs
        self.dpose_obs = False
        self.time_based_alice = False

        self.robot_dim = _OBS_ROBOT_DIM
        self.obj_state_dim = _OBS_OBJ_STATE_DIM
        self.alice_obs_dim = _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM   # 20
        self.bob_obs_dim = _OBS_DIM                                 # 28

        box = lambda d: gym.spaces.Box(-np.inf, np.inf, (d,), np.float32)
        self.alice_observation_space = box(self.alice_obs_dim)
        self.bob_observation_space = box(self.bob_obs_dim)
        self.observation_space = box(self.bob_obs_dim)
        self.state_space = self.observation_space
        self.action_space = gym.spaces.Box(0.0, 20.0, (4,), np.float32)

        self.episode_manager = EpisodeManager(
            num_envs=self._n, device=self.device,
            alice_timesteps=alice_pushes, bob_timesteps=bob_pushes,
            max_goals_per_episode=max_goals_per_episode)

        self.table_bounds = {"x_range": (-0.70, 0.70), "y_range": (-0.10, 0.90),
                             "z_min": -0.2, "z_max": 0.15}
        self.placement_bounds = {"x_range": (-0.50, 0.50), "y_range": (0.25, 0.70)}
        self.spawn_x_range = (-0.04, 0.04)
        self.spawn_y_range = (0.40, 0.45)

        self.delayed_alice_reward = torch.zeros(self._n, device=self.device)
        self.bob_init_pos_err = torch.zeros(self._n, device=self.device)
        self.bob_init_rot_err = torch.zeros(self._n, device=self.device)
        self._bob_gave_completion = torch.zeros(self._n, dtype=torch.bool, device=self.device)
        self.push_count = torch.zeros(self._n, dtype=torch.long, device=self.device)
        self.alice_phase_push_count = torch.zeros(self._n, dtype=torch.long, device=self.device)
        self.prev_obj_pos = torch.zeros(self._n, 3, device=self.device)
        self.prev_obj_euler = torch.zeros(self._n, 3, device=self.device)
        self._iter_stats = self._make_iter_stats()

        self._rng = np.random.default_rng(seed)
        self.envs = [PushTEnv(obs_type="state", render_mode="rgb_array",
                              arena_width=ARENA, arena_height=ARENA) for _ in range(self._n)]
        for i, e in enumerate(self.envs):
            e.reset(seed=seed + i)
        self._raw = np.zeros((self._n, 5), np.float64)

    @property
    def num_envs(self):
        return self._n

    # ── iteration stats ───────────────────────────────────────────────────────
    def _make_iter_stats(self):
        return {"invalid_goals": 0, "valid_goals": 0, "bob_successes": 0,
                "bob_failures": 0, "alice_total": 0, "alice_disp_3d_sum": 0.0,
                "alice_not_moved": 0}

    def reset_iter_stats(self):
        self._iter_stats = self._make_iter_stats()

    def get_iter_stats(self):
        return dict(self._iter_stats)

    # ── object reset / spawn ──────────────────────────────────────────────────
    def _place_env(self, i, bx_m, by_m, bang):
        bx, by = meter_to_pixel(bx_m, by_m)
        ax, ay = meter_to_pixel(bx_m, by_m - 0.12)
        ax = float(np.clip(ax, 12, ARENA - 12)); ay = float(np.clip(ay, 12, ARENA - 12))
        self.envs[i].reset(options={"reset_to_state":
                                    np.array([ax, ay, bx, by, bang])})
        self._raw[i] = self.envs[i].get_obs()

    def _rand_reset_objs(self, env_ids):
        ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
        n = len(ids)
        tlocal = torch.zeros(n, 3, device=self.device)
        tyaw = torch.zeros(n, device=self.device)
        for k, i in enumerate(ids):
            bx_m = float(self._rng.uniform(*self.spawn_x_range))
            by_m = float(self._rng.uniform(*self.spawn_y_range))
            bang = float(self._rng.uniform(-math.pi, math.pi))
            self._place_env(i, bx_m, by_m, bang)
            tlocal[k] = torch.tensor([bx_m, by_m, SETTLED_Z], device=self.device)
            tyaw[k] = bang
        return {"target_local": tlocal, "target_yaw": tyaw}

    def _initial_states_from_spawn(self, spawn_info, n):
        tlocal = spawn_info["target_local"]
        tyaw = spawn_info["target_yaw"]
        eul = torch.zeros(n, 3, device=self.device)
        eul[:, 2] = tyaw
        return torch.cat([tlocal, eul], dim=-1)

    def _set_block_pose(self, env_ids, states):
        """Reset given envs' block to the provided [pos(3)+euler(3)] meter states."""
        ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
        for k, i in enumerate(ids):
            bx_m = float(states[k, 0]); by_m = float(states[k, 1]); bang = float(states[k, 5])
            self._place_env(i, bx_m, by_m, bang)

    # ── observations ──────────────────────────────────────────────────────────
    def _refresh_raw(self):
        for i in range(self._n):
            self._raw[i] = self.envs[i].get_obs()

    def _get_push_obs(self):
        gs = self.episode_manager.goal_states  # (N,6) or None
        out = np.zeros((self._n, _OBS_DIM), np.float32)
        for i in range(self._n):
            ax, ay, bx, by, bang = self._raw[i]
            eex, eey = pixel_to_meter(ax, ay)
            ox, oy = pixel_to_meter(bx, by)
            bang = _wrap_pi(bang)
            if gs is not None:
                gx = float(gs[i, 0]); gy = float(gs[i, 1]); gyaw = _wrap_pi(float(gs[i, 5]))
            else:
                gx = gy = gyaw = 0.0
            dist_ee = math.hypot(eex - ox, eey - oy)
            pos_dist = math.hypot(ox - gx, oy - gy)
            rot_diff = abs(_wrap_pi(bang - gyaw))
            out[i, 0:6] = [eex, eey, 0.05, 0, 0, 0]
            out[i, 6:20] = [ox, oy, 0.05, 0, 0, bang, 0, 0, 0, 0, 0, 0, dist_ee, 1.0]
            out[i, 20:26] = [gx, gy, 0.05, 0, 0, gyaw]
            if self.rel_obs:
                out[i, 26:28] = [gx - ox, gy - oy]
            else:
                out[i, 26:28] = [pos_dist, rot_diff]
        return torch.from_numpy(out).to(self.device)

    def _get_alice_obs(self, o):
        return o[:, :self.alice_obs_dim]

    def _get_bob_obs(self, o):
        return o

    def _get_obj_pos(self, o):
        return o[:, _OBS_ROBOT_DIM:_OBS_ROBOT_DIM + 3]

    def _get_obj_euler(self, o):
        return o[:, _OBS_ROBOT_DIM + 3:_OBS_ROBOT_DIM + 6]

    def _get_goal_pos(self, o):
        return o[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM:_OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3]

    def _get_goal_euler(self, o):
        return o[:, _OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 3:_OBS_ROBOT_DIM + _OBS_OBJ_STATE_DIM + 6]

    def _extract_object_states(self, o):
        return torch.cat([self._get_obj_pos(o), self._get_obj_euler(o)], dim=-1)

    def capture_pre_push(self, o):
        self.prev_obj_pos = self._get_obj_pos(o).clone()
        self.prev_obj_euler = self._get_obj_euler(o).clone()

    def construct_bob_observation(self, alice_obs, goal_states):
        if alice_obs.dim() == 1:
            alice_obs = alice_obs.unsqueeze(0)
        if goal_states.dim() == 1:
            goal_states = goal_states.unsqueeze(0)
        robot = alice_obs[:, :self.robot_dim]
        obj_state = alice_obs[:, self.robot_dim:self.robot_dim + self.obj_state_dim]
        goal_pose = goal_states[:, :6]
        obj_pos = obj_state[:, :3]; obj_euler = obj_state[:, 3:6]
        goal_pos = goal_pose[:, :3]; goal_euler = goal_pose[:, 3:6]
        if self.rel_obs:
            tail = torch.cat([goal_pos[:, 0:1] - obj_pos[:, 0:1],
                              goal_pos[:, 1:2] - obj_pos[:, 1:2]], dim=-1)
        else:
            pos_dist = (obj_pos - goal_pos).norm(dim=-1, keepdim=True)
            rot_dist = _rot_distance_rad(obj_euler, goal_euler).unsqueeze(-1)
            tail = torch.cat([pos_dist, rot_dist], dim=-1)
        return torch.cat([robot, obj_state, goal_pose, tail], dim=-1)

    # ── push execution (meters) ───────────────────────────────────────────────
    def execute_push(self, Xs, Ys, length, theta):
        Xf = Xs + length * torch.cos(theta)
        Yf = Ys + length * torch.sin(theta)
        Xs = Xs.detach().cpu().numpy(); Ys = Ys.detach().cpu().numpy()
        Xf = Xf.detach().cpu().numpy(); Yf = Yf.detach().cpu().numpy()
        starts = np.zeros((self._n, 2), np.float32); ends = np.zeros((self._n, 2), np.float32)
        for i in range(self._n):
            sx, sy = meter_to_pixel(float(Xs[i]), float(Ys[i]))
            fx, fy = meter_to_pixel(float(Xf[i]), float(Yf[i]))
            starts[i] = [np.clip(sx, 10, ARENA - 10), np.clip(sy, 10, ARENA - 10)]
            ends[i] = [np.clip(fx, 10, ARENA - 10), np.clip(fy, 10, ARENA - 10)]
        for steps, tgt in ((APPROACH_STEPS, starts), (PUSH_STEPS, ends)):
            for _ in range(steps):
                for i in range(self._n):
                    o, _, _, _, _ = self.envs[i].step(tgt[i])
                    self._raw[i] = o

    # ── ASP phase handlers (ported from PushASPEnvWrapper) ─────────────────────
    def handle_alice_phase_end(self, env_ids, push_obs):
        self.alice_phase_push_count[env_ids] = self.episode_manager.phase_step[env_ids].long()
        goal_state = self._extract_object_states(push_obs)
        initial_state = self.episode_manager.initial_states
        active_goal = goal_state[env_ids]
        active_initial = initial_state[env_ids]

        valid, val_reward, _ = validate_goal(
            active_initial, active_goal, self.table_bounds, self.placement_bounds,
            pos_threshold=0.05, rot_threshold=0.25, min_meaningful_disp=0.10,
            require_all_moved=False, skip_shallow_penalty=self.time_based_alice)

        self.delayed_alice_reward[env_ids] = val_reward
        self._iter_stats["valid_goals"] += int(valid.sum().item())
        self._iter_stats["invalid_goals"] += int((~valid).sum().item())
        self._iter_stats["alice_total"] += len(env_ids)
        self._iter_stats["alice_disp_3d_sum"] += torch.norm(
            active_goal[:, 0:3] - active_initial[:, 0:3], dim=-1).sum().item()

        self.episode_manager.store_goal_state(active_goal, env_ids)
        self.episode_manager.mark_goal_valid(env_ids, valid)
        self.episode_manager.mark_alice_base_reward(env_ids, val_reward)

        valid_ids = env_ids[valid]
        invalid_ids = env_ids[~valid]
        if len(valid_ids) > 0:
            self._transition_to_bob(valid_ids, push_obs)
        if len(invalid_ids) > 0:
            self.episode_manager.reset_episode(invalid_ids, reason="Alice Invalid Goal")
            sp = self._rand_reset_objs(invalid_ids)
            self.episode_manager.initial_states[invalid_ids] = \
                self._initial_states_from_spawn(sp, len(invalid_ids))
        return valid_ids, invalid_ids

    def _transition_to_bob(self, env_ids, push_obs):
        self.episode_manager.transition_to_bob(env_ids)
        start_states = self.episode_manager.initial_states[env_ids]
        goal_state = self._extract_object_states(push_obs)
        # reset object to start pose
        self._set_block_pose(env_ids, start_states)
        # trivial-goal check
        _gp = goal_state[env_ids, 0:3]; _ge = goal_state[env_ids, 3:6]
        _sp = start_states[:, 0:3]; _se = start_states[:, 3:6]
        too_easy = ((_sp - _gp).norm(dim=-1) < POS_THRESHOLD) & \
                   (_rot_distance_rad(_se, _ge) < ROT_THRESHOLD)
        too_easy_ids = env_ids[too_easy]
        valid_ids = env_ids[~too_easy]
        if len(too_easy_ids) > 0:
            self.episode_manager.reset_episode(too_easy_ids, reason="Too-Easy Goal")
            sp = self._rand_reset_objs(too_easy_ids)
            self.episode_manager.initial_states[too_easy_ids] = \
                self._initial_states_from_spawn(sp, len(too_easy_ids))
            self._iter_stats["valid_goals"] -= len(too_easy_ids)
            self._iter_stats["invalid_goals"] += len(too_easy_ids)
            self.delayed_alice_reward[too_easy_ids] = -3.0
        if len(valid_ids) > 0:
            self.push_count[valid_ids] = 0
            self._bob_gave_completion[valid_ids] = False
            self._capture_bob_init_errors(valid_ids)

    def _capture_bob_init_errors(self, env_ids):
        if len(env_ids) == 0:
            return
        o = self._get_push_obs()
        cur_pos = self._get_obj_pos(o)[env_ids]; cur_eul = self._get_obj_euler(o)[env_ids]
        gp = self._get_goal_pos(o)[env_ids]; ge = self._get_goal_euler(o)[env_ids]
        self.bob_init_pos_err[env_ids] = (cur_pos - gp).norm(dim=-1)
        self.bob_init_rot_err[env_ids] = _rot_distance_rad(cur_eul, ge)

    def compute_bob_progress_reward(self, ids):
        prog = torch.zeros(self._n, device=self.device)
        if len(ids) == 0:
            return prog
        o = self._get_push_obs()
        cur_pos = self._get_obj_pos(o)[ids]; cur_eul = self._get_obj_euler(o)[ids]
        gp = self._get_goal_pos(o)[ids]; ge = self._get_goal_euler(o)[ids]
        ip = self.bob_init_pos_err[ids]; ir = self.bob_init_rot_err[ids]
        fp = (cur_pos - gp).norm(dim=-1); fr = _rot_distance_rad(cur_eul, ge)
        r = (0.6 * (ip - fp) / (ip + 1e-6) + 0.4 * (ir - fr) / (ir + 1e-6)).clamp(-1.0, 1.0)
        prog[ids] = r
        return prog

    def handle_bob_phase_end(self, env_ids, push_obs):
        cur_pos = self._get_obj_pos(push_obs)[env_ids]
        cur_eul = self._get_obj_euler(push_obs)[env_ids]
        gp = self._get_goal_pos(push_obs)[env_ids]
        ge = self._get_goal_euler(push_obs)[env_ids]
        pos_err = (cur_pos - gp).norm(dim=-1)
        rot_err = _rot_distance_rad(cur_eul, ge)
        success = (pos_err < POS_THRESHOLD) & (rot_err < ROT_THRESHOLD)
        self.episode_manager.mark_bob_success(env_ids, success)
        self._iter_stats["bob_successes"] += int(success.sum().item())
        self._iter_stats["bob_failures"] += int((~success).sum().item())

        outcome = torch.where(success,
                              torch.tensor(ALICE_BOB_SUCCESS_REWARD, device=self.device),
                              torch.tensor(ALICE_BOB_FAIL_REWARD, device=self.device))
        self.delayed_alice_reward[env_ids] += outcome

        step_pos = torch.zeros(self._n, device=self.device); step_pos[env_ids] = pos_err
        step_rot = torch.zeros(self._n, device=self.device); step_rot[env_ids] = rot_err
        step_done = torch.zeros(self._n, dtype=torch.bool, device=self.device); step_done[env_ids] = True
        step_succ = torch.zeros(self._n, dtype=torch.bool, device=self.device); step_succ[env_ids] = success
        prog = self.compute_bob_progress_reward(env_ids)

        self._end_or_continue(env_ids)
        return step_succ, step_pos, step_rot, step_done, prog

    def handle_bob_early_success(self, env_ids, push_obs):
        self.episode_manager.bob_success[env_ids] = True
        self.episode_manager.completion_given[env_ids] = True
        prog = self.compute_bob_progress_reward(env_ids)
        self._iter_stats["bob_successes"] += len(env_ids)
        self.delayed_alice_reward[env_ids] += ALICE_BOB_SUCCESS_REWARD
        self._end_or_continue(env_ids)
        return prog

    def _end_or_continue(self, env_ids):
        can_continue = self.episode_manager.goal_count[env_ids] < self.episode_manager.max_goals
        cont = env_ids[can_continue]; rst = env_ids[~can_continue]
        for ids, fn in ((cont, self.episode_manager.transition_to_alice),
                        (rst, lambda x: self.episode_manager.reset_episode(x, reason="Episode End"))):
            if len(ids) > 0:
                fn(ids)
                sp = self._rand_reset_objs(ids)
                self.episode_manager.initial_states[ids] = \
                    self._initial_states_from_spawn(sp, len(ids))
                self.push_count[ids] = 0

    # ── no-ops kept for train_c interface parity ──────────────────────────────
    def set_table_color(self, *a, **k):
        pass

    def hide_goal_ghost(self, *a, **k):
        pass

    def _update_goal_in_extras(self):
        pass

    def is_alice_phase(self):
        return self.episode_manager.is_alice_phase()

    def is_bob_phase(self):
        return self.episode_manager.is_bob_phase()

    def reset(self):
        env_ids = torch.arange(self._n, device=self.device)
        self.episode_manager.reset_episode(env_ids, reason="Global Reset")
        self.delayed_alice_reward[:] = 0.0
        self.push_count[:] = 0
        self.alice_phase_push_count[:] = 0
        self._bob_gave_completion[:] = False
        sp = self._rand_reset_objs(env_ids)
        self.episode_manager.initial_states = self._initial_states_from_spawn(sp, self._n)
        o = self._get_push_obs()
        self.capture_pre_push(o)
        return o

    def close(self):
        for e in self.envs:
            e.close()
