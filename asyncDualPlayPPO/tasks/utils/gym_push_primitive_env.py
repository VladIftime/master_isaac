"""
Smart gym-pusht push-primitive env for PBRS Models A/B/C.

One gym.Env.step() == one push macro-action: decode 4D x 21-bin action ->
object-relative push -> drive the circular agent (APPROACH + PUSH control
steps) -> compute the 30-D observation + PBRS reward + thesis-gate done.

The env NEVER signals terminated/truncated (so gymnasium vector autoreset is
disabled); it self-resets internally on done and reports done + metrics through
`info`.  This makes it trivially compatible with AsyncVectorEnv for CPU
parallelism, and with the project's custom PPO (which drives the rollout loop
manually) via TorchVecAdapter.

No Isaac / cuRobo dependency — runs in .master_venv.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
from functools import partial

import numpy as np
import torch
torch.set_num_threads(1)
import gymnasium as gym
from gymnasium import spaces

from gym_pusht.envs.pusht import PushTEnv

from asyncDualPlayPPO.tasks.utils.action_push_relative import decode_push_action_relative
from asyncDualPlayPPO.tasks.utils.reward_pbrs import (
    potential_pos, potential_rot, compute_pbrs_reward, check_done_pbrs,
)

# ── meter <-> pixel frame (matches validate_pusht_gym.py) ──────────────────────
SCALE = 750.0
ARENA = 750
TABLE_CENTER_Y = 0.475
APPROACH_STEPS = 40
PUSH_STEPS = 60

_WS_X = (-0.30, 0.30)
_WS_Y = (0.30, 0.65)
_GOAL_MIN_DIST = 0.06
_GOAL_MAX_DIST = 0.45

ROBOT_DIM = 6
OBJ_STATE_DIM = 14
OBS_DIM = 30
NUM_BINS = 21


def meter_to_pixel(mx, my):
    return mx * SCALE + ARENA / 2.0, (my - TABLE_CENTER_Y) * SCALE + ARENA / 2.0


def pixel_to_meter(px, py):
    return (px - ARENA / 2.0) / SCALE, (py - ARENA / 2.0) / SCALE + TABLE_CENTER_Y


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GymPushPrimitiveEnv(gym.Env):
    """Single-agent PBRS push env (Models A / B).  One step = one push."""

    metadata = {"render_modes": []}

    def __init__(self, max_pushes_per_episode=5, seed=0):
        super().__init__()
        self.max_pushes = int(max_pushes_per_episode)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.MultiDiscrete([NUM_BINS] * 4)
        self._env = PushTEnv(obs_type="state", render_mode="rgb_array",
                             arena_width=ARENA, arena_height=ARENA)
        self._env.reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._goal_px = np.zeros(3)
        self._push_count = 0
        self._prev_phi_pos = torch.zeros(1)
        self._prev_phi_rot = torch.zeros(1)
        self._gave_completion = torch.zeros(1, dtype=torch.bool)
        self._gave_rot_bonus = torch.zeros(1, dtype=torch.bool)
        # curriculum hooks (B); A leaves defaults
        self.w_pos = 10.0
        self.w_rot = 10.0
        self.enable_rot_sparse = True
        self.pos_term_threshold = 0.0

    def set_curriculum(self, w_rot, pos_term_threshold, enable_rot_sparse):
        self.w_rot = float(w_rot)
        self.pos_term_threshold = float(pos_term_threshold)
        self.enable_rot_sparse = bool(enable_rot_sparse)

    # ── sampling ──────────────────────────────────────────────────────────────
    def _sample_block(self):
        bx_m = self._rng.uniform(*_WS_X); by_m = self._rng.uniform(*_WS_Y)
        bang = self._rng.uniform(-math.pi, math.pi)
        bx, by = meter_to_pixel(bx_m, by_m)
        ax, ay = meter_to_pixel(bx_m, by_m - 0.12)
        ax = float(np.clip(ax, 12, ARENA - 12)); ay = float(np.clip(ay, 12, ARENA - 12))
        return np.array([ax, ay, bx, by, bang]), (bx_m, by_m)

    def _sample_goal(self, block_m):
        bx_m, by_m = block_m
        gx_m = gy_m = 0.0
        for _ in range(20):
            gx_m = self._rng.uniform(*_WS_X); gy_m = self._rng.uniform(*_WS_Y)
            if _GOAL_MIN_DIST <= math.hypot(gx_m - bx_m, gy_m - by_m) <= _GOAL_MAX_DIST:
                break
        gang = self._rng.uniform(-math.pi, math.pi)
        gx, gy = meter_to_pixel(gx_m, gy_m)
        return np.array([gx, gy, gang])

    def _raw(self):
        return self._env.get_obs()  # [ax,ay,bx,by,bangle]

    def _build_obs(self):
        ax, ay, bx, by, bang = self._raw()
        gpx, gpy, gang = self._goal_px
        eex, eey = pixel_to_meter(ax, ay)
        ox, oy = pixel_to_meter(bx, by)
        gx, gy = pixel_to_meter(gpx, gpy)
        bang = _wrap_pi(bang); gang = _wrap_pi(gang)
        dist_ee = math.hypot(eex - ox, eey - oy)
        pos_dist = math.hypot(ox - gx, oy - gy)
        rot_diff = abs(_wrap_pi(bang - gang))
        o = np.zeros(OBS_DIM, np.float32)
        o[0:6] = [eex, eey, 0.05, 0.0, 0.0, 0.0]
        o[6:20] = [ox, oy, 0.05, 0.0, 0.0, bang, 0, 0, 0, 0, 0, 0, dist_ee, 1.0]
        o[20:26] = [gx, gy, 0.02, 0.0, 0.0, gang]
        o[26:28] = [pos_dist, rot_diff]
        o[28:30] = [gx - ox, gy - oy]
        return o

    def _phi(self, obs_np):
        t = torch.from_numpy(obs_np).unsqueeze(0)
        op = t[:, ROBOT_DIM:ROBOT_DIM + 3]; oe = t[:, ROBOT_DIM + 3:ROBOT_DIM + 6]
        gp = t[:, ROBOT_DIM + OBJ_STATE_DIM:ROBOT_DIM + OBJ_STATE_DIM + 3]
        ge = t[:, ROBOT_DIM + OBJ_STATE_DIM + 3:ROBOT_DIM + OBJ_STATE_DIM + 6]
        return potential_pos(op, gp), potential_rot(oe[..., 2], ge[..., 2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, block_m = self._sample_block()
        self._env.reset(options={"reset_to_state": state})
        self._goal_px = self._sample_goal(block_m)
        self._env.goal_pose = self._goal_px.copy()
        self._push_count = 0
        self._gave_completion[:] = False
        self._gave_rot_bonus[:] = False
        obs = self._build_obs()
        self._prev_phi_pos, self._prev_phi_rot = self._phi(obs)
        return obs, {}

    def _run_push(self, bins_np):
        t = torch.from_numpy(self._build_obs()).unsqueeze(0)
        obj_xy = t[:, ROBOT_DIM:ROBOT_DIM + 2]
        obj_yaw = t[:, ROBOT_DIM + 5]
        bins = torch.from_numpy(np.asarray(bins_np, dtype=np.float32)).unsqueeze(0)
        Xs, Ys, length, theta = decode_push_action_relative(
            bins, obj_xy, obj_yaw, num_bins=NUM_BINS, min_r=0.03, max_r=0.08, max_len=0.20)
        Xf = Xs + length * torch.cos(theta); Yf = Ys + length * torch.sin(theta)
        sx, sy = meter_to_pixel(float(Xs), float(Ys))
        fx, fy = meter_to_pixel(float(Xf), float(Yf))
        start = np.array([np.clip(sx, 10, ARENA - 10), np.clip(sy, 10, ARENA - 10)], np.float32)
        end = np.array([np.clip(fx, 10, ARENA - 10), np.clip(fy, 10, ARENA - 10)], np.float32)
        for _ in range(APPROACH_STEPS):
            self._env.step(start)
        for _ in range(PUSH_STEPS):
            self._env.step(end)

    def step(self, action):
        self._run_push(action)
        obs = self._build_obs()
        t = torch.from_numpy(obs).unsqueeze(0)
        op = t[:, ROBOT_DIM:ROBOT_DIM + 3]; oe = t[:, ROBOT_DIM + 3:ROBOT_DIM + 6]
        gp = t[:, ROBOT_DIM + OBJ_STATE_DIM:ROBOT_DIM + OBJ_STATE_DIM + 3]
        ge = t[:, ROBOT_DIM + OBJ_STATE_DIM + 3:ROBOT_DIM + OBJ_STATE_DIM + 6]
        res = compute_pbrs_reward(op, oe, gp, ge, self._prev_phi_pos, self._prev_phi_rot,
                                  self._gave_completion, self._gave_rot_bonus,
                                  w_pos=self.w_pos, w_rot=self.w_rot,
                                  enable_rot_sparse=self.enable_rot_sparse)
        self._gave_completion = res["gave_completion"]
        self._gave_rot_bonus = res["gave_rot_bonus"]
        reward = float(res["reward"][0])
        self._push_count += 1
        pc = torch.tensor([self._push_count])
        done_t, reasons = check_done_pbrs(t, torch.zeros(1, dtype=torch.bool), pc,
                                          self.max_pushes, res["at_goal"],
                                          robot_dim=ROBOT_DIM, obj_state_dim=OBJ_STATE_DIM,
                                          pos_term_threshold=self.pos_term_threshold)
        if bool((reasons["launched"] | reasons["tipped"] | reasons["oob"])[0]):
            reward = -10.0
        done = bool(done_t[0])
        info = {
            "done": done,
            "pos_err": float(res["pos_err"][0]),
            "cos_rot_err": float(res["cos_rot_err"][0]),
            "at_goal": bool(res["at_goal"][0]),
            "pos_only": bool(reasons["pos_only"][0]),
            "success": bool(reasons["success"][0]),
            "ep_pushes": int(self._push_count),
        }
        if done:
            obs, _ = self.reset()
        else:
            self._prev_phi_pos = res["phi_pos_now"]
            self._prev_phi_rot = res["phi_rot_now"]
        return obs, reward, False, False, info

    def close(self):
        self._env.close()


def make_single_env(seed=0, max_pushes_per_episode=5):
    return GymPushPrimitiveEnv(max_pushes_per_episode=max_pushes_per_episode, seed=seed)


class TorchVecAdapter:
    """Wraps AsyncVectorEnv to present the custom-PPO vec_env tensor contract."""

    def __init__(self, num_envs=16, device="cuda", max_pushes_per_episode=5,
                 seed=0, asp=False):
        self.num_envs = int(num_envs)
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        fns = [partial(make_single_env, seed=seed + i,
                       max_pushes_per_episode=max_pushes_per_episode)
               for i in range(self.num_envs)]
        self.venv = gym.vector.AsyncVectorEnv(
            fns, context="spawn", autoreset_mode=gym.vector.AutoresetMode.DISABLED)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.state_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(0.0, float(NUM_BINS - 1), (4,), np.float32)
        self.robot_dim = ROBOT_DIM
        self.obj_state_dim = OBJ_STATE_DIM
        self.obs_dim = OBS_DIM

    def reset(self):
        obs, _ = self.venv.reset()
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)

    def step(self, action_tensor):
        bins = action_tensor.detach().cpu().numpy().astype(np.int64)
        obs, reward, term, trunc, info = self.venv.step(bins)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        rew_t = torch.as_tensor(np.asarray(reward), dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(np.asarray(info["done"]), dtype=torch.bool, device=self.device)
        return obs_t, rew_t, done_t, info

    def set_curriculum(self, w_rot, pos_term_threshold, enable_rot_sparse):
        self.venv.call("set_curriculum", w_rot, pos_term_threshold, enable_rot_sparse)

    def get_state(self):
        return None

    def close(self):
        self.venv.close()
