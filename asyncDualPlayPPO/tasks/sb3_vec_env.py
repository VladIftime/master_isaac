"""VecEnv wrapper for DirectRLEnv -> stable-baselines3 compatibility.

DirectRLEnv already handles vectorized environments (replicate_physics=True).
This wrapper exposes the VecEnv interface that SB3 expects, passing through
calls to a single DirectRLEnv instance that runs N envs internally.

Supports dict observations for HER (HerReplayBuffer).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


class DirectRLVecEnv(VecEnv):
    """Wraps a DirectRLEnv for stable-baselines3 compatibility.

    Extracts observation/action spaces from the env's config and
    passes through all VecEnv calls to the single internal env.
    """

    def __init__(self, env):
        self._env = env
        self._device = env.device
        num_envs = env.num_envs

        obs_sample = env._get_observations()

        if "achieved_goal" in obs_sample and "desired_goal" in obs_sample:
            obs_keys = list(obs_sample.keys())
            obs_spaces = {}
            for k in obs_keys:
                v = obs_sample[k]
                obs_spaces[k] = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=v.shape[1:], dtype=np.float32,
                )
            obs_space = gym.spaces.Dict(obs_spaces)
        else:
            obs_dim = env.cfg.observation_space
            obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )

        act_dim = env.cfg.action_space
        act_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
        )

        super().__init__(num_envs, obs_space, act_space)

        self._last_actions: torch.Tensor | None = None
        self._uses_dict_obs = isinstance(obs_space, gym.spaces.Dict)

    def step_async(self, actions: np.ndarray) -> None:
        self._last_actions = torch.from_numpy(actions).float().to(self._device)

    def step_wait(self) -> tuple:
        obs, reward, terminated, truncated, extras = self._env.step(self._last_actions)
        rew_arr = reward.cpu().numpy()
        term_arr = terminated.cpu().numpy()
        trunc_arr = truncated.cpu().numpy()
        done_arr = (terminated | truncated).cpu().numpy()

        if self._uses_dict_obs:
            obs_arr = {}
            for k, v in obs.items():
                obs_arr[k] = v.cpu().numpy()
        else:
            obs_arr = obs["policy"].cpu().numpy()

        infos: list[dict] = []
        for i in range(self.num_envs):
            info: dict = {
                "TimeLimit.truncated": bool(trunc_arr[i].item()),
                "terminal_observation": {},
            }
            if self._uses_dict_obs:
                for k, v in obs_arr.items():
                    info["terminal_observation"][k] = v[i].copy()

            achieved = obs_arr["achieved_goal"][i].copy() if self._uses_dict_obs else np.zeros(3)
            info["achieved_goal"] = achieved
            info["desired_goal"] = obs_arr["desired_goal"][i].copy() if self._uses_dict_obs else np.zeros(3)

            prev_obj_local = None
            if hasattr(self._env, "prev_obj_pos"):
                p = self._env.prev_obj_pos[i].cpu().numpy()
                e = self._env.prev_obj_euler[i].cpu().numpy()
                prev_obj_local = np.array([p[0], p[1], e[2]], dtype=np.float32)
            info["prev_achieved_goal"] = prev_obj_local if prev_obj_local is not None else achieved

            infos.append(info)

        return obs_arr, rew_arr, done_arr, infos

    def reset(self) -> np.ndarray:
        obs, _info = self._env.reset()
        if self._uses_dict_obs:
            return {k: v.cpu().numpy() for k, v in obs.items()}
        return obs["policy"].cpu().numpy()

    def close(self) -> None:
        self._env.close()

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self._env, attr_name)]

    def set_attr(self, attr_name: str, value, indices=None):
        setattr(self._env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return [getattr(self._env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [isinstance(self._env, wrapper_class)]

    def seed(self, seed: int | None = None):
        pass

    def compute_reward(self, achieved_goal, desired_goal, infos):
        return self._env.compute_reward(achieved_goal, desired_goal, infos)

    def get_images(self):
        return []
