"""VecEnv wrapper for DirectRLEnv → stable-baselines3 compatibility.

DirectRLEnv already handles vectorized environments (replicate_physics=True).
This wrapper exposes the VecEnv interface that SB3 expects, passing through
calls to a single DirectRLEnv instance that runs N envs internally.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


class DirectRLVecEnv(VecEnv):
    """Wraps a DirectRLEnv for stable-baselines3 compatibility."""

    def __init__(self, env):
        self._env = env
        self._device = env.device
        num_envs = env.num_envs
        obs_dim = env.cfg.observation_space  # type: ignore
        act_dim = env.cfg.action_space  # type: ignore

        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        act_space = gym.spaces.Box(
            low=np.array([-1, -1, 0.05, 0.1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )
        super().__init__(num_envs, obs_space, act_space)

        self._last_actions: torch.Tensor | None = None

    def step_async(self, actions: np.ndarray) -> None:
        self._last_actions = torch.from_numpy(actions).float().to(self._device)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        obs, reward, terminated, truncated, _extras = self._env.step(self._last_actions)
        obs_arr = obs["policy"].cpu().numpy()
        rew_arr = reward.cpu().numpy()
        done_arr = (terminated | truncated).cpu().numpy()

        infos: list[dict] = []
        for i in range(self.num_envs):
            info: dict = {"TimeLimit.truncated": bool(truncated[i].item())}
            infos.append(info)

        return obs_arr, rew_arr, done_arr, infos

    def reset(self) -> np.ndarray:
        obs, _info = self._env.reset()
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

    def get_images(self):
        return []
