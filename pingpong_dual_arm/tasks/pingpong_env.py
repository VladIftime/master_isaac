"""Ping pong environment class — two dual-arm robots playing competitive ping pong.

Provides a standard Gymnasium interface compatible with any RL library
(SB3, rsl-rl, torchrl, etc.).
"""

from __future__ import annotations

import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import events

from .pingpong_env_cfg import PingPongDualArmEnvCfg


class PingPongDualArmEnv(ManagerBasedRLEnv):
    """Environment with two dual-arm robots playing ping pong."""

    cfg: PingPongDualArmEnvCfg

    def __init__(self, cfg: PingPongDualArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _configure_env_timers(self) -> None:
        if self.cfg.rewards:
            self._step_timers["rewards"] = self._Timer()
        if self.cfg.terminations:
            self._step_timers["terminations"] = self._Timer()
        self._step_timers["physics_step"] = self._Timer()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        return

    def _apply_action(self) -> None:
        for action_term in self.action_manager._terms.values():
            action_term.apply()
