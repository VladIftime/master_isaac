"""DirectRLEnv with TiledCamera for rendering performance benchmarking.

Subclasses ThrowingDirectEnv and adds a TiledCamera sensor. The camera
output is read every step but not used for policy (observations remain
state-based). This isolates rendering overhead for benchmarking.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from .throwing_direct_env import ThrowingDirectEnv
from .throwing_direct_camera_env_cfg import ThrowingDirectCameraEnvCfg


class ThrowingDirectCameraEnv(ThrowingDirectEnv):
    """ThrowingDirectEnv + TiledCamera for rendering benchmarks."""

    cfg: ThrowingDirectCameraEnvCfg

    def __init__(self, cfg: ThrowingDirectCameraEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.clone_environments(copy_from_source=False)

    def _get_observations(self):
        obs = super()._get_observations()
        _rgb = self._tiled_camera.data.output["rgb"]
        return obs
