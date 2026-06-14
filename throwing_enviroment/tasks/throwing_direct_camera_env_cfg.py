"""DirectRLEnv configuration with TiledCamera for visual observation benchmarking.

Extends ThrowingDirectEnvCfg by adding a TiledCamera sensor aimed at the basket
target area. Used to benchmark tiled rendering performance across parallel envs.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .throwing_direct_env_cfg import ThrowingDirectEnvCfg, ThrowingDirectSceneCfg


@configclass
class ThrowingDirectCameraSceneCfg(ThrowingDirectSceneCfg):
    pass


@configclass
class ThrowingDirectCameraEnvCfg(ThrowingDirectEnvCfg):
    """ThrowingDirectEnv + TiledCamera for rendering performance benchmark."""

    scene: ThrowingDirectCameraSceneCfg = ThrowingDirectCameraSceneCfg(
        num_envs=2048, env_spacing=3.0
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 2.5, 1.5),
            rot=(0.8536, 0.1464, 0.1464, -0.4732),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=128,
        height=128,
    )

    write_image_to_file: bool = False
