"""Configuration for the Throwing Primitive gym.Env wrapper.

Simple dataclass config — not a DirectRLEnvCfg since the environment
wraps ThrowingEnv (ManagerBasedRLEnv) internally.
"""

from dataclasses import dataclass, field

from .throw_primitive import DRINK_WORLD_X, DRINK_WORLD_Y, DRINK_WORLD_Z


@dataclass
class ThrowingPrimitiveEnvCfg:
    """Config for the Gazebo-style macro-action throwing environment."""

    num_envs: int = 64
    playing_arm_side: str = "right"
    randomize_target: bool = True
    target_x_range: tuple = (0.0, 0.45)
    target_y_range: tuple = (1.0, 1.4)

    drink_x: float = DRINK_WORLD_X
    drink_y: float = DRINK_WORLD_Y
    drink_z: float = DRINK_WORLD_Z

    seed: int = 42
