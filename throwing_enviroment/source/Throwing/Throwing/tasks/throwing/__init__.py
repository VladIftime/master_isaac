# (c) Open source contributors
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Throwing-Direct-v0",
    entry_point="tasks.throwing_env:ThrowingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.throwing_env_cfg:ThrowingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Throwing-Primitive-v0",
    entry_point="tasks.throwing_primitive_env:ThrowingPrimitiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.throwing_primitive_env_cfg:ThrowingPrimitiveEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Throwing-Primitive-Direct-v0",
    entry_point="tasks.throwing_direct_env:ThrowingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.throwing_direct_env_cfg:ThrowingDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
