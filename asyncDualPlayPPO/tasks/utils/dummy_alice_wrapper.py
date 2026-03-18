import torch
from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper

class DummyAliceWrapper(AsyncDualPlayEnvWrapper):
    """
    A debugging wrapper for Asymmetric Self-Play.
    Teleports the block during Alice's phase to guarantee a valid goal,
    forcing the environment to let Bob play and train on a static coordinate.
    """
    def step(self, action):
        # 1. Let the environment step normally
        obs, rew, done, truncated, extras = super().step(action)

        # 2. Identify which environments are currently in Alice's phase,
        #    but SKIP step 0 (phase just reset — block is at safe-reset position,
        #    which is the initial position for the "Moved" calculation; teleporting
        #    here would make every subsequent episode start at the goal → Moved ≈ 0
        #    → invalid goal every time).
        alice_mask = (
            self.episode_manager.is_alice_phase()
            & (self.episode_manager.phase_step > 0)
        )
        alice_envs = alice_mask.nonzero(as_tuple=True)[0]

        # 3. Teleport the block ONLY during mid-episode Alice steps
        if len(alice_envs) > 0:
            target_obj = self.env.scene["target_object"]

            # Goal in env-LOCAL frame: [0.15, 0.5, 0.05].
            # Safe reset is [-0.15, 0.5, 0.05] local, so Bob has to push 30 cm right.
            # Convert to WORLD frame by adding each env's origin (env_origins is (N_total, 3)).
            fixed_local = torch.tensor([0.15, 0.5, 0.05], device=self.device)
            env_origins = self.env.scene.env_origins[alice_envs]   # (N_alice, 3)
            fixed_pos = env_origins + fixed_local                   # (N_alice, 3) world coords

            # Identity quaternion (flat/level block)
            fixed_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(alice_envs), 1)

            root_states = target_obj.data.root_state_w.clone()
            root_states[alice_envs, 0:3] = fixed_pos
            root_states[alice_envs, 3:7] = fixed_quat
            root_states[alice_envs, 7:] = 0.0  # Zero velocity so block doesn't drift

            target_obj.write_root_state_to_sim(root_states[alice_envs], env_ids=alice_envs)

        return obs, rew, done, truncated, extras