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
        alice_mask = self.episode_manager.is_alice_phase() & (
            self.episode_manager.phase_step > 0
        )
        alice_envs = alice_mask.nonzero(as_tuple=True)[0]

        # 3. Teleport the block ONLY during mid-episode Alice steps
        if len(alice_envs) > 0:
            target_obj = self.env.scene["target_object"]

            # Goal in env-LOCAL frame: [0.15, 0.5, 0.05].
            # Safe reset is [-0.15, 0.5, 0.05] local, so Bob has to push 30 cm right.
            # Convert to WORLD frame by adding each env's origin (env_origins is (N_total, 3)).
            fixed_local = torch.tensor([0.15, 0.5, 0.05], device=self.device)
            env_origins = self.env.scene.env_origins[alice_envs]  # (N_alice, 3)
            fixed_pos = env_origins + fixed_local  # (N_alice, 3) world coords

            # Identity quaternion (flat/level block)
            fixed_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                len(alice_envs), 1
            )

            root_states = target_obj.data.root_state_w.clone()
            root_states[alice_envs, 0:3] = fixed_pos
            root_states[alice_envs, 3:7] = fixed_quat
            root_states[alice_envs, 7:] = 0.0  # Zero velocity so block doesn't drift

            target_obj.write_root_state_to_sim(
                root_states[alice_envs], env_ids=alice_envs
            )

        return obs, rew, done, truncated, extras


class DummyBobWrapper(DummyAliceWrapper):
    """
    Test wrapper (--test_bob_reward): at a fixed Bob step, teleports the target
    object to its goal position.  Use this to verify the reward pipeline:
      - Sparse +1 reward should fire the step after teleport (target at goal)
      - +5 completion bonus should fire when all objects are at goal
      - SR should rise to ~1.0 within the first few iterations

    Extends DummyAliceWrapper so Alice still produces a valid fixed goal
    (target teleported to [0.15, 0.5, 0.05] local) before Bob plays.
    """

    def __init__(
        self, env, device, alice_timesteps=100, bob_timesteps=200, teleport_step=50
    ):
        # DummyAliceWrapper has no __init__, so call the grandparent directly
        # to avoid linter confusion with object.__init__
        AsyncDualPlayEnvWrapper.__init__(
            self,
            env=env,
            device=device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
        )
        self.teleport_step = teleport_step

    def step(self, action):
        obs, rew, done, truncated, extras = super().step(action)

        # At exactly teleport_step into Bob's phase, snap target to its goal.
        bob_mask = self.episode_manager.is_bob_phase() & (
            self.episode_manager.phase_step == self.teleport_step
        )
        bob_envs = bob_mask.nonzero(as_tuple=True)[0]

        if len(bob_envs) > 0 and self.episode_manager.goal_states is not None:
            target_obj = self.env.scene["target_object"]
            # goal_states shape: (N_total, 14) = [target_pose(7) | cube_pose(7)]
            # target_pose = [pos(3) | quat(4)] in world frame
            goal_states = self.episode_manager.goal_states

            # goal_states stores LOCAL coords; convert to WORLD by adding env origins
            env_origins_bob = self.env.scene.env_origins[bob_envs]  # (N_bob, 3)
            goal_pos_world = goal_states[bob_envs, 0:3] + env_origins_bob

            root_states = target_obj.data.root_state_w.clone()
            root_states[bob_envs, 0:3] = goal_pos_world  # world pos
            root_states[bob_envs, 3:7] = goal_states[
                bob_envs, 3:7
            ]  # goal quat (orientation only, no offset)
            root_states[bob_envs, 7:] = 0.0  # zero vel

            target_obj.write_root_state_to_sim(root_states[bob_envs], env_ids=bob_envs)
            print(
                f"[DummyBob] Teleported target→goal for {len(bob_envs)} envs "
                f"at Bob step {self.teleport_step} | "
                f"goal_pos_local={goal_states[bob_envs[0], 0:3].tolist()} "
                f"goal_pos_world={goal_pos_world[0].tolist()}",
                flush=True,
            )

        return obs, rew, done, truncated, extras
