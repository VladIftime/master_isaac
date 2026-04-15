import torch
from isaaclab.managers import SceneEntityCfg
from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
from asyncDualPlayPPO.tasks.utils.observations import goal_distance, _euler_xyz_to_quat
from asyncDualPlayPPO.utils.goal_validator import validate_goal
from asyncDualPlayPPO.tasks.utils.events import reset_objects_to_fixed_safe_pose, reset_robot_joints


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
        #    but SKIP steps 0 and 1.
        #    Step 0: transition step — physics write not committed yet.
        #    Step 1: initial_states capture — must read the post-reset position.
        #    Teleporting before step 1 is captured would record the goal position
        #    as the start, making every subsequent Moved distance ≈ 0.
        alice_mask = self.episode_manager.is_alice_phase() & (
            self.episode_manager.phase_step > 1
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
        self, env, device, alice_timesteps=100, bob_timesteps=200, teleport_step=50,
        num_objects=2,
    ):
        # DummyAliceWrapper has no __init__, so call the grandparent directly
        # to avoid linter confusion with object.__init__
        AsyncDualPlayEnvWrapper.__init__(
            self,
            env=env,
            device=device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            num_objects=num_objects,
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
            # goal_states layout: [t_pos(3), t_euler(3), c_pos(3), c_euler(3)] = 12D LOCAL
            goal_states = self.episode_manager.goal_states

            # goal_states layout: [t_pos(3), t_euler(3), c_pos(3), c_euler(3)] = 12D LOCAL
            env_origins_bob = self.env.scene.env_origins[bob_envs]  # (N_bob, 3)
            goal_pos_world = goal_states[bob_envs, 0:3] + env_origins_bob
            goal_quat = _euler_xyz_to_quat(goal_states[bob_envs, 3:6])  # euler → quat

            root_states = target_obj.data.root_state_w.clone()
            root_states[bob_envs, 0:3] = goal_pos_world  # world pos
            root_states[bob_envs, 3:7] = goal_quat  # goal orientation as quat
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


# ─────────────────────────────────────────────────────────────────────────────
# DummyGoalDistanceWrapper
# Tests that goal_distance() returns ~0 when the object is AT the stored goal.
#
# How to use (add --dummy_goal_distance flag or instantiate directly):
#   env = DummyGoalDistanceWrapper(base_env, device=device)
#   # Run normally; at teleport_step into Bob's phase both objects are snapped
#   # to their stored goals.  The wrapper then prints the measured goal_distance.
#   # Expected: pos_dist < 0.01, rot_dist < 0.01.
#   # A pos_dist matching the env_origin magnitude (~2 m) indicates a double-
#   # subtraction bug in goal_distance(); any other large value indicates a
#   # local/world frame mismatch in goal_states storage.
# ─────────────────────────────────────────────────────────────────────────────
class DummyGoalDistanceWrapper(DummyAliceWrapper):
    """
    Diagnostic wrapper: snaps both objects to their stored goals during Bob's
    phase, then immediately measures goal_distance() and asserts ~0.

    Detects:
    - Double env_origins subtraction in goal_distance()
    - Local/world frame mismatch in stored goal_states

    Expected console output at teleport_step:
        [DistCheck] Env X step=T: target pos=0.0000 rot=0.0000 | cube pos=0.0000 rot=0.0000 ✓
    """

    def __init__(
        self,
        env,
        device,
        alice_timesteps=100,
        bob_timesteps=200,
        teleport_step=30,
        num_objects=2,
    ):
        AsyncDualPlayEnvWrapper.__init__(
            self,
            env=env,
            device=device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            num_objects=num_objects,
        )
        self.teleport_step = teleport_step

    def step(self, action):
        obs, rew, done, truncated, extras = super().step(action)

        bob_mask = self.episode_manager.is_bob_phase() & (
            self.episode_manager.phase_step == self.teleport_step
        )
        bob_envs = bob_mask.nonzero(as_tuple=True)[0]

        if len(bob_envs) > 0 and self.episode_manager.goal_states is not None:
            goal_states = self.episode_manager.goal_states  # (N_total, 12), LOCAL
            target_obj = self.env.scene["target_object"]
            cube_obj = self.env.scene["cube"]

            env_origins = self.env.scene.env_origins[bob_envs]

            # goal_states layout: [t_pos(3), t_euler(3), c_pos(3), c_euler(3)] = 12D LOCAL
            # Snap target to stored goal (LOCAL → WORLD)
            tgt_pos_world = goal_states[bob_envs, 0:3] + env_origins
            tgt_quat = _euler_xyz_to_quat(goal_states[bob_envs, 3:6])
            root_t = target_obj.data.root_state_w.clone()
            root_t[bob_envs, 0:3] = tgt_pos_world
            root_t[bob_envs, 3:7] = tgt_quat
            root_t[bob_envs, 7:] = 0.0
            target_obj.write_root_state_to_sim(root_t[bob_envs], env_ids=bob_envs)

            # Snap cube to stored goal too
            cube_pos_world = goal_states[bob_envs, 6:9] + env_origins
            cube_quat = _euler_xyz_to_quat(goal_states[bob_envs, 9:12])
            root_c = cube_obj.data.root_state_w.clone()
            root_c[bob_envs, 0:3] = cube_pos_world
            root_c[bob_envs, 3:7] = cube_quat
            root_c[bob_envs, 7:] = 0.0
            cube_obj.write_root_state_to_sim(root_c[bob_envs], env_ids=bob_envs)

            self.env.scene.write_data_to_sim()

            # Measure distances AFTER snap — both must be ~0
            tgt_dists = goal_distance(self.env, SceneEntityCfg("target_object"))
            cube_dists = goal_distance(self.env, SceneEntityCfg("cube"))

            for env_id in bob_envs:
                t_pos = tgt_dists[env_id, 0].item()
                t_rot = tgt_dists[env_id, 1].item()
                c_pos = cube_dists[env_id, 0].item()
                c_rot = cube_dists[env_id, 1].item()
                ok = t_pos < 0.01 and t_rot < 0.01 and c_pos < 0.01 and c_rot < 0.01
                status = "✓" if ok else "✗ BUG"
                print(
                    f"[DistCheck] Env {env_id.item()} step={self.teleport_step}: "
                    f"target pos={t_pos:.4f} rot={t_rot:.4f} | "
                    f"cube pos={c_pos:.4f} rot={c_rot:.4f} {status}",
                    flush=True,
                )
                if not ok:
                    t_world = target_obj.data.root_pos_w[env_id].tolist()
                    t_goal_l = goal_states[env_id, 0:3].tolist()
                    origin = self.env.scene.env_origins[env_id].tolist()
                    print(
                        f"  [DistCheck DEBUG] target world={[round(x,3) for x in t_world]} "
                        f"goal_local={[round(x,3) for x in t_goal_l]} "
                        f"env_origin={[round(x,3) for x in origin]}",
                        flush=True,
                    )

        return obs, rew, done, truncated, extras


# ─────────────────────────────────────────────────────────────────────────────
# DummyMovementWrapper
# Tests that validate_goal()'s "moved" check correctly identifies a known
# displacement.
#
# How to use:
#   env = DummyMovementWrapper(base_env, device=device)
#   # During Alice's phase the target is teleported to target_local.  At Alice
#   # phase end, validate_goal should see it as moved (dist ≈ 0.30 m) and emit
#   # "Valid Goal (+1.0)".  If instead "Not Moved" appears, the initial_state
#   # capture or the distance formula is broken.
# ─────────────────────────────────────────────────────────────────────────────
class DummyMovementWrapper(AsyncDualPlayEnvWrapper):
    """
    Diagnostic wrapper: teleports the target to a known LOCAL position during
    Alice's phase, then reports what distance the movement check measured
    and whether validate_goal() accepted it.

    Detects:
    - Stale/wrong initial_states (measured distance ≈ 0 despite visible move)
    - Wrong coordinate frame in movement calculation
    - pos_threshold misconfigured

    Expected console output (at Alice phase end):
        [MoveCheck] Env X: measured_dist=0.300 expected≈0.300 ✓
    If 'measured_dist≈0.000' the initial_state is stale or in the wrong frame.
    """

    def __init__(
        self,
        env,
        device,
        alice_timesteps=100,
        bob_timesteps=200,
        target_local=None,
        num_objects=2,
    ):
        AsyncDualPlayEnvWrapper.__init__(
            self,
            env=env,
            device=device,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            num_objects=num_objects,
        )
        # Default target: teleport to [0.15, 0.5, 0.05] local.
        # Safe reset is [-0.15, 0.7, 0.023] (gravity-settled height matches wrapper.py).
        if target_local is None:
            target_local = [0.15, 0.5, 0.05]
        self._target_local = torch.tensor(
            target_local, dtype=torch.float32, device=device
        )
        safe_reset_local = torch.tensor(
            [-0.15, 0.7, 0.023], dtype=torch.float32, device=device
        )
        self._expected_dist = torch.norm(self._target_local - safe_reset_local).item()

    def step(self, action):
        # Snapshot ending Alice envs BEFORE super().step() transitions them to Bob.
        # phase_step is the count at the END of the previous step, so the last
        # Alice step has phase_step == alice_timesteps - 1 here.
        move_snapshots = []
        if self.episode_manager.initial_states is not None:
            ending_mask = self.episode_manager.is_alice_phase() & (
                self.episode_manager.phase_step
                >= self.episode_manager.alice_timesteps - 1
            )
            env_origins = self.env.scene.env_origins
            for env_id in ending_mask.nonzero(as_tuple=True)[0]:
                init_pos = self.episode_manager.initial_states[env_id, 0:3].clone()
                cur_pos_w = (
                    self.env.scene["target_object"].data.root_pos_w[env_id].clone()
                )
                cur_pos_l = cur_pos_w - env_origins[env_id]
                move_snapshots.append((env_id, init_pos, cur_pos_l))

        obs, rew, done, truncated, extras = super().step(action)

        # Teleport during Alice's phase, but skip steps 0 and 1.
        # Step 0: transition step — physics write not yet committed.
        # Step 1: initial_states capture step — must read the reset position, not the goal.
        alice_mask = self.episode_manager.is_alice_phase() & (
            self.episode_manager.phase_step > 1
        )
        alice_envs = alice_mask.nonzero(as_tuple=True)[0]

        if len(alice_envs) > 0:
            target_obj = self.env.scene["target_object"]
            env_origins = self.env.scene.env_origins[alice_envs]
            fixed_pos_world = env_origins + self._target_local
            fixed_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
            ).expand(len(alice_envs), -1)

            root_states = target_obj.data.root_state_w.clone()
            root_states[alice_envs, 0:3] = fixed_pos_world
            root_states[alice_envs, 3:7] = fixed_quat
            root_states[alice_envs, 7:] = 0.0
            target_obj.write_root_state_to_sim(
                root_states[alice_envs], env_ids=alice_envs
            )

        # Log movement check using pre-transition snapshots
        for env_id, init_pos, cur_pos_l in move_snapshots:
            measured = torch.norm(cur_pos_l - init_pos).item()
            ok = abs(measured - self._expected_dist) < 0.05
            status = "✓" if ok else "✗ BUG"
            print(
                f"[MoveCheck] Env {env_id.item()}: measured_dist={measured:.3f} "
                f"expected≈{self._expected_dist:.3f} {status} | "
                f"init_local={[round(x, 3) for x in init_pos.tolist()]} "
                f"cur_local={[round(x, 3) for x in cur_pos_l.tolist()]}",
                flush=True,
            )

        return obs, rew, done, truncated, extras


# ─────────────────────────────────────────────────────────────────────────────
# DiagnosticAliceWrapper
# Use for TEST 2 (Alice Exploration Sandbox) only — NOT for full training.
#
# Identical to AsyncDualPlayEnvWrapper except _handle_alice_completion uses:
#   alice_pos_req = 0.02m  (prod: 0.05m) — within ~1 std-dev of random-policy
#                                           action noise; early Alice stumbles
#                                           into successes more frequently.
#   _MIN_XY_DISP  = 0.05m (prod: 0.07m) — must stay > Bob's pos_threshold=0.04m.
#
# Also prints a [AliceDisp] summary line each Alice phase end so you can
# distinguish two failure modes:
#   avg XY ≈ 0.000 and not-moved = N/N  → Alice not touching object at all
#                                          (physics/reset bug)
#   avg XY > 0.02  but valid = 0        → filter or frame-mismatch bug
#   valid count rising                   → PPO buffer functioning, ready for T3
# ─────────────────────────────────────────────────────────────────────────────
class DiagnosticAliceWrapper(AsyncDualPlayEnvWrapper):
    """
    Diagnostic wrapper for Test 2 — relaxed thresholds + verbose logging.
    Do NOT use for full training runs.
    """

    # Relaxed thresholds — test-only.
    # INVARIANT: _MIN_XY_DISP must be > Bob's pos_threshold (0.04m).
    # If Alice's goal XY displacement ≤ Bob's success radius, Bob starts inside
    # the goal zone and wins in 1 step, leaving storage too small for a mini-batch.
    _ALICE_POS_REQ = 0.02   # m  (production: 0.05) — easy 3D movement requirement
    _ALICE_ROT_REQ = 0.25   # rad (unchanged)
    _MIN_XY_DISP   = 0.05   # m  (production: 0.07) — must stay > pos_threshold=0.04

    def _handle_alice_completion(self, obs_dict, env_ids):
        """Override of parent — same logic, different thresholds + [AliceDisp] log."""
        goal_state    = self._extract_object_states(obs_dict)
        initial_state = self.episode_manager.initial_states

        active_goal    = goal_state[env_ids]     # (N, 12) Euler local
        active_initial = initial_state[env_ids]  # (N, 12) Euler local

        # 1. Validate goal with relaxed movement threshold
        valid, val_reward, reasons = validate_goal(
            active_initial,
            active_goal,
            self.table_bounds,
            self.placement_bounds,
            pos_threshold=self._ALICE_POS_REQ,
            rot_threshold=self._ALICE_ROT_REQ,
        )

        # 2. Relaxed XY-displacement filter (consistent with _ALICE_POS_REQ)
        target_xy_disp = torch.norm(
            active_goal[:, 0:2] - active_initial[:, 0:2], dim=-1
        )
        if self.num_objects == 2:
            cube_xy_disp = torch.norm(
                active_goal[:, 6:8] - active_initial[:, 6:8], dim=-1
            )
            sufficient_xy = (target_xy_disp > self._MIN_XY_DISP) | (
                cube_xy_disp > self._MIN_XY_DISP
            )
        else:
            sufficient_xy = target_xy_disp > self._MIN_XY_DISP
        xy_fail    = valid & ~sufficient_xy
        val_reward = val_reward.clone()
        val_reward[xy_fail] = 0.0
        valid      = valid & sufficient_xy
        for i in range(len(env_ids)):
            if xy_fail[i].item():
                reasons[i] = "XY Disp Too Small (0.0)"

        self.delayed_alice_reward[env_ids] = val_reward

        # 3. Per-env logging + aggregate counts
        start_pos = active_initial[:, 0:3]
        final_pos = active_goal[:, 0:3]
        dist_xy   = torch.norm(final_pos[:, :2] - start_pos[:, :2], dim=-1)
        dist_z    = (final_pos[:, 2] - start_pos[:, 2]).abs()
        for i, env_id in enumerate(env_ids):
            rew = val_reward[i].item()
            if valid[i].item():
                self._iter_stats["valid_goals"] += 1
                print(
                    f"  [AliceRew] Env {env_id.item()}: {rew:+.1f} valid goal | "
                    f"XY={dist_xy[i]:.3f}m Z={dist_z[i]:.3f}m",
                    flush=True,
                )
            else:
                self._iter_stats["invalid_goals"] += 1
                print(
                    f"  [AliceRew] Env {env_id.item()}: {rew:+.1f} invalid goal "
                    f"({reasons[i]}) | XY={dist_xy[i]:.3f}m",
                    flush=True,
                )

        # Verbose displacement summary — key diagnostic output
        disp_3d     = torch.norm(final_pos - start_pos, dim=-1)
        n_total     = len(env_ids)
        n_valid     = valid.sum().item()
        n_not_moved = (disp_3d <= self._ALICE_POS_REQ).sum().item()
        print(
            f"  [AliceDisp] {n_valid}/{n_total} valid | "
            f"avg 3D={disp_3d.mean():.3f}m  avg XY={dist_xy.mean():.3f}m  "
            f"max XY={dist_xy.max():.3f}m | "
            f"not-moved(≤{self._ALICE_POS_REQ:.2f}m): {n_not_moved}/{n_total}",
            flush=True,
        )

        # 4. Store goal + marks for Bob
        self.episode_manager.store_goal_state(active_goal, env_ids)
        self.episode_manager.mark_goal_valid(env_ids, valid)
        self.episode_manager.mark_alice_base_reward(env_ids, val_reward)

        # 5. Transition valid envs to Bob
        valid_env_ids = env_ids[valid]
        if len(valid_env_ids) > 0:
            self.episode_manager.transition_to_bob(valid_env_ids)
            start_states = self.episode_manager.initial_states[valid_env_ids]
            origins      = self.env.scene.env_origins[valid_env_ids]

            t_pos_local = start_states[:, 0:3]
            t_quat      = _euler_xyz_to_quat(start_states[:, 3:6])
            self.env.scene["target_object"].write_root_pose_to_sim(
                torch.cat([t_pos_local + origins, t_quat], dim=-1),
                env_ids=valid_env_ids,
            )
            if self.num_objects == 2:
                c_pos_local = start_states[:, 6:9]
                c_quat      = _euler_xyz_to_quat(start_states[:, 9:12])
                self.env.scene["cube"].write_root_pose_to_sim(
                    torch.cat([c_pos_local + origins, c_quat], dim=-1),
                    env_ids=valid_env_ids,
                )
            reset_robot_joints(self.env, valid_env_ids)
            self.env.scene.write_data_to_sim()

        # 6. Reset invalid envs
        invalid_env_ids = env_ids[~valid]
        if len(invalid_env_ids) > 0:
            self.episode_manager.reset_episode(invalid_env_ids, reason="Alice Invalid Goal")
            reset_objects_to_fixed_safe_pose(self.env, invalid_env_ids)
            reset_robot_joints(self.env, invalid_env_ids)
            self.episode_manager.initial_states[invalid_env_ids] = (
                self._safe_reset_state.unsqueeze(0)
                .expand(len(invalid_env_ids), -1)
                .clone()
            )
            self.env.scene.write_data_to_sim()

        return valid, invalid_env_ids
