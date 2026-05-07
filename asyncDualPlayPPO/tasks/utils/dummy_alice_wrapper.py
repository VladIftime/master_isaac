"""
Unified diagnostic wrappers for Asymmetric Self-Play.

All wrappers share the same teleport mechanism:
  - Teleport ONCE at a fixed step (default step 10 into each phase)
  - Call reset_robot_joints + write_data_to_sim after every teleport
  - Use _alice_teleported / _bob_teleported per-env flags

Hierarchy:
  DiagnosticAliceWrapper(AsyncDualPlayEnvWrapper)  ← base for all
  DummyAliceWrapper(DiagnosticWrapperBase)
  DummyBobWrapper(DummyAliceWrapper)
  DummyGoalDistanceWrapper(DummyAliceWrapper)
  DummyMovementWrapper(DiagnosticWrapperBase)
"""

import math
import torch
from isaaclab.managers import SceneEntityCfg
from asyncDualPlayPPO.tasks.utils.wrapper import AsyncDualPlayEnvWrapper
from asyncDualPlayPPO.tasks.utils.observations import goal_distance, _euler_xyz_to_quat
from asyncDualPlayPPO.utils.goal_validator import validate_goal
from asyncDualPlayPPO.tasks.utils.events import reset_objects_to_fixed_safe_pose, reset_robot_joints


# ── Default safe action: arm at init joint positions, gripper open ────────
# This keeps the arm in the URDF-initialised pose (above table) so the
# teleported object is NOT in collision with the arm on the next physics step.
_SAFE_ARM_POS = (1.57, -1.57, 1.57, -1.57, -1.57, 0.0)
_SAFE_GRIPPER = 0.0       # BinaryJointPositionActionCfg: 0 = open


def _safe_action_tensor(device: torch.device, n: int = 1) -> torch.Tensor:
    """Return (n, 7) safe default action: arm at init pose, gripper open."""
    a = torch.tensor([*_SAFE_ARM_POS, _SAFE_GRIPPER], device=device)
    return a.unsqueeze(0).expand(n, -1).clone()


# ══════════════════════════════════════════════════════════════════════════
#  DiagnosticWrapperBase — base for all diagnostic wrappers
# ══════════════════════════════════════════════════════════════════════════

class DiagnosticWrapperBase(AsyncDualPlayEnvWrapper):
    """
    Base diagnostic wrapper.

    At ``alice_teleport_step`` (default 10) the target object is snapped to
    a fixed local position.  At ``bob_teleport_step`` (default 10) the target
    is snapped to the stored goal.  After every teleport the robot joints are
    reset and physics is flushed.

    The wrapper replaces the incoming action with a safe default action (arm
    at init joint positions, gripper open) so that the arm never collides with
    the teleported object during diagnostic runs.

    Subclasses can override ``_after_alice_teleport``,
    ``_after_bob_teleport`` and ``_handle_alice_completion``.
    """

    _ALICE_POS_REQ = 0.05   # m  (same as production)
    _ALICE_ROT_REQ = 0.25   # rad
    _MIN_XY_DISP   = 0.07   # m  (same as production)

    def __init__(
        self,
        env,
        device,
        alice_timesteps=100,
        bob_timesteps=200,
        max_goals_per_episode=5,
        num_objects=2,
        arm_config="default",
        alice_teleport_step=10,
        bob_teleport_step=10,
        teleport_pos=None,
        use_safe_action=True,
    ):
        AsyncDualPlayEnvWrapper.__init__(
            self,
            env=env,
            alice_timesteps=alice_timesteps,
            bob_timesteps=bob_timesteps,
            max_goals_per_episode=max_goals_per_episode,
            num_objects=num_objects,
            device=device,
            arm_config=arm_config,
        )
        self.alice_teleport_step = alice_teleport_step
        self.bob_teleport_step = bob_teleport_step
        self.teleport_pos = torch.tensor(
            teleport_pos or [0.15, 0.5, 0.05], device=device
        )
        self._use_safe_action = use_safe_action
        # Per-env flags to prevent repeat teleports
        self._alice_teleported = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._bob_teleported = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    # ── Public step ──────────────────────────────────────────────────────
    def step(self, action):
        # Replace action with safe default (prevents arm-object collision)
        if self._use_safe_action:
            action = _safe_action_tensor(self.device, self.num_envs)

        obs, rew, done, truncated, extras = super().step(action)

        # ── Clear teleport flags for envs entering a fresh phase ───────
        # phase_step==1 marks the first step after reset/transition
        # (episode_manager.step() incremented it from 0).
        fresh_alice = self.episode_manager.is_alice_phase() & (self.episode_manager.phase_step == 1)
        self._alice_teleported[fresh_alice] = False
        fresh_bob = self.episode_manager.is_bob_phase() & (self.episode_manager.phase_step == 1)
        self._bob_teleported[fresh_bob] = False

        # ── Alice teleport (once per env, at fixed step) ────────────────
        alice_mask = (
            self.episode_manager.is_alice_phase()
            & (self.episode_manager.phase_step == self.alice_teleport_step)
            & ~self._alice_teleported
        )
        alice_ids = alice_mask.nonzero(as_tuple=True)[0]
        if len(alice_ids) > 0:
            self._teleport_object_to(alice_ids, self.teleport_pos)
            self._alice_teleported[alice_ids] = True
            self._after_alice_teleport(alice_ids)

        # ── Bob teleport (once per env, at fixed step) ─────────────────
        bob_mask = (
            self.episode_manager.is_bob_phase()
            & (self.episode_manager.phase_step == self.bob_teleport_step)
            & ~self._bob_teleported
        )
        bob_ids = bob_mask.nonzero(as_tuple=True)[0]
        if len(bob_ids) > 0 and self.episode_manager.goal_states is not None:
            self._teleport_object_to_goal(bob_ids)
            self._bob_teleported[bob_ids] = True
            self._after_bob_teleport(bob_ids)

        return obs, rew, done, truncated, extras

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._alice_teleported.zero_()
        self._bob_teleported.zero_()
        return result

    # ── Teleport helpers ─────────────────────────────────────────────────

    def _teleport_object_to(self, env_ids, local_pos):
        """Teleport target_object to a fixed local position."""
        target = self.env.scene["target_object"]
        origins = self.env.scene.env_origins[env_ids]
        world_pos = origins + local_pos.to(self.device)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        quat = identity_quat.unsqueeze(0).expand(len(env_ids), -1)
        rs = target.data.root_state_w.clone()
        rs[env_ids, 0:3] = world_pos
        rs[env_ids, 3:7] = quat
        rs[env_ids, 7:] = 0.0
        target.write_root_state_to_sim(rs[env_ids], env_ids=env_ids)
        reset_robot_joints(self.env, env_ids)
        self.env.scene.write_data_to_sim()

    def _teleport_object_to_goal(self, env_ids):
        """Teleport target_object to the stored goal position."""
        gs = self.episode_manager.goal_states
        if gs is None:
            return
        target = self.env.scene["target_object"]
        origins = self.env.scene.env_origins[env_ids]
        goal_pos_world = gs[env_ids, 0:3] + origins
        goal_quat = _euler_xyz_to_quat(gs[env_ids, 3:6])
        rs = target.data.root_state_w.clone()
        rs[env_ids, 0:3] = goal_pos_world
        rs[env_ids, 3:7] = goal_quat
        rs[env_ids, 7:] = 0.0
        target.write_root_state_to_sim(rs[env_ids], env_ids=env_ids)
        reset_robot_joints(self.env, env_ids)
        self.env.scene.write_data_to_sim()

    # ── Hooks for subclasses ─────────────────────────────────────────────
    def _after_alice_teleport(self, env_ids):
        """Called after Alice-phase teleport (override in subclasses)."""
        pass

    def _after_bob_teleport(self, env_ids):
        """Called after Bob-phase teleport (override in subclasses)."""
        pass

    def _handle_alice_completion(self, obs_dict, env_ids):
        """Override of parent — same logic, uses class-level thresholds."""
        goal_state = self._extract_object_states(obs_dict)
        initial_state = self.episode_manager.initial_states

        active_goal = goal_state[env_ids]
        active_initial = initial_state[env_ids]

        valid, val_reward, reasons = validate_goal(
            active_initial,
            active_goal,
            self.table_bounds,
            self.placement_bounds,
            pos_threshold=self._ALICE_POS_REQ,
            rot_threshold=self._ALICE_ROT_REQ,
        )

        # XY-displacement filter
        target_xy_disp = torch.norm(active_goal[:, 0:2] - active_initial[:, 0:2], dim=-1)
        if self.num_objects == 2:
            cube_xy_disp = torch.norm(active_goal[:, 6:8] - active_initial[:, 6:8], dim=-1)
            sufficient_xy = (target_xy_disp > self._MIN_XY_DISP) | (cube_xy_disp > self._MIN_XY_DISP)
        else:
            sufficient_xy = target_xy_disp > self._MIN_XY_DISP
        xy_fail = valid & ~sufficient_xy
        val_reward = val_reward.clone()
        val_reward[xy_fail] = 0.0
        valid = valid & sufficient_xy
        for i in range(len(env_ids)):
            if xy_fail[i].item():
                reasons[i] = "XY Disp Too Small (0.0)"

        self.delayed_alice_reward[env_ids] = val_reward

        # Stats
        start_pos = active_initial[:, 0:3]
        final_pos = active_goal[:, 0:3]
        dist_3d = torch.norm(final_pos - start_pos, dim=-1)
        dist_xy = torch.norm(final_pos[:, :2] - start_pos[:, :2], dim=-1)
        self._alice_phase_initialized[env_ids] = False
        n = len(env_ids)
        self._iter_stats["valid_goals"] += int(valid.sum().item())
        self._iter_stats["invalid_goals"] += int((~valid).sum().item())
        self._iter_stats["alice_total"] += n
        self._iter_stats["alice_disp_3d_sum"] += dist_3d.sum().item()
        self._iter_stats["alice_disp_xy_sum"] += dist_xy.sum().item()
        self._iter_stats["alice_disp_xy_max"] = max(
            self._iter_stats["alice_disp_xy_max"], dist_xy.max().item()
        )
        self._iter_stats["alice_not_moved"] += int(
            (dist_3d <= self._ALICE_POS_REQ).sum().item()
        )

        # Store goal + marks for Bob
        self.episode_manager.store_goal_state(active_goal, env_ids)
        self.episode_manager.mark_goal_valid(env_ids, valid)
        self.episode_manager.mark_alice_base_reward(env_ids, val_reward)

        # Transition valid envs to Bob
        valid_env_ids = env_ids[valid]
        if len(valid_env_ids) > 0:
            self.episode_manager.transition_to_bob(valid_env_ids)
            start_states = self.episode_manager.initial_states[valid_env_ids]
            origins = self.env.scene.env_origins[valid_env_ids]
            t_pos_local = start_states[:, 0:3]
            t_quat = _euler_xyz_to_quat(start_states[:, 3:6])
            self.env.scene["target_object"].write_root_pose_to_sim(
                torch.cat([t_pos_local + origins, t_quat], dim=-1),
                env_ids=valid_env_ids,
            )
            if self.num_objects == 2:
                c_pos_local = start_states[:, 6:9]
                c_quat = _euler_xyz_to_quat(start_states[:, 9:12])
                self.env.scene["cube"].write_root_pose_to_sim(
                    torch.cat([c_pos_local + origins, c_quat], dim=-1),
                    env_ids=valid_env_ids,
                )
            reset_robot_joints(self.env, valid_env_ids)
            self.env.scene.write_data_to_sim()

        # Reset invalid envs
        invalid_env_ids = env_ids[~valid]
        if len(invalid_env_ids) > 0:
            self.episode_manager.reset_episode(invalid_env_ids, reason="Alice Invalid Goal")
            reset_objects_to_fixed_safe_pose(self.env, invalid_env_ids)
            reset_robot_joints(self.env, invalid_env_ids)
            self.episode_manager.initial_states[invalid_env_ids] = (
                self._safe_reset_state.unsqueeze(0)
                .expand(len(invalid_env_ids), -1).clone()
            )
            self.env.scene.write_data_to_sim()

        return valid, invalid_env_ids


# ══════════════════════════════════════════════════════════════════════════
#  DummyAliceWrapper
#  Teleport target to a fixed position at Alice step N, producing a valid goal.
# ══════════════════════════════════════════════════════════════════════════

class DummyAliceWrapper(DiagnosticWrapperBase):
    """
    Teleport target to a fixed local position at ``alice_teleport_step``
    (default 10) during Alice's phase.  The movement is detected by
    validate_goal at Alice phase end, producing a valid goal for Bob.

    Does NOT teleport during Bob phase (bob_teleport_step is disabled by
    setting it to a very large number).
    """

    def __init__(self, env, device, alice_timesteps=100, bob_timesteps=200,
                 num_objects=2, alice_teleport_step=10):
        super().__init__(
            env=env, device=device,
            alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
            num_objects=num_objects,
            alice_teleport_step=alice_teleport_step,
            bob_teleport_step=999999,    # disabled
        )
        print(f"  [DummyAlice] Teleport at Alice step {alice_teleport_step}, "
              f"target→{self.teleport_pos.tolist()}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  DummyBobWrapper
#  Alice teleport + Bob teleport to goal → full reward pipeline test.
# ══════════════════════════════════════════════════════════════════════════

class DummyBobWrapper(DummyAliceWrapper):
    """
    Extends DummyAliceWrapper with a Bob-phase teleport to the goal.
    At Bob step ``teleport_step`` (default 10) the target is snapped to the
    stored goal, so the sparse +1/+5 reward fires on the next env step.

    Use with ``--test_reward_pipeline`` or ``--test_bob_reward``.
    """

    def __init__(self, env, device, alice_timesteps=100, bob_timesteps=200,
                 teleport_step=10, num_objects=2):
        super().__init__(
            env=env, device=device,
            alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
            num_objects=num_objects,
        )
        self.bob_teleport_step = teleport_step
        print(f"  [DummyBob] Teleport at Bob step {teleport_step}", flush=True)

    def _after_bob_teleport(self, env_ids):
        print(
            f"[DummyBob] Teleported target→goal for {len(env_ids)} envs "
            f"at Bob step {self.bob_teleport_step}",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════
#  DummyGoalDistanceWrapper
#  Same as DummyBobWrapper + immediate distance measurement after teleport.
# ══════════════════════════════════════════════════════════════════════════

class DummyGoalDistanceWrapper(DummyAliceWrapper):
    """
    At Bob teleport step, snaps both target and cube to their stored goals
    and measures goal_distance() immediately.  Asserts ~0.

    Use with ``--dummy_goal_distance``.
    """

    def __init__(self, env, device, alice_timesteps=100, bob_timesteps=200,
                 teleport_step=10, num_objects=2):
        super().__init__(
            env=env, device=device,
            alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
            num_objects=num_objects,
        )
        self.bob_teleport_step = teleport_step

    def _after_bob_teleport(self, env_ids):
        gs = self.episode_manager.goal_states
        if gs is None:
            return

        # Snap cube to its stored goal too
        if self.num_objects == 2:
            cube = self.env.scene["cube"]
            origins = self.env.scene.env_origins[env_ids]
            c_pos_world = gs[env_ids, 6:9] + origins
            c_quat = _euler_xyz_to_quat(gs[env_ids, 9:12])
            rs = cube.data.root_state_w.clone()
            rs[env_ids, 0:3] = c_pos_world
            rs[env_ids, 3:7] = c_quat
            rs[env_ids, 7:] = 0.0
            cube.write_root_state_to_sim(rs[env_ids], env_ids=env_ids)
            reset_robot_joints(self.env, env_ids)
            self.env.scene.write_data_to_sim()

        # Measure distances
        tgt_dists = goal_distance(self.env, SceneEntityCfg("target_object"))
        cube_dists = goal_distance(self.env, SceneEntityCfg("cube")) if self.num_objects == 2 else None

        for eid in env_ids:
            t_pos = tgt_dists[eid, 0].item()
            t_rot = tgt_dists[eid, 1].item()
            if cube_dists is not None:
                c_pos = cube_dists[eid, 0].item()
                c_rot = cube_dists[eid, 1].item()
                ok = t_pos < 0.01 and t_rot < 0.01 and c_pos < 0.01 and c_rot < 0.01
                status = "✓" if ok else "✗ BUG"
                print(
                    f"[DistCheck] Env {eid.item()}: "
                    f"target pos={t_pos:.4f} rot={t_rot:.4f} | "
                    f"cube pos={c_pos:.4f} rot={c_rot:.4f} {status}",
                    flush=True,
                )
            else:
                ok = t_pos < 0.01 and t_rot < 0.01
                status = "✓" if ok else "✗ BUG"
                print(
                    f"[DistCheck] Env {eid.item()}: "
                    f"target pos={t_pos:.4f} rot={t_rot:.4f} {status}",
                    flush=True,
                )


# ══════════════════════════════════════════════════════════════════════════
#  DummyMovementWrapper
#  Alice-phase teleport + movement-distance verification at phase end.
# ══════════════════════════════════════════════════════════════════════════

class DummyMovementWrapper(DiagnosticWrapperBase):
    """
    At a fixed Alice step, teleport the target and verify validate_goal
    detects the movement correctly.  Bob teleport is disabled.
    """

    def __init__(self, env, device, alice_timesteps=100, bob_timesteps=200,
                 target_local=None, num_objects=2):
        tp = list(target_local) if target_local else [0.15, 0.5, 0.05]
        super().__init__(
            env=env, device=device,
            alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
            num_objects=num_objects,
            alice_teleport_step=10,
            bob_teleport_step=999999,
            teleport_pos=tp,
        )
        safe_reset_local = torch.tensor([-0.15, 0.7, 0.023], device=device)
        self._expected_dist = torch.norm(
            self.teleport_pos - safe_reset_local
        ).item()

        # Snapshot initial states for later comparison
        self._move_snapshots = []

    def step(self, action):
        # Snapshot Alice-phase-ending envs BEFORE super().step transitions them
        self._move_snapshots = []
        if self.episode_manager.initial_states is not None:
            ending = (
                self.episode_manager.is_alice_phase()
                & (self.episode_manager.phase_step >= self.episode_manager.alice_timesteps - 1)
            )
            for eid in ending.nonzero(as_tuple=True)[0]:
                init = self.episode_manager.initial_states[eid, 0:3].clone()
                cur_w = self.env.scene["target_object"].data.root_pos_w[eid].clone()
                cur_l = cur_w - self.env.scene.env_origins[eid]
                self._move_snapshots.append((eid, init, cur_l))

        obs, rew, done, truncated, extras = super().step(action)

        # Log movement check using pre-transition snapshots
        for eid, init_pos, cur_pos_l in self._move_snapshots:
            measured = torch.norm(cur_pos_l - init_pos).item()
            ok = abs(measured - self._expected_dist) < 0.05
            status = "✓" if ok else "✗ BUG"
            print(
                f"[MoveCheck] Env {eid.item()}: measured_dist={measured:.3f} "
                f"expected≈{self._expected_dist:.3f} {status} | "
                f"init_local={[round(x,3) for x in init_pos.tolist()]} "
                f"cur_local={[round(x,3) for x in cur_pos_l.tolist()]}",
                flush=True,
            )

        return obs, rew, done, truncated, extras


# ══════════════════════════════════════════════════════════════════════════
#  DiagnosticAliceWrapper (relaxed thresholds)
# ══════════════════════════════════════════════════════════════════════════

class DiagnosticAliceWrapper(DiagnosticWrapperBase):
    """
    Relaxed-threshold version for Test 2 (Alice Exploration Sandbox):
      - ``alice_pos_req = 0.02`` m (prod: 0.05)
      - ``_MIN_XY_DISP = 0.05`` m (prod: 0.07)
      - Prints per-env [AliceEnd] summary.
    """

    _ALICE_POS_REQ = 0.02
    _ALICE_ROT_REQ = 0.25
    _MIN_XY_DISP = 0.05

    def __init__(self, env, device, alice_timesteps=100, bob_timesteps=200,
                 num_objects=2):
        super().__init__(
            env=env, device=device,
            alice_timesteps=alice_timesteps, bob_timesteps=bob_timesteps,
            num_objects=num_objects,
            alice_teleport_step=10,
            bob_teleport_step=999999,
        )
        self._alice_dense_accum = torch.zeros(self.num_envs, device=self.device)
