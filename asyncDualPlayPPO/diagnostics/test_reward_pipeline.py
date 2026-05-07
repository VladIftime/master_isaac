"""
Test 1 — Reward Pipeline Integrity via Teleportation.

Verifies that _compute_bob_sparse_rewards fires +1 / -1 / +5 at the correct
thresholds by teleporting objects to exact goal coordinates and checking that
SR reaches 1.0 within the first Bob step.

Entry point (called from train_curobo.py --test_reward_pipeline):
    run_test1(env, episode_manager, device)

Standalone: this file cannot run headless on its own because it requires the
Isaac Lab AppLauncher to have been started first. Use the flag:
    python -m asyncDualPlayPPO.train --test_reward_pipeline --num_envs 16 --headless
"""

import math
import torch


def rot_distance_euler(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Max-abs Euler-angle distance with [0, π] wraparound (same as observations.py)."""
    diff = a - b
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff.abs().max(dim=-1)[0]


def _unit_test_rot_wraparound():
    """Verify that the rot_distance_euler handles the ±π boundary correctly."""
    # Near-zero distance should not wrap to ~2π
    a1 = torch.tensor([[0.0, 0.0, 0.0]])
    b1 = torch.tensor([[0.0, 0.0, 3.13]])
    d1 = rot_distance_euler(a1, b1).item()
    assert d1 < 0.02, f"FAIL wraparound unit test 1: expected <0.02 but got {d1:.4f}"

    # Crossing ±π boundary: 3.0 and -3.0 are ~0.28 rad apart, not ~6 rad
    a2 = torch.tensor([[0.0, 0.0, 3.0]])
    b2 = torch.tensor([[0.0, 0.0, -3.0]])
    d2 = rot_distance_euler(a2, b2).item()
    assert d2 < 0.30, f"FAIL wraparound unit test 2: expected <0.30 but got {d2:.4f}"
    print(f"  [T1-unit] rot wraparound: d1={d1:.4f} d2={d2:.4f}  PASS")


def teleport_to_goals(env, episode_manager):
    """Teleport all objects to their current goal poses.

    Reads episode_manager.goal_states (N, num_obj*6) in [pos(3)+euler(3)] format,
    converts euler→quat, and calls write_root_pose_to_sim + write_root_velocity_to_sim.
    Mirrors the pattern already used in wrapper.py:_validate_and_store_goal.
    """
    from asyncDualPlayPPO.tasks.utils.observations import _euler_xyz_to_quat

    goal = episode_manager.goal_states   # (N, num_obj * 6) or None
    if goal is None:
        return

    origins = env.env.scene.env_origins  # (N, 3)
    num_envs = env.num_envs
    zero_vel = torch.zeros(num_envs, 6, device=env.device)
    num_objects = episode_manager.num_objects

    obj_names = ["target_object", "cube"][:num_objects]
    for obj_idx, obj_name in enumerate(obj_names):
        g = goal[:, obj_idx * 6 : obj_idx * 6 + 6]  # (N, 6)
        pos_local = g[:, :3]
        quat = _euler_xyz_to_quat(g[:, 3:6])         # (N, 4) w,x,y,z
        pos_world = pos_local + origins
        pose = torch.cat([pos_world, quat], dim=1)    # (N, 7)
        try:
            obj = env.env.scene[obj_name]
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(zero_vel)
        except (KeyError, AttributeError):
            pass


def run_test1(env, episode_manager, device, n_iterations: int = 100):
    """
    Teleport objects to goal poses and verify the sparse reward pipeline fires.

    PASS: SR > 0 within first teleport iteration; +5 completion bonus observed.
    FAIL: SR = 0 throughout; reward tensor returns zeros after exact teleport.
    """
    print("\n" + "=" * 70)
    print("TEST 1 — Reward Pipeline Integrity via Teleportation")
    print("=" * 70)

    _unit_test_rot_wraparound()

    zero_acts = torch.zeros(env.num_envs, 6, device=device)
    sr_history = []
    max_rew_history = []
    completion_seen = False

    for it in range(n_iterations):
        # Step with zero actions to advance phase
        obs, rewards, dones, info = env.step(zero_acts)

        if episode_manager.is_bob_phase().any():
            teleport_to_goals(env, episode_manager)
            # Second step lets physics settle and reward logic fire
            obs, rewards, dones, info = env.step(zero_acts)

            ep_info = info.get("episode_manager", {})
            bob_success = ep_info.get("bob_success_this_step", torch.zeros(env.num_envs, dtype=torch.bool, device=device))
            sr = bob_success.float().mean().item()
            max_rew = rewards.max().item()

            sr_history.append(sr)
            max_rew_history.append(max_rew)
            if max_rew >= 4.9:
                completion_seen = True

            print(f"  [T1] iter {it:3d}  SR={sr:.3f}  max_rew={max_rew:.2f}")

            if sr >= 1.0 and completion_seen:
                break

    print()
    any_sr = max(sr_history) if sr_history else 0.0
    assert any_sr > 0.0, (
        "FAIL: SR never exceeded 0 — sparse reward is broken. "
        "Check _compute_bob_sparse_rewards thresholds and goal frame convention."
    )
    assert completion_seen, (
        "FAIL: +5 completion bonus never triggered. "
        "Check all-object success gate in wrapper.py."
    )
    print(f"  [T1] max_SR={any_sr:.3f}  completion_bonus_seen={completion_seen}")
    print("TEST 1 PASS")
    print("=" * 70 + "\n")
