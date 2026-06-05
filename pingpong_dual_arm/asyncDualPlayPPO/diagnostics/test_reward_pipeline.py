"""
Test 1 — Reward Pipeline Integrity (--test_reward_pipeline).

Uses DummyBobWrapper which:
  1. Teleports target to [0.15, 0.5, 0.05] at Alice step 10 → valid goal.
  2. Teleports target to stored goal at Bob step 10 → sparsity fires.

All actions are replaced with safe default joint positions inside the wrapper
so the arm stays in a non-colliding configuration throughout the test.
"""

import math
import torch


def rot_distance_euler(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff.abs().max(dim=-1)[0]


def _unit_test_rot_wraparound():
    a1 = torch.tensor([[0.0, 0.0, 3.0]])
    b1 = torch.tensor([[0.0, 0.0, -3.0]])
    d1 = rot_distance_euler(a1, b1).item()
    assert d1 < 0.30, f"FAIL: expected <0.30 but got {d1:.4f}"
    a2 = torch.tensor([[0.0, 0.0, 3.13]])
    b2 = torch.tensor([[0.0, 0.0, -3.13]])
    d2 = rot_distance_euler(a2, b2).item()
    assert d2 < 0.03, f"FAIL: expected <0.03 but got {d2:.4f}"
    print(f"  [T1-unit] rot wraparound: d1={d1:.4f} d2={d2:.4f}  PASS")


def run_test1(env, episode_manager, device, n_iterations: int = 120):
    """
    Step through the environment.  Expect the Bob sparse reward
    (+1 per-object + +5 completion) to fire at ~step 111
    (Alice teleport at step 10 → Alice end at 100 → Bob teleport at 10
     → reward at 11) when stagger is disabled.
    """
    print("\n" + "=" * 70)
    print("TEST 1 — Reward Pipeline Integrity (DummyBobWrapper)")
    print("=" * 70)
    _unit_test_rot_wraparound()

    for it in range(n_iterations):
        _, rewards, _, _, _ = env.step(torch.empty(0))
        max_rew = rewards.max().item()
        if max_rew > 0:
            print(f"  [T1] iter {it:3d}  max_rew={max_rew:.2f}  PASS")
            break
    else:
        print("  [T1] No positive reward detected — check TB logs")
    print("=" * 70 + "\n")
