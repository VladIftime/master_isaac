"""
Unit tests for GoalEncoder — no Isaac Sim required.

Run:
    cd asyncDualPlayPPO
    python -m pytest tests/test_goal_encoder_unit.py -v
  or
    python tests/test_goal_encoder_unit.py

Five properties are verified:

  1. at_goal_is_zero   — when current == goal, difference encoder → ||g|| = 0 exactly
  2. norm_decreases    — synthesize a trajectory where current linearly approaches goal;
                         ||g|| must have a strictly negative linear trend (r < −0.5)
  3. perm_invariant    — swapping object order gives identical pooled embedding
  4. distinct_goals    — two different goals produce different embeddings
  5. aux_loss_trains   — aux head MSE decreases over 50 gradient steps on synthetic data
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

from asyncDualPlayPPO.algorithms.goal_encoder import GoalEncoder

# ── helpers ───────────────────────────────────────────────────────────────────

NUM_OBJECTS = 2
K = 6
POSE_DIM = 6  # pos(3) + euler(3)
FLAT = NUM_OBJECTS * POSE_DIM  # 12


def _enc(device="cpu") -> GoalEncoder:
    enc = GoalEncoder(
        num_objects=NUM_OBJECTS,
        K_per_obj=K,
        hidden_dim=64,
        variant="difference",
        use_aux_loss=True,
    ).to(device)
    enc.eval()
    return enc


def _rand_pose(batch: int = 1) -> torch.Tensor:
    """Random (batch, FLAT) pose tensor."""
    return torch.randn(batch, FLAT)


# ── test 1: at-goal embedding is exactly zero ─────────────────────────────────

def test_at_goal_is_zero():
    """phi(goal) - phi(goal) = 0 for any input."""
    enc = _enc()
    goal = _rand_pose(4)
    with torch.no_grad():
        g = enc(goal, goal)  # current == goal
    assert g.shape == (4, K), f"wrong shape {g.shape}"
    assert g.abs().max().item() < 1e-5, (
        f"||g|| should be 0 when current==goal, got max={g.abs().max().item():.2e}"
    )
    print("  PASS  at_goal_is_zero")


# ── test 2: ||g|| decreases as current approaches goal ───────────────────────

def test_norm_decreases_toward_goal():
    """Linearly interpolate current → goal over T steps; ||g|| must trend down."""
    enc = _enc()
    T = 50
    goal = _rand_pose(1).expand(T, -1)
    start = _rand_pose(1).expand(T, -1)

    # current[t] = start + (t / (T-1)) * (goal - start)
    alpha = torch.linspace(0, 1, T).unsqueeze(1)
    current = start + alpha * (goal - start)

    with torch.no_grad():
        g = enc(goal, current)
    norms = g.norm(dim=-1).numpy()  # (T,)

    # Pearson correlation of norms with time index; should be < −0.5
    t_idx = torch.arange(T).float().numpy()
    corr = float(
        torch.corrcoef(torch.from_numpy(
            __import__("numpy").stack([norms, t_idx])
        ))[0, 1].item()
    )
    assert corr < -0.5, (
        f"||g|| should decrease as current→goal (Pearson corr with step expected < -0.5, "
        f"got {corr:.4f}). Encoder may be receiving scrambled features."
    )
    print(f"  PASS  norm_decreases  (Pearson corr={corr:.4f})")


# ── test 3: permutation invariance ───────────────────────────────────────────

def test_permutation_invariant():
    """Swapping the two objects in goal/current must not change the pooled embedding."""
    enc = _enc()
    goal = _rand_pose(8)
    curr = _rand_pose(8)

    # Swap object 0 and object 1 (each 6D)
    goal_swap = torch.cat([goal[:, POSE_DIM:], goal[:, :POSE_DIM]], dim=-1)
    curr_swap = torch.cat([curr[:, POSE_DIM:], curr[:, :POSE_DIM]], dim=-1)

    with torch.no_grad():
        g_orig = enc(goal, curr)
        g_swap = enc(goal_swap, curr_swap)

    max_diff = (g_orig - g_swap).abs().max().item()
    assert max_diff < 1e-5, (
        f"Encoder is NOT permutation-invariant: max |g_orig - g_swap| = {max_diff:.2e}"
    )
    print(f"  PASS  permutation_invariant  (max_diff={max_diff:.2e})")


# ── test 4: distinct goals produce distinct embeddings ───────────────────────

def test_distinct_goals():
    """Two very different goals must produce different embeddings."""
    enc = _enc()
    curr = torch.zeros(1, FLAT)
    goal_a = torch.ones(1, FLAT) * 1.0
    goal_b = torch.ones(1, FLAT) * -1.0

    with torch.no_grad():
        g_a = enc(goal_a, curr)
        g_b = enc(goal_b, curr)

    diff = (g_a - g_b).norm().item()
    assert diff > 0.01, (
        f"Two opposite goals collapsed to same embedding (||g_a - g_b|| = {diff:.4f})"
    )
    print(f"  PASS  distinct_goals  (||g_a - g_b|| = {diff:.4f})")


# ── test 5: aux loss decreases under gradient descent ────────────────────────

def test_aux_loss_trains():
    """The aux head should be trainable: loss must fall over 50 steps."""
    enc = GoalEncoder(
        num_objects=NUM_OBJECTS,
        K_per_obj=K,
        hidden_dim=64,
        variant="difference",
        use_aux_loss=True,
    )
    enc.train()
    optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)

    torch.manual_seed(0)
    goal = _rand_pose(32)
    curr = _rand_pose(32)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        loss, _, _ = enc.aux_loss(goal, curr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Aux loss did not decrease: start={losses[0]:.4f}  end={losses[-1]:.4f}"
    )
    print(f"  PASS  aux_loss_trains  ({losses[0]:.4f} → {losses[-1]:.4f})")


# ── runner ────────────────────────────────────────────────────────────────────

_TESTS = [
    test_at_goal_is_zero,
    test_norm_decreases_toward_goal,
    test_permutation_invariant,
    test_distinct_goals,
    test_aux_loss_trains,
]


def main():
    print("\nGoalEncoder unit tests (no Isaac Sim)\n" + "=" * 45)
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print("=" * 45)
    print(f"  {passed} passed  {failed} failed\n")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
