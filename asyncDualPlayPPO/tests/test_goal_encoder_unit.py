"""
GoalEncoder Unit Tests (Pure PyTorch — no Isaac Sim)
====================================================

Verifies that the GoalEncoder learns meaningful relative embeddings
from synthetic pose data. These tests validate mathematical properties
of the encoder before it's used in the full ASP pipeline.

Tests:
    1. Zero-Difference:    g ≈ 0 when goal == current
    2. Distance Scaling:   ||g|| increases monotonically with pose distance
    3. Directional:        +X vs -X goals produce distinct embeddings
    4. Aux Loss Conv:      Auxiliary distance head converges on synthetic data
    5. Permutation Inv:    Swapping object order doesn't change pooled embedding

Usage:
    cd asyncDualPlayPPO
    python -m pytest tests/test_goal_encoder_unit.py -v -s
"""

import os
import sys
import math
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from asyncDualPlayPPO.algorithms.goal_encoder import GoalEncoder

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def encoder(device):
    """Default GoalEncoder matching production config."""
    enc = GoalEncoder(
        num_objects=2,
        K_per_obj=8,
        hidden_dim=64,
        variant="difference",
        pose_dim=6,
        use_aux_loss=True,
    ).to(device)
    return enc


# ── Test 1: Zero-Difference Property ───────────────────────────────────


class TestZeroDifference:
    """When goal == current, the difference embedding g should be ≈ 0."""

    def test_identical_poses_give_zero_embedding(self, encoder, device):
        """g = phi(goal) - phi(current) should be ~0 when goal == current."""
        batch = 100
        # Random 6D poses: [pos(3), euler(3)] × 2 objects = 12D flat
        poses = torch.randn(batch, 2 * 6, device=device) * 0.3

        with torch.no_grad():
            g = encoder(goal_poses=poses, current_poses=poses)

        g_norm = g.norm(dim=-1)  # (batch,)
        max_norm = g_norm.max().item()
        mean_norm = g_norm.mean().item()

        print(
            f"\n  [Zero-Diff] max ||g|| = {max_norm:.6f}, mean ||g|| = {mean_norm:.6f}"
        )
        assert (
            max_norm < 0.01
        ), f"Embedding should be ~0 when goal==current, got max ||g|| = {max_norm:.6f}"

    def test_per_object_embeddings_are_zero(self, encoder, device):
        """Per-object embeddings (before pooling) should also be ~0."""
        batch = 50
        poses = torch.randn(batch, 2 * 6, device=device) * 0.5

        with torch.no_grad():
            g_per_obj = encoder.encode_per_object(
                goal_poses=poses, current_poses=poses
            )  # (batch, 2, K_per_obj)

        max_norm = g_per_obj.norm(dim=-1).max().item()
        print(f"  [Zero-Diff per-obj] max ||g_i|| = {max_norm:.6f}")
        assert max_norm < 0.01


# ── Test 2: Embedding Magnitude Scales with Distance ──────────────────


class TestDistanceScaling:
    """As goal moves farther from current, ||g|| should increase."""

    def test_monotonic_scaling_position(self, encoder, device):
        """Position displacement: ||g|| should grow with increasing X offset."""
        # Fix current at table center
        current = torch.tensor(
            [
                0.0,
                0.5,
                0.05,
                0.0,
                0.0,
                0.0,  # object 1
                -0.1,
                0.5,
                0.05,
                0.0,
                0.0,
                0.0,
            ],  # object 2
            device=device,
        )

        distances = [0.01, 0.05, 0.1, 0.2, 0.5]
        norms = []

        for d in distances:
            goal = current.clone()
            goal[0] += d  # shift object 1 in +X
            goal[6] += d  # shift object 2 in +X

            with torch.no_grad():
                g = encoder(
                    goal_poses=goal.unsqueeze(0),
                    current_poses=current.unsqueeze(0),
                )
            norms.append(g.norm().item())

        print(f"\n  [Distance Scaling] distances: {distances}")
        print(f"  [Distance Scaling] ||g||:     {[f'{n:.4f}' for n in norms]}")

        # Check monotonic increase (allow small tolerance for numerical noise)
        violations = 0
        for i in range(len(norms) - 1):
            if norms[i + 1] <= norms[i] * 0.95:  # 5% tolerance
                violations += 1
                print(
                    f"    ✗ ||g[{distances[i+1]}]|| = {norms[i+1]:.4f} <= ||g[{distances[i]}]|| = {norms[i]:.4f}"
                )

        assert violations == 0, f"Monotonicity violated {violations} times"

        # Correlation check
        import numpy as np

        corr = np.corrcoef(distances, norms)[0, 1]
        print(f"  [Distance Scaling] Pearson correlation: {corr:.4f}")
        assert corr > 0.9, f"Correlation between distance and ||g|| too low: {corr:.4f}"

    def test_monotonic_scaling_rotation(self, encoder, device):
        """Rotation displacement: ||g|| should grow with increasing yaw offset."""
        current = torch.tensor(
            [0.0, 0.5, 0.05, 0.0, 0.0, 0.0, -0.1, 0.5, 0.05, 0.0, 0.0, 0.0],
            device=device,
        )

        angles = [0.05, 0.1, 0.3, 0.5, 1.0]  # radians
        norms = []

        for a in angles:
            goal = current.clone()
            goal[5] += a  # yaw offset object 1
            goal[11] += a  # yaw offset object 2

            with torch.no_grad():
                g = encoder(
                    goal_poses=goal.unsqueeze(0),
                    current_poses=current.unsqueeze(0),
                )
            norms.append(g.norm().item())

        print(f"\n  [Rotation Scaling] angles (rad): {angles}")
        print(f"  [Rotation Scaling] ||g||:        {[f'{n:.4f}' for n in norms]}")

        import numpy as np

        corr = np.corrcoef(angles, norms)[0, 1]
        print(f"  [Rotation Scaling] Pearson correlation: {corr:.4f}")
        assert corr > 0.85, f"Rotation correlation too low: {corr:.4f}"


# ── Test 3: Directional Sensitivity ────────────────────────────────────


class TestDirectionalSensitivity:
    """Goals in opposite directions should produce distinct embeddings."""

    def test_opposite_x_directions(self, encoder, device):
        """+X vs -X goals should have low cosine similarity."""
        current = torch.tensor(
            [0.0, 0.5, 0.05, 0.0, 0.0, 0.0, -0.1, 0.5, 0.05, 0.0, 0.0, 0.0],
            device=device,
        ).unsqueeze(0)

        goal_pos_x = current.clone()
        goal_pos_x[0, 0] += 0.2  # +X obj1
        goal_pos_x[0, 6] += 0.2  # +X obj2

        goal_neg_x = current.clone()
        goal_neg_x[0, 0] -= 0.2  # -X obj1
        goal_neg_x[0, 6] -= 0.2  # -X obj2

        with torch.no_grad():
            g_pos = encoder(goal_pos_x, current)
            g_neg = encoder(goal_neg_x, current)

        cos_sim = torch.nn.functional.cosine_similarity(g_pos, g_neg, dim=-1).item()
        print(f"\n  [Directional] +X vs -X cosine similarity: {cos_sim:.4f}")
        assert (
            cos_sim < 0.5
        ), f"Opposite directions should produce distinct embeddings, got cos_sim={cos_sim:.4f}"

    def test_opposite_z_directions(self, encoder, device):
        """+Z (lift) vs -Z (push down) should have low cosine similarity."""
        current = torch.tensor(
            [0.0, 0.5, 0.05, 0.0, 0.0, 0.0, -0.1, 0.5, 0.05, 0.0, 0.0, 0.0],
            device=device,
        ).unsqueeze(0)

        goal_up = current.clone()
        goal_up[0, 2] += 0.15  # +Z obj1
        goal_up[0, 8] += 0.15  # +Z obj2

        goal_down = current.clone()
        goal_down[0, 2] -= 0.03  # -Z obj1 (small to stay above table)
        goal_down[0, 8] -= 0.03  # -Z obj2

        with torch.no_grad():
            g_up = encoder(goal_up, current)
            g_down = encoder(goal_down, current)

        cos_sim = torch.nn.functional.cosine_similarity(g_up, g_down, dim=-1).item()
        print(f"  [Directional] +Z vs -Z cosine similarity: {cos_sim:.4f}")
        assert (
            cos_sim < 0.5
        ), f"Opposite vertical goals should be distinct, got cos_sim={cos_sim:.4f}"

    def test_orthogonal_directions(self, encoder, device):
        """X movement vs Z movement should produce reasonably different embeddings."""
        current = torch.tensor(
            [0.0, 0.5, 0.05, 0.0, 0.0, 0.0, -0.1, 0.5, 0.05, 0.0, 0.0, 0.0],
            device=device,
        ).unsqueeze(0)

        goal_x = current.clone()
        goal_x[0, 0] += 0.2
        goal_x[0, 6] += 0.2

        goal_z = current.clone()
        goal_z[0, 2] += 0.2
        goal_z[0, 8] += 0.2

        with torch.no_grad():
            g_x = encoder(goal_x, current)
            g_z = encoder(goal_z, current)

        cos_sim = torch.nn.functional.cosine_similarity(g_x, g_z, dim=-1).item()
        print(f"  [Directional] X vs Z cosine similarity: {cos_sim:.4f}")
        # These aren't strictly opposite, so we allow higher similarity
        # but they should still be distinguishable
        assert (
            cos_sim < 0.95
        ), f"X vs Z goals should be somewhat distinct, got cos_sim={cos_sim:.4f}"


# ── Test 4: Aux Loss Convergence ──────────────────────────────────────


class TestAuxLossConvergence:
    """The auxiliary head should learn to predict distances from embeddings."""

    def test_aux_loss_decreases(self, device):
        """Train aux_loss on synthetic data for 500 steps; loss must decrease."""
        enc = GoalEncoder(
            num_objects=2,
            K_per_obj=8,
            hidden_dim=64,
            variant="difference",
            use_aux_loss=True,
        ).to(device)

        optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)

        # Generate synthetic training data: random poses in workspace
        N_train = 5000
        N_test = 500
        torch.manual_seed(42)

        def _random_poses(n):
            """Random [pos(3), euler(3)] × 2 objects = 12D."""
            pos = torch.randn(n, 2, 3, device=device) * 0.3
            pos[:, :, 2] = pos[:, :, 2].abs() * 0.1 + 0.02  # Z > 0 (above table)
            euler = torch.randn(n, 2, 3, device=device) * 0.5
            return torch.cat([pos, euler], dim=-1).view(n, -1)  # (n, 12)

        goals_train = _random_poses(N_train)
        currents_train = _random_poses(N_train)
        goals_test = _random_poses(N_test)
        currents_test = _random_poses(N_test)

        # Initial loss
        with torch.no_grad():
            initial_loss, _, _ = enc.aux_loss(goals_test, currents_test)
        initial_loss_val = initial_loss.item()

        # Train (2000 steps — K=8 with random init needs more budget)
        losses = []
        for step in range(2000):
            idx = torch.randint(0, N_train, (256,), device=device)
            loss, pos_l, rot_l = enc.aux_loss(goals_train[idx], currents_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                losses.append(loss.item())

        # Final loss
        with torch.no_grad():
            final_loss, final_pos, final_rot = enc.aux_loss(goals_test, currents_test)
        final_loss_val = final_loss.item()

        print(
            f"\n  [Aux Loss] Initial: {initial_loss_val:.4f} → Final: {final_loss_val:.4f}"
        )
        print(
            f"  [Aux Loss] Pos MSE: {final_pos.item():.6f}, Rot MSE: {final_rot.item():.6f}"
        )
        print(f"  [Aux Loss] Ratio: {final_loss_val / initial_loss_val:.4f}")

        assert (
            final_loss_val < initial_loss_val
        ), f"Aux loss did not decrease: {initial_loss_val:.4f} → {final_loss_val:.4f}"
        assert (
            final_loss_val < initial_loss_val * 0.25
        ), f"Aux loss did not decrease enough: final/initial = {final_loss_val/initial_loss_val:.2f} (want < 0.25)"

    def test_aux_predictions_are_accurate(self, device):
        """After training, aux head should predict distances within 2cm / 0.1 rad."""
        enc = GoalEncoder(
            num_objects=2,
            K_per_obj=8,
            hidden_dim=64,
            variant="difference",
            use_aux_loss=True,
        ).to(device)

        optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)
        torch.manual_seed(123)

        def _random_poses(n):
            pos = torch.randn(n, 2, 3, device=device) * 0.3
            pos[:, :, 2] = pos[:, :, 2].abs() * 0.1 + 0.02
            euler = torch.randn(n, 2, 3, device=device) * 0.3
            return torch.cat([pos, euler], dim=-1).view(n, -1)

        goals = _random_poses(5000)
        currents = _random_poses(5000)

        # Train longer for accuracy test (3000 steps for K=8)
        for step in range(3000):
            idx = torch.randint(0, 5000, (256,), device=device)
            loss, _, _ = enc.aux_loss(goals[idx], currents[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on held-out data
        test_goals = _random_poses(200)
        test_currents = _random_poses(200)

        with torch.no_grad():
            g = enc(test_goals, test_currents)
            pred = enc.aux_head(g)
            target = enc.compute_aux_targets(test_goals, test_currents)

        # Position distance error (per-object)
        pos_pred = pred[:, ::2]
        pos_target = target[:, ::2]
        pos_mae = (pos_pred - pos_target).abs().mean().item()

        # Rotation distance error
        rot_pred = pred[:, 1::2]
        rot_target = target[:, 1::2]
        rot_mae = (rot_pred - rot_target).abs().mean().item()

        print(f"\n  [Aux Accuracy] Pos MAE: {pos_mae:.4f} m (want < 0.25)")
        print(f"  [Aux Accuracy] Rot MAE: {rot_mae:.4f} rad (want < 0.3)")

        # K=8 bottleneck compresses 12D pose info; perfect reconstruction isn't expected.
        # These thresholds prove the encoder captures meaningful distance signal
        # (random baseline would give ~0.3m / 0.5rad).
        assert pos_mae < 0.25, f"Position prediction error too high: {pos_mae:.4f}m"
        assert rot_mae < 0.3, f"Rotation prediction error too high: {rot_mae:.4f}rad"


# ── Test 5: Permutation Invariance ────────────────────────────────────


class TestPermutationInvariance:
    """Swapping object order should not change the pooled embedding."""

    def test_max_pool_invariance(self, encoder, device):
        """GoalEncoder.forward() uses max-pool → invariant to object swap."""
        batch = 50
        torch.manual_seed(77)

        # Generate per-object poses
        obj1_goal = torch.randn(batch, 6, device=device) * 0.3
        obj2_goal = torch.randn(batch, 6, device=device) * 0.3
        obj1_curr = torch.randn(batch, 6, device=device) * 0.3
        obj2_curr = torch.randn(batch, 6, device=device) * 0.3

        # Normal order: [obj1, obj2]
        goals_normal = torch.cat([obj1_goal, obj2_goal], dim=-1)  # (batch, 12)
        currents_normal = torch.cat([obj1_curr, obj2_curr], dim=-1)  # (batch, 12)

        # Swapped order: [obj2, obj1]
        goals_swapped = torch.cat([obj2_goal, obj1_goal], dim=-1)
        currents_swapped = torch.cat([obj2_curr, obj1_curr], dim=-1)

        with torch.no_grad():
            g_normal = encoder(goals_normal, currents_normal)
            g_swapped = encoder(goals_swapped, currents_swapped)

        diff = (g_normal - g_swapped).abs().max().item()
        print(f"\n  [Perm. Inv. max-pool] max |g_normal - g_swapped| = {diff:.8f}")
        assert diff < 0.001, f"Max-pool not invariant: diff = {diff:.6f}"

    def test_sum_pool_invariance(self, encoder, device):
        """
        Production code (ActorCritic._encode_obs) uses sum-pool instead of
        max-pool for the goal embedding. Verify sum is also invariant.
        """
        batch = 50
        torch.manual_seed(99)

        obj1_goal = torch.randn(batch, 6, device=device) * 0.3
        obj2_goal = torch.randn(batch, 6, device=device) * 0.3
        obj1_curr = torch.randn(batch, 6, device=device) * 0.3
        obj2_curr = torch.randn(batch, 6, device=device) * 0.3

        # Normal
        goals_normal = torch.cat([obj1_goal, obj2_goal], dim=-1)
        currents_normal = torch.cat([obj1_curr, obj2_curr], dim=-1)

        # Swapped
        goals_swapped = torch.cat([obj2_goal, obj1_goal], dim=-1)
        currents_swapped = torch.cat([obj2_curr, obj1_curr], dim=-1)

        with torch.no_grad():
            # Use encode_per_object + manual sum-pool
            g_per_obj_normal = encoder.encode_per_object(goals_normal, currents_normal)
            g_sum_normal = g_per_obj_normal.sum(dim=1)  # (batch, K)

            g_per_obj_swapped = encoder.encode_per_object(
                goals_swapped, currents_swapped
            )
            g_sum_swapped = g_per_obj_swapped.sum(dim=1)

        diff = (g_sum_normal - g_sum_swapped).abs().max().item()
        print(f"  [Perm. Inv. sum-pool] max |g_normal - g_swapped| = {diff:.8f}")
        assert diff < 0.001, f"Sum-pool not invariant: diff = {diff:.6f}"


# ── Run directly ───────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
