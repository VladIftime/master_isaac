"""
Goal encoder E: maps per-object (goal_pose, current_pose) pairs to a pooled
goal embedding g ∈ R^K.

Based on Sukhbaatar et al. (2018) "Learning Goal Embeddings via Self-Play for
Hierarchical Reinforcement Learning", adapted for multi-object manipulation
with permutation invariance.

Architecture (per-object):
    1. phi MLP maps 6D pose (pos3 + euler3) → R^{K_per_obj}  [matches observation space]
    2. Difference variant: g_i = phi(goal_i) - phi(current_i)
       Absolute variant:   g_i = phi(goal_i)

ZYX Euler angles are used throughout to match the observation pipeline.  Linear
subtraction phi(goal) - phi(current) is well-defined on Euler angles, which is
why they are preferred over quaternions for the difference variant.

Permutation invariance:
    Per-object embeddings g_i are max-pooled across objects to produce the
    final K-dimensional goal vector.  This ensures the encoder is invariant
    to object ordering in the observation tensor.

Typical usage:
    encoder = GoalEncoder(num_objects=2, K_per_obj=6, variant="difference")
    g = encoder(goal_poses, current_poses)   # (batch, K_per_obj)
    # g is used for additive injection into the actor
"""

import torch
import torch.nn as nn
from typing import Literal


class GoalEncoder(nn.Module):
    """
    Per-object goal encoder with permutation-invariant pooling.

    Produces a K_per_obj-dimensional goal embedding from per-object
    (goal_pose, current_pose) pairs, pooled across objects.

    Args:
        num_objects:  Number of objects (2 for target_object + cube).
        K_per_obj:    Output embedding dimension per object (pooled to this).
        hidden_dim:   Hidden layer width for the phi MLP.
        variant:      "difference" (phi(g) - phi(s)) or "absolute" (phi(g)).
        pose_dim:     Dimension of each pose (6: pos(3) + euler(3), matching observation space).
    """

    def __init__(
        self,
        num_objects: int = 2,
        K_per_obj: int = 6,
        hidden_dim: int = 64,
        variant: Literal["difference", "absolute"] = "difference",
        pose_dim: int = 6,  # pos(3) + euler(3) — matches ZYX Euler observation space
        use_aux_loss: bool = True,
    ):
        super().__init__()

        if variant not in ("difference", "absolute"):
            raise ValueError(
                f"variant must be 'difference' or 'absolute', got {variant!r}"
            )

        self.variant = variant
        self.num_objects = num_objects
        self.K_per_obj = K_per_obj
        self.pose_dim = pose_dim
        self.use_aux_loss = use_aux_loss
        self._obj_state_dim = (
            6  # pos(3) + euler(3) — matches paper's Euler representation
        )

        # phi: 6D pose → R^{K_per_obj}
        # Two-layer MLP, no final non-linearity (per paper)
        # Uses tanh intermediate (paper Section 2.4) to bound internal representations
        self.phi = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, K_per_obj),
            # No activation — output is unconstrained R^K
        )

        # Initialize with small weights for stable early training
        for m in self.phi:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        if self.use_aux_loss:
            self.aux_head = nn.Linear(K_per_obj, 2 * num_objects)

    def _preprocess_pose(self, pose_6d: torch.Tensor) -> torch.Tensor:
        """Pass-through: observations already deliver ZYX Euler poses (pos3 + euler3)."""
        return pose_6d

    def forward(
        self,
        goal_poses: torch.Tensor,
        current_poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute permutation-invariant goal embedding.

        Args:
            goal_poses:    (batch, num_objects * 6) — per-object [pos(3), euler(3)]
            current_poses: (batch, num_objects * 6) — per-object [pos(3), euler(3)]

        Returns:
            g: (batch, K_per_obj) — pooled goal embedding.
        """
        batch = goal_poses.shape[0]

        # Reshape to per-object: (batch, num_objects, 6)
        goals = goal_poses.view(batch, self.num_objects, 6)
        currents = current_poses.view(batch, self.num_objects, 6)

        goals_6d = self._preprocess_pose(goals)
        currents_6d = self._preprocess_pose(currents)

        # Flatten for shared MLP: (batch * num_objects, 6)
        goals_flat = goals_6d.reshape(batch * self.num_objects, self.pose_dim)
        currents_flat = currents_6d.reshape(batch * self.num_objects, self.pose_dim)

        # Encode through phi
        phi_goal = self.phi(goals_flat)  # (batch * num_objects, K_per_obj)
        phi_curr = self.phi(currents_flat)  # (batch * num_objects, K_per_obj)

        # Per-object goal embedding
        if self.variant == "difference":
            g_per_obj = phi_goal - phi_curr  # (batch * num_objects, K_per_obj)
        else:
            g_per_obj = phi_goal

        # Reshape back: (batch, num_objects, K_per_obj)
        g_per_obj = g_per_obj.view(batch, self.num_objects, self.K_per_obj)

        # Permutation-invariant pooling (max-pool across objects)
        g_pooled, _ = g_per_obj.max(dim=1)  # (batch, K_per_obj)

        return g_pooled

    def encode_per_object(
        self,
        goal_poses: torch.Tensor,
        current_poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-object goal embeddings WITHOUT pooling.

        Useful for injecting per-object g_i into the PermInvEncoder's
        per-object feature vectors.

        Args:
            goal_poses:    (batch, num_objects * 6) — per-object [pos(3), euler(3)]
            current_poses: (batch, num_objects * 6) — per-object [pos(3), euler(3)]

        Returns:
            (batch, num_objects, K_per_obj) — per-object embeddings.
        """
        batch = goal_poses.shape[0]

        goals = goal_poses.view(batch, self.num_objects, 6)
        currents = current_poses.view(batch, self.num_objects, 6)

        goals_6d = self._preprocess_pose(goals)
        currents_6d = self._preprocess_pose(currents)

        goals_flat = goals_6d.reshape(batch * self.num_objects, self.pose_dim)
        currents_flat = currents_6d.reshape(batch * self.num_objects, self.pose_dim)

        phi_goal = self.phi(goals_flat)
        phi_curr = self.phi(currents_flat)

        if self.variant == "difference":
            g_per_obj = phi_goal - phi_curr
        else:
            g_per_obj = phi_goal

        return g_per_obj.view(batch, self.num_objects, self.K_per_obj)

    def compute_aux_targets(
        self, s_star: torch.Tensor, s_t: torch.Tensor
    ) -> torch.Tensor:
        import math

        targets = []
        for i in range(self.num_objects):
            base = i * self._obj_state_dim  # now 6D per object
            pos_star = s_star[:, base : base + 3]
            euler_star = s_star[:, base + 3 : base + 6]  # roll, pitch, yaw
            pos_t = s_t[:, base : base + 3]
            euler_t = s_t[:, base + 3 : base + 6]

            pos_dist = torch.norm(pos_star - pos_t, dim=-1, keepdim=True)

            # Max Euler diff with wraparound (matches paper + goal_distance())
            euler_diff = euler_star - euler_t
            euler_diff = (euler_diff + math.pi) % (2 * math.pi) - math.pi
            rot_dist = euler_diff.abs().max(dim=-1, keepdim=True)[0]

            targets.append(pos_dist)
            targets.append(rot_dist)
        return torch.cat(targets, dim=-1)

    def aux_loss(self, s_star: torch.Tensor, s_t: torch.Tensor):
        import torch.nn.functional as F

        g = self.forward(s_star, s_t)
        pred = self.aux_head(g)
        target = self.compute_aux_targets(s_star, s_t)

        pos_pred = pred[:, ::2]
        rot_pred = pred[:, 1::2]
        pos_tgt = target[:, ::2]
        rot_tgt = target[:, 1::2]

        pos_loss = F.mse_loss(pos_pred, pos_tgt)
        rot_loss = F.mse_loss(rot_pred, rot_tgt)

        return pos_loss + 2.0 * rot_loss, pos_loss, rot_loss

    def extra_repr(self) -> str:
        return (
            f"num_objects={self.num_objects}, K_per_obj={self.K_per_obj}, "
            f"variant={self.variant!r}, pose_dim={self.pose_dim}"
        )


def build_goal_encoder(
    num_objects: int = 2,
    K_per_obj: int = 6,
    hidden_dim: int = 64,
    variant: Literal["difference", "absolute"] = "difference",
    device: str = "cuda",
) -> GoalEncoder:
    """Convenience factory: build and move encoder to device."""
    enc = GoalEncoder(
        num_objects=num_objects,
        K_per_obj=K_per_obj,
        hidden_dim=hidden_dim,
        variant=variant,
    )
    return enc.to(device)
