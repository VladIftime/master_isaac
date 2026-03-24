"""
Goal encoder E: maps (s*, s_t) -> g in R^K.

Two variants from Sukhbaatar et al. (2018) "Learning Goal Embeddings via
Self-Play for Hierarchical Reinforcement Learning":

  - DifferenceEncoder:  g = phi(s*) - phi(s_t)    (most common in practice)
  - AbsoluteEncoder:    g = phi(s*)               (ignores current state)

phi is a two-layer MLP without a final non-linearity, so the output lives in
an unconstrained R^K latent space.  Bob's policy is then conditioned on g
instead of the raw goal state, allowing the latent space to be shaped jointly
during training.

Input state format (14 dims, matching _extract_object_states):
    [target_pos(3), target_quat(4), cube_pos(3), cube_quat(4)]
All coordinates in the LOCAL environment frame.

Typical usage:
    encoder = GoalEncoder(state_dim=14, embed_dim=8, variant="difference")
    g = encoder(s_star, s_t)           # (batch, 8)
    bob_obs = torch.cat([robot_obs, g], dim=-1)
"""

import torch
import torch.nn as nn
from typing import Literal


class GoalEncoder(nn.Module):
    """
    Encodes a (goal_state, current_state) pair into a K-dimensional goal vector.

    Args:
        state_dim:  Dimensionality of the state input (14 for 2-object scenes).
        embed_dim:  Output embedding dimension K.  Paper recommends 8-16.
        hidden_dim: Hidden layer width for the phi MLP.
        variant:    "difference" (phi(s*) - phi(s_t)) or "absolute" (phi(s*)).
    """

    def __init__(
        self,
        state_dim: int = 14,
        embed_dim: int = 8,
        hidden_dim: int = 64,
        variant: Literal["difference", "absolute"] = "difference",
    ):
        super().__init__()

        if variant not in ("difference", "absolute"):
            raise ValueError(
                f"variant must be 'difference' or 'absolute', got {variant!r}"
            )

        self.variant = variant
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # phi: state -> R^K, no final non-linearity (per paper)
        self.phi = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            # No activation — output is unconstrained
        )

    def forward(self, s_star: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """
        Compute goal embedding.

        Args:
            s_star: Goal state (batch, state_dim) in LOCAL frame.
            s_t:    Current state (batch, state_dim) in LOCAL frame.

        Returns:
            g: Goal embedding (batch, embed_dim).
        """
        if self.variant == "difference":
            return self.phi(s_star) - self.phi(s_t)
        else:  # absolute
            return self.phi(s_star)

    def encode_goal(self, s_star: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """Alias for forward(), provided for readability at call sites."""
        return self.forward(s_star, s_t)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, embed_dim={self.embed_dim}, "
            f"variant={self.variant!r}"
        )


def build_goal_encoder(
    state_dim: int = 14,
    embed_dim: int = 8,
    hidden_dim: int = 64,
    variant: Literal["difference", "absolute"] = "difference",
    device: str = "cuda",
) -> GoalEncoder:
    """Convenience factory: build and move encoder to device."""
    enc = GoalEncoder(
        state_dim=state_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        variant=variant,
    )
    return enc.to(device)
