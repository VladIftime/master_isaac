import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

# ---------------------------------------------------------------------------
# Multi-Categorical distribution helper
# ---------------------------------------------------------------------------


class MultiCategorical:
    """
    Product of N independent Categorical distributions (one per action dimension).

    Used to model a discretized action space where each dimension has `num_bins`
    possible values, matching the original ASP paper's multi-categorical head.

    Bin layout (11 bins):
        bin 0 → -max_delta   (full negative)
        bin 5 → 0            (no movement)
        bin 10 → +max_delta  (full positive)

    Log-probability and entropy are the SUM across all dimensions.
    """

    def __init__(self, logits: torch.Tensor):
        """
        Args:
            logits: (batch, num_dims, num_bins)
        """
        self.num_dims = logits.shape[1]
        self.dists = [Categorical(logits=logits[:, i, :]) for i in range(self.num_dims)]

    def sample(self) -> torch.Tensor:
        """Returns (batch, num_dims) integer bin indices."""
        return torch.stack([d.sample() for d in self.dists], dim=-1)

    def log_prob(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bin_indices: (batch, num_dims) — integer dtype
        Returns:
            (batch,) summed log-probabilities
        """
        return sum(
            self.dists[i].log_prob(bin_indices[:, i]) for i in range(self.num_dims)
        )

    def entropy(self) -> torch.Tensor:
        """Returns (batch,) summed entropy."""
        return sum(d.entropy() for d in self.dists)


# ---------------------------------------------------------------------------
# Permutation-Invariant Encoder (Fix 7)
# ---------------------------------------------------------------------------


class PermInvEncoder(nn.Module):
    """
    Permutation-invariant object encoder using element-wise max-pooling (ASP paper).

    Each object's features are independently encoded by a shared MLP, then
    max-pooled across objects to produce an embedding invariant to object ordering.
    """

    def __init__(self, per_obj_dim: int, emb_dim: int = 512):
        super().__init__()
        self.per_obj_dim = per_obj_dim
        self.emb_dim = emb_dim
        self.obj_encoder = nn.Sequential(
            nn.Linear(per_obj_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
        nn.init.orthogonal_(self.obj_encoder[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.obj_encoder[3].weight, gain=np.sqrt(2))

    def forward(
        self, robot_state: torch.Tensor, obj_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            robot_state:  (batch, robot_dim)
            obj_features: (batch, num_objects * per_obj_dim)
        Returns:
            (batch, robot_dim + emb_dim)
        """
        batch = obj_features.shape[0]
        num_objs = obj_features.shape[1] // self.per_obj_dim
        objs = obj_features.reshape(batch * num_objs, self.per_obj_dim)
        enc = self.obj_encoder(objs).reshape(batch, num_objs, self.emb_dim)
        pooled = enc.sum(dim=1)
        return torch.cat([robot_state, pooled], dim=-1)


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """
    Actor-critic supporting:
      - Continuous Gaussian (default, legacy)
      - Multi-categorical discretized actions (use_multicategorical: true)
      - Optional LSTM trunk (use_lstm: true)
      - Optional permutation-invariant encoder (use_pi_encoder: true)

    Multi-categorical output
    ------------------------
    Actor outputs `num_cat_dims * num_bins` logits, reshaped to
    (batch, num_cat_dims, num_bins) and fed into MultiCategorical.

    `act()` / `act_with_hidden()` return bin indices (float32) as "actions",
    so they drop cleanly into the existing RolloutStorage (which stores float).

    `evaluate(obs, states, actions)` interprets `actions` as bin indices (casts
    to int64 internally) when multi-categorical is enabled.

    `bins_to_delta(bin_indices)` converts stored bin indices to continuous
    deltas that can be zero-padded and sent to RMPFlow.
    """

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        model_cfg,
        asymmetric=False,
    ):
        super().__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
            self.use_lstm = False
            self.use_pi_encoder = False
            self.use_multicategorical = False
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            self.use_lstm = model_cfg.get("use_lstm", False)
            self.use_pi_encoder = model_cfg.get("use_pi_encoder", False)
            self.use_multicategorical = model_cfg.get("use_multicategorical", False)

        # --- Multi-categorical params ---
        if self.use_multicategorical:
            self.num_cat_dims = model_cfg.get("num_cat_dims", 4)
            self.num_bins = model_cfg.get("num_bins", 11)
            self.max_delta = model_cfg.get("max_delta_m", 0.05)
            actor_out_dim = self.num_cat_dims * self.num_bins
        else:
            actor_out_dim = actions_shape[0]

        actor_in_dim = obs_shape[0]

        # --- Permutation-Invariant Encoder ---
        if self.use_pi_encoder:
            self.robot_state_dim = model_cfg.get("robot_state_dim", 8)
            per_obj_dim = model_cfg.get("pi_obj_dim", 15)
            pi_emb_dim = model_cfg.get("pi_emb_dim", 512)
            self.pi_encoder = PermInvEncoder(per_obj_dim, pi_emb_dim)
            actor_in_dim = self.robot_state_dim + pi_emb_dim

        # --- Actor: MLP or LSTM ---
        if self.use_lstm:
            lstm_hidden = model_cfg.get("lstm_hidden_size", actor_hidden_dim[-1])
            trunk_dims = (
                actor_hidden_dim[:-1] if len(actor_hidden_dim) > 1 else actor_hidden_dim
            )
            trunk_out = actor_hidden_dim[-1]
            self.actor_trunk = _build_mlp(
                actor_in_dim, trunk_dims, trunk_out, activation
            )
            self.actor_lstm = nn.LSTMCell(trunk_out, lstm_hidden)
            self.actor_head = nn.Linear(lstm_hidden, actor_out_dim)
            self.lstm_hidden_size = lstm_hidden
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        else:
            self.actor = _build_mlp(
                actor_in_dim, actor_hidden_dim, actor_out_dim, activation
            )

        # --- Critic ---
        critic_in = states_shape[0] if asymmetric else obs_shape[0]
        self.critic = _build_mlp(critic_in, critic_hidden_dim, 1, activation)

        # Log-std only used for Gaussian mode
        if not self.use_multicategorical:
            self.log_std = nn.Parameter(
                np.log(initial_std) * torch.ones(*actions_shape)
            )

        if self.use_lstm:
            print(
                f"Actor trunk: {self.actor_trunk}\nActor LSTM: {self.actor_lstm}\nActor head: {self.actor_head}"
            )
        else:
            print(self.actor)
        print(self.critic)

        if not self.use_lstm:
            actor_scales = [np.sqrt(2)] * len(actor_hidden_dim) + [0.01]
            _init_weights(self.actor, actor_scales)
        critic_scales = [np.sqrt(2)] * len(critic_hidden_dim) + [1.0]
        _init_weights(self.critic, critic_scales)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, observations: torch.Tensor) -> torch.Tensor:
        if self.use_pi_encoder:
            robot = observations[:, : self.robot_state_dim]
            objs = observations[:, self.robot_state_dim :]
            return self.pi_encoder(robot, objs)
        return observations

    def _actor_forward(self, observations: torch.Tensor, hidden_state=None):
        """
        Returns (raw_output, new_hidden).
        raw_output is (batch, actor_out_dim):
          - Gaussian mode:  action means
          - MC mode:        num_cat_dims * num_bins logits
        """
        enc = self._encode_obs(observations)
        if self.use_lstm:
            feat = self.actor_trunk(enc)
            if hidden_state is None:
                h = torch.zeros(enc.shape[0], self.lstm_hidden_size, device=enc.device)
                c = torch.zeros(enc.shape[0], self.lstm_hidden_size, device=enc.device)
            else:
                h, c = hidden_state
            h, c = self.actor_lstm(feat, (h, c))
            return self.actor_head(h), (h.detach(), c.detach())
        return self.actor(enc), None

    def _make_distribution(self, raw_output: torch.Tensor):
        """Build either MultivariateNormal or MultiCategorical from actor output."""
        if self.use_multicategorical:
            logits = raw_output.view(-1, self.num_cat_dims, self.num_bins)
            return MultiCategorical(logits)
        else:
            scale_tril = torch.diag(self.log_std.exp())
            return MultivariateNormal(raw_output, scale_tril=scale_tril)

    def bins_to_delta(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert integer bin indices (batch, num_cat_dims) to env-ready deltas.

        XYZ (dims 0-2): delta = (bin - center) / center * max_delta
          bin 0  →  -max_delta,  bin 5 → 0.0,  bin 10 → +max_delta

        Gripper (dim 3): sign of normalized → -1 / 0 / +1
          Collapses continuous delta to three states so BinaryJointPositionActionCfg
          does not oscillate between open/close on bins near center.

        Returns (batch, num_cat_dims) float tensor.
        """
        center = (self.num_bins - 1) / 2.0
        normalized = (bin_indices.float() - center) / center  # [-1, 1]
        xyz = normalized[:, :3] * self.max_delta
        rot_xy = normalized[:, 3:5] * 0.5  # 0.5 rad (~28 deg)
        gripper = torch.sign(normalized[:, 5:6])  # -1 / 0 / +1
        return torch.cat([xyz, rot_xy, gripper], dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        """
        Sample action.

        Returns:
            actions:          (batch, action_dim)   — bin indices (MC) or continuous (Gaussian)
            actions_log_prob: (batch,)
            value:            (batch, 1)
            mu:               (batch, action_dim)   — logits sum (MC) or mean (Gaussian)
            sigma:            (batch, action_dim)   — zeros (MC) or log_std (Gaussian)
        """
        raw, _ = self._actor_forward(observations)
        dist = self._make_distribution(raw)
        value = self.critic(states if self.asymmetric else observations)

        if self.use_multicategorical:
            bin_indices = dist.sample()  # (batch, num_cat_dims) int
            actions_log_prob = dist.log_prob(bin_indices)  # (batch,)
            actions = bin_indices.float()  # store as float
            mu = raw.view(-1, self.num_cat_dims, self.num_bins).sum(-1)  # sentinel
            sigma = torch.zeros_like(actions)
        else:
            actions = dist.sample()
            actions_log_prob = dist.log_prob(actions)
            mu = raw
            sigma = self.log_std.repeat(raw.shape[0], 1)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            mu.detach(),
            sigma.detach(),
        )

    def act_with_hidden(self, observations, states, hidden_state=None):
        """Same as act() but also propagates LSTM hidden state."""
        raw, new_hidden = self._actor_forward(observations, hidden_state)
        dist = self._make_distribution(raw)
        value = self.critic(states if self.asymmetric else observations)

        if self.use_multicategorical:
            bin_indices = dist.sample()
            actions_log_prob = dist.log_prob(bin_indices)
            actions = bin_indices.float()
            mu = raw.view(-1, self.num_cat_dims, self.num_bins).sum(-1)
            sigma = torch.zeros_like(actions)
        else:
            actions = dist.sample()
            actions_log_prob = dist.log_prob(actions)
            mu = raw
            sigma = self.log_std.repeat(raw.shape[0], 1)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            mu.detach(),
            sigma.detach(),
            new_hidden,
        )

    def act_inference(self, observations):
        """Deterministic action for evaluation."""
        raw, _ = self._actor_forward(observations)
        if self.use_multicategorical:
            logits = raw.view(-1, self.num_cat_dims, self.num_bins)
            # Greedy: argmax per dimension
            return logits.argmax(dim=-1).float()
        return raw  # action means for Gaussian

    def evaluate(self, observations, states, actions):
        """
        Evaluate log-probabilities and entropy for the given actions.

        For multi-categorical: `actions` is interpreted as float bin indices
        (cast to int64 internally).
        For Gaussian: `actions` is continuous values.
        """
        raw, _ = self._actor_forward(observations)  # zero hidden for LSTM
        dist = self._make_distribution(raw)
        value = self.critic(states if self.asymmetric else observations)

        if self.use_multicategorical:
            bin_indices = actions.long()  # (batch, num_cat_dims)
            actions_log_prob = dist.log_prob(bin_indices)  # (batch,)
            entropy = dist.entropy()  # (batch,)
            mu = raw.view(-1, self.num_cat_dims, self.num_bins).sum(-1)
            sigma = torch.zeros_like(actions)
        else:
            actions_log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            mu = raw
            sigma = self.log_std.repeat(raw.shape[0], 1)

        return actions_log_prob, entropy, value, mu, sigma


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _build_mlp(in_dim, hidden_dims, out_dim, activation):
    """Build a feedforward network with the given hidden layer structure."""
    layers = [nn.Linear(in_dim, hidden_dims[0]), activation]
    for i in range(len(hidden_dims)):
        out = out_dim if i == len(hidden_dims) - 1 else hidden_dims[i + 1]
        layers.append(nn.Linear(hidden_dims[i], out))
        if i < len(hidden_dims) - 1:
            layers.append(activation)
    return nn.Sequential(*layers)


def _init_weights(sequential, scales):
    """Orthogonal weight initialisation with per-layer gain scaling."""
    for idx, module in enumerate(m for m in sequential if isinstance(m, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=scales[idx])


def get_activation(act_name):
    """Return the nn.Module activation corresponding to act_name."""
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "crelu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if act_name not in activations:
        raise ValueError(
            f"Unknown activation function: '{act_name}'. Choose from {list(activations)}"
        )
    return activations[act_name]
