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
        pooled, _ = enc.max(dim=1)
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
      - Optional goal encoder bottleneck (use_goal_encoder: true)

    Goal encoder (HSP / Charlie paper)
    ------------------------------------
    When use_goal_encoder is true, Bob's raw goal+distance portion of the
    observation is replaced by a low-dimensional goal embedding g computed
    per-object and max-pooled.  The encoder is trained end-to-end with RL
    loss (gradients detached during ABC to prevent latent distortion).

    The goal embedding is injected additively after the first actor hidden
    layer (paper Section 2.4):  h = activation(W1 @ enc + W_g @ g)

    The critic always sees the FULL raw observation (no bottleneck).

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
            self.use_goal_encoder = False
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            self.use_lstm = model_cfg.get("use_lstm", False)
            self.use_pi_encoder = model_cfg.get("use_pi_encoder", False)
            self.use_multicategorical = model_cfg.get("use_multicategorical", False)
            self.use_goal_encoder = model_cfg.get("use_goal_encoder", False)

        # --- Multi-categorical params ---
        if self.use_multicategorical:
            self.num_cat_dims = model_cfg.get("num_cat_dims", 4)
            self.num_bins = model_cfg.get("num_bins", 11)
            self.max_delta = model_cfg.get("max_delta_m", 0.05)
            actor_out_dim = self.num_cat_dims * self.num_bins
        else:
            actor_out_dim = actions_shape[0]

        actor_in_dim = obs_shape[0]

        # --- Goal Encoder (HSP / Charlie paper) ---
        # Must be initialized BEFORE PermInvEncoder since it changes per_obj_dim.
        self.goal_encoder = None
        self._ge_num_objects = 0
        self._ge_K_per_obj = 0
        if self.use_goal_encoder and model_cfg is not None:
            from ...goal_encoder import GoalEncoder

            self._ge_num_objects = model_cfg.get("num_objects", 2)
            self._ge_K_per_obj = model_cfg.get("goal_embed_dim", 6)
            ge_hidden = model_cfg.get("goal_encoder_hidden_dim", 64)
            ge_variant = model_cfg.get("goal_encoder_variant", "difference")

            self.goal_encoder = GoalEncoder(
                num_objects=self._ge_num_objects,
                K_per_obj=self._ge_K_per_obj,
                hidden_dim=ge_hidden,
                variant=ge_variant,
            )

            # Observation slicing config for extracting goal/object poses
            # Bob obs layout: [robot(8) | obj1(15)+goal1(7)+dist1(2) | obj2(15)+goal2(7)+dist2(2)]
            # With PI encoder: per_obj_dim = 24 = 15 + 7 + 2
            # After goal encoding: per_obj_dim = 15 + K_per_obj (drop raw goal + dist)
            self._ge_robot_dim = model_cfg.get("robot_state_dim", 8)
            self._ge_obj_state_dim = 15  # pos(3)+quat(4)+linvel(3)+angvel(3)+dist(1)+contact(1)
            self._ge_goal_dim = 7   # pos(3)+quat(4) per object
            self._ge_dist_dim = 2   # pos_dist(1)+rot_dist(1) per object
            self._ge_raw_per_obj = self._ge_obj_state_dim + self._ge_goal_dim + self._ge_dist_dim  # 24

        # --- Permutation-Invariant Encoder ---
        if self.use_pi_encoder:
            self.robot_state_dim = model_cfg.get("robot_state_dim", 8)

            if self.use_goal_encoder:
                # With goal encoder: per-object features become [obj_state(15) | g_i(K)]
                # instead of [obj_state(15) | goal(7) | dist(2)]
                per_obj_dim = self._ge_obj_state_dim + self._ge_K_per_obj  # 15 + K
            else:
                per_obj_dim = model_cfg.get("pi_obj_dim", 15)

            pi_emb_dim = model_cfg.get("pi_emb_dim", 512)
            self.pi_encoder = PermInvEncoder(per_obj_dim, pi_emb_dim)
            actor_in_dim = self.robot_state_dim + pi_emb_dim

        # --- Goal projection for additive injection ---
        # Projects pooled goal embedding K_per_obj → first actor hidden dim
        # Used for additive injection: h = act(W1 @ enc + W_g @ g)
        self._goal_proj = None
        self._goal_ln = None
        if self.use_goal_encoder:
            first_hidden = actor_hidden_dim[0]
            self._goal_proj = nn.Linear(self._ge_K_per_obj, first_hidden, bias=False)
            self._goal_ln = nn.LayerNorm(first_hidden)
            nn.init.orthogonal_(self._goal_proj.weight, gain=0.5)

        # --- Actor: explicit layers for additive injection ---
        # We break the actor into layer1, activation, remaining layers
        # so the goal embedding can be injected after layer1.
        if self.use_lstm:
            lstm_hidden = model_cfg.get("lstm_hidden_size", actor_hidden_dim[-1])
            trunk_dims = (
                actor_hidden_dim[:-1] if len(actor_hidden_dim) > 1 else actor_hidden_dim
            )
            trunk_out = actor_hidden_dim[-1]

            if self.use_goal_encoder:
                # Break trunk into: layer1 + activation | rest_of_trunk
                # So we can inject goal after layer1
                self.actor_trunk_layer1 = nn.Linear(actor_in_dim, trunk_dims[0])
                self.actor_trunk_act1 = activation
                if len(trunk_dims) > 1:
                    self.actor_trunk_rest = _build_mlp(
                        trunk_dims[0], trunk_dims[1:], trunk_out, activation
                    )
                else:
                    # trunk_dims has only one entry, so trunk_out == trunk_dims[0]
                    self.actor_trunk_rest = nn.Identity()
                nn.init.orthogonal_(self.actor_trunk_layer1.weight, gain=np.sqrt(2))
            else:
                self.actor_trunk = _build_mlp(
                    actor_in_dim, trunk_dims, trunk_out, activation
                )

            self.actor_lstm = nn.LSTMCell(trunk_out, lstm_hidden)
            self.actor_head = nn.Linear(lstm_hidden, actor_out_dim)
            self.lstm_hidden_size = lstm_hidden
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        else:
            if self.use_goal_encoder:
                # Break actor into: layer1 + activation | rest → head
                self.actor_layer1 = nn.Linear(actor_in_dim, actor_hidden_dim[0])
                self.actor_act1 = activation
                if len(actor_hidden_dim) > 1:
                    self.actor_rest = _build_mlp(
                        actor_hidden_dim[0], actor_hidden_dim[1:], actor_out_dim, activation
                    )
                else:
                    self.actor_rest = nn.Linear(actor_hidden_dim[0], actor_out_dim)
                nn.init.orthogonal_(self.actor_layer1.weight, gain=np.sqrt(2))
            else:
                self.actor = _build_mlp(
                    actor_in_dim, actor_hidden_dim, actor_out_dim, activation
                )

        # --- Critic ---
        # Critic always sees full raw observation (no goal encoder bottleneck)
        critic_in = states_shape[0] if asymmetric else obs_shape[0]
        self.critic = _build_mlp(critic_in, critic_hidden_dim, 1, activation)

        # Log-std only used for Gaussian mode
        if not self.use_multicategorical:
            self.log_std = nn.Parameter(
                np.log(initial_std) * torch.ones(*actions_shape)
            )

        # --- Print architecture ---
        if self.use_goal_encoder:
            print(f"[GoalEncoder] K_per_obj={self._ge_K_per_obj}, "
                  f"objects={self._ge_num_objects}, "
                  f"variant={self.goal_encoder.variant}")
        if self.use_lstm:
            if self.use_goal_encoder:
                print(f"Actor trunk_layer1: {self.actor_trunk_layer1}")
                print(f"Actor trunk_rest: {self.actor_trunk_rest}")
                print(f"Actor goal_proj: {self._goal_proj}")
            else:
                print(f"Actor trunk: {self.actor_trunk}")
            print(f"Actor LSTM: {self.actor_lstm}\nActor head: {self.actor_head}")
        else:
            if self.use_goal_encoder:
                print(f"Actor layer1: {self.actor_layer1}")
                print(f"Actor rest: {self.actor_rest}")
                print(f"Actor goal_proj: {self._goal_proj}")
            else:
                print(self.actor)
        print(self.critic)

        # --- Weight init ---
        if not self.use_lstm and not self.use_goal_encoder:
            actor_scales = [np.sqrt(2)] * len(actor_hidden_dim) + [0.01]
            _init_weights(self.actor, actor_scales)
        if not self.use_lstm and self.use_goal_encoder:
            # Init rest layers
            if len(actor_hidden_dim) > 1 and not isinstance(self.actor_rest, nn.Linear):
                rest_scales = [np.sqrt(2)] * (len(actor_hidden_dim) - 1) + [0.01]
                _init_weights(self.actor_rest, rest_scales)
            elif isinstance(self.actor_rest, nn.Linear):
                nn.init.orthogonal_(self.actor_rest.weight, gain=0.01)
        critic_scales = [np.sqrt(2)] * len(critic_hidden_dim) + [1.0]
        _init_weights(self.critic, critic_scales)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, observations: torch.Tensor, detach_goal_encoder: bool = False) -> tuple:
        """
        Encode observations for the actor.

        When use_goal_encoder is True:
            1. Extract robot state, per-object state, goal poses, distances
            2. Compute per-object goal embeddings g_i via GoalEncoder
            3. Reassemble per-object features as [obj_state(15) | g_i(K)]
            4. Pass through PermInvEncoder
            5. Return (encoded_obs, g_pooled) where g_pooled is for additive injection

        When use_goal_encoder is False:
            Standard path — returns (encoded_obs, None).

        Args:
            observations: (batch, obs_dim) raw observations
            detach_goal_encoder: if True, detach g from computation graph
                                 (used during ABC to prevent encoder distortion)

        Returns:
            (encoded_obs, g_pooled) tuple.
            g_pooled is (batch, K_per_obj) or None.
        """
        if self.use_goal_encoder and self.use_pi_encoder:
            batch = observations.shape[0]
            robot = observations[:, :self._ge_robot_dim]

            # Extract per-object chunks from flat obs
            # Bob obs: [robot(8) | obj1(15)+goal1(7)+dist1(2) | obj2(15)+goal2(7)+dist2(2)]
            obj_section = observations[:, self._ge_robot_dim:]
            obj_chunks = obj_section.view(batch, self._ge_num_objects, self._ge_raw_per_obj)

            # Split each object's chunk
            obj_states = obj_chunks[:, :, :self._ge_obj_state_dim]     # (B, N, 15)
            goal_poses = obj_chunks[:, :, self._ge_obj_state_dim:
                                         self._ge_obj_state_dim + self._ge_goal_dim]  # (B, N, 7)
            # distances are dropped — encoder subsumes their role

            # Extract current object poses (pos+quat) from obj_states for encoder
            current_poses = obj_states[:, :, :7]  # (B, N, 7) — pos(3)+quat(4)

            # Flatten for goal encoder: (batch, num_objects * 7)
            goal_flat = goal_poses.reshape(batch, -1)
            current_flat = current_poses.reshape(batch, -1)

            # Compute per-object goal embeddings
            g_per_obj = self.goal_encoder.encode_per_object(goal_flat, current_flat)
            # (batch, num_objects, K_per_obj)

            # Pooled goal for additive injection
            g_pooled, _ = g_per_obj.max(dim=1)  # (batch, K_per_obj)

            if detach_goal_encoder:
                g_per_obj = g_per_obj.detach()
                g_pooled = g_pooled.detach()

            # Reassemble per-object features: [obj_state(15) | g_i(K_per_obj)]
            new_obj_features = torch.cat([obj_states, g_per_obj], dim=-1)
            # (batch, num_objects, 15 + K_per_obj)

            # Flatten for PI encoder: (batch, num_objects * (15 + K_per_obj))
            new_obj_flat = new_obj_features.reshape(batch, -1)

            # Pass through PermInvEncoder
            encoded = self.pi_encoder(robot, new_obj_flat)
            return encoded, g_pooled

        elif self.use_pi_encoder:
            robot = observations[:, :self.robot_state_dim]
            objs = observations[:, self.robot_state_dim:]
            return self.pi_encoder(robot, objs), None

        return observations, None

    def _actor_forward(self, observations: torch.Tensor, hidden_state=None,
                       detach_goal_encoder: bool = False):
        """
        Returns (raw_output, new_hidden).
        raw_output is (batch, actor_out_dim):
          - Gaussian mode:  action means
          - MC mode:        num_cat_dims * num_bins logits

        Args:
            observations: raw observations
            hidden_state: LSTM hidden state (h, c) or None
            detach_goal_encoder: if True, detach goal encoder from computation graph
        """
        enc, g_pooled = self._encode_obs(observations, detach_goal_encoder)

        if self.use_lstm:
            if self.use_goal_encoder and g_pooled is not None:
                # Additive injection: h = act(LN(W1 @ enc + W_g @ g))
                h1 = self.actor_trunk_layer1(enc)
                h1 = self.actor_trunk_act1(self._goal_ln(h1 + self._goal_proj(g_pooled)))
                feat = self.actor_trunk_rest(h1)
            else:
                feat = self.actor_trunk(enc)

            if hidden_state is None:
                h = torch.zeros(enc.shape[0], self.lstm_hidden_size, device=enc.device)
                c = torch.zeros(enc.shape[0], self.lstm_hidden_size, device=enc.device)
            else:
                h, c = hidden_state
            h, c = self.actor_lstm(feat, (h, c))
            return self.actor_head(h), (h.detach(), c.detach())

        if self.use_goal_encoder and g_pooled is not None:
            # Additive injection: h = act(LN(W1 @ enc + W_g @ g))
            h1 = self.actor_layer1(enc)
            h1 = self.actor_act1(self._goal_ln(h1 + self._goal_proj(g_pooled)))
            return self.actor_rest(h1), None

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

    def evaluate(self, observations, states, actions, detach_goal_encoder=False):
        """
        Evaluate log-probabilities and entropy for the given actions.

        For multi-categorical: `actions` is interpreted as float bin indices
        (cast to int64 internally).
        For Gaussian: `actions` is continuous values.

        Args:
            observations: raw observations
            states: states (for asymmetric critic) or None
            actions: actions to evaluate
            detach_goal_encoder: if True, detach goal encoder gradients
                                 (used during ABC updates)
        """
        raw, _ = self._actor_forward(observations, detach_goal_encoder=detach_goal_encoder)
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
