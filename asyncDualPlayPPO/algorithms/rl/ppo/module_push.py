"""
Simplified ActorCritic for push-PPO baseline.

No permutation-invariant encoder, no goal encoder — just a flat MLP trunk
with optional LSTM.  Outputs 6D MultiCategorical action logits.

Architecture:
  obs (29D) → Linear(29→512)→ReLU → Linear(512→256)→ReLU
    → LSTM(256→256) → actor_head(256→66) + critic_head(256→128→1)
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from .module import MultiCategorical


class ActorCriticPush(nn.Module):
    def __init__(
        self,
        obs_shape: tuple,
        states_shape: tuple,
        actions_shape: tuple,
        init_noise_std: float,
        model_cfg: dict,
        asymmetric: bool = False,
    ):
        super().__init__()

        self.asymmetric = asymmetric
        self.use_multicategorical = model_cfg.get("use_multicategorical", True)
        self.use_lstm = model_cfg.get("use_lstm", True)

        obs_dim = obs_shape[0]
        self.num_cat_dims = model_cfg.get("num_cat_dims", 6)
        self.num_bins = model_cfg.get("num_bins", 11)
        self.actor_out_dim = self.num_cat_dims * self.num_bins

        pi_hid_sizes = model_cfg.get("pi_hid_sizes", [512, 256, 128])
        vf_hid_sizes = model_cfg.get("vf_hid_sizes", [512, 256, 128])
        activation = model_cfg.get("activation", "relu")
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

        self.lstm_hidden_size = model_cfg.get("lstm_hidden_size", 256)

        # ── Actor trunk ──────────────────────────────────────────────────
        actor_layers = []
        in_dim = obs_dim
        for h in pi_hid_sizes:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(act_fn())
            in_dim = h
        self.actor_trunk = nn.Sequential(*actor_layers)
        self.actor_trunk_dim = in_dim  # 128

        # LSTM
        if self.use_lstm:
            self.lstm = nn.LSTMCell(self.actor_trunk_dim, self.lstm_hidden_size)
            lstm_out = self.lstm_hidden_size
        else:
            self.lstm = None
            lstm_out = self.actor_trunk_dim

        # Actor head
        self.actor_head = nn.Linear(lstm_out, self.actor_out_dim)

        # ── Critic (value head) ──────────────────────────────────────────
        critic_layers = []
        c_in = obs_dim
        for h in vf_hid_sizes:
            critic_layers.append(nn.Linear(c_in, h))
            critic_layers.append(act_fn())
            c_in = h
        critic_layers.append(nn.Linear(c_in, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self._action_space = gym.spaces.Box(
            low=0.0, high=float(self.num_bins - 1),
            shape=(self.num_cat_dims,), dtype=np.float32,
        )

    def forward(self, observations, states=None, masks=None):
        raise NotImplementedError("Use act() or evaluate()")

    def _actor_forward(self, obs, hidden_state, detach_goal_encoder=False):
        x = self.actor_trunk(obs)
        if self.lstm is not None and hidden_state is not None:
            h, c = hidden_state
            h, c = self.lstm(x, (h, c))
            x = h
            new_hidden = (h, c)
        else:
            new_hidden = None
        raw = self.actor_head(x)
        return raw, new_hidden

    def _make_distribution(self, raw):
        logits = raw.view(-1, self.num_cat_dims, self.num_bins)
        return MultiCategorical(logits)

    @torch.no_grad()
    def act(self, observations, states=None, masks=None, deterministic=False):
        raw, new_hidden = self._actor_forward(observations, states)
        dist = self._make_distribution(raw)
        if deterministic:
            actions = torch.argmax(dist.logits, dim=-1).float()
        else:
            actions = dist.sample()
        log_prob = dist.log_prob(actions.long())
        value = self.critic(observations)
        mu = torch.zeros_like(actions)
        sigma = torch.zeros_like(actions)
        return actions, log_prob, value, mu, sigma

    @torch.no_grad()
    def act_with_hidden(self, observations, states, hidden_state):
        raw, new_hidden = self._actor_forward(observations, hidden_state)
        dist = self._make_distribution(raw)
        actions = dist.sample()
        log_prob = dist.log_prob(actions.long())
        value = self.critic(observations)
        mu = torch.zeros_like(actions)
        sigma = torch.zeros_like(actions)
        return actions, log_prob, value, mu, sigma, new_hidden

    def evaluate(self, observations, states, actions):
        raw, _ = self._actor_forward(observations, None)
        dist = self._make_distribution(raw)
        log_prob = dist.log_prob(actions.long())
        entropy = dist.entropy()
        value = self.critic(observations)
        mu = torch.zeros_like(actions)
        sigma = torch.zeros_like(actions)
        return log_prob, entropy, value, mu, sigma

    def get_action_log_prob(self, observations, actions):
        raw, _ = self._actor_forward(observations, None)
        dist = self._make_distribution(raw)
        return dist.log_prob(actions.long())
