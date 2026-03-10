import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    """
    Diagonal-Gaussian actor-critic network for continuous action spaces.

    The actor and critic share no weights.  Action noise is learned as a
    global log-standard-deviation parameter (not input-dependent), matching
    the standard PPO setup used for continuous locomotion and manipulation.
    """

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super().__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim  = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation        = get_activation("selu")
        else:
            actor_hidden_dim  = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation        = get_activation(model_cfg["activation"])

        self.actor  = _build_mlp(*obs_shape, actor_hidden_dim, *actions_shape, activation)
        critic_in   = states_shape[0] if asymmetric else obs_shape[0]
        self.critic = _build_mlp(critic_in, critic_hidden_dim, 1, activation)

        print(self.actor)
        print(self.critic)

        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        actor_scales  = [np.sqrt(2)] * len(actor_hidden_dim)  + [0.01]
        critic_scales = [np.sqrt(2)] * len(critic_hidden_dim) + [1.0]
        _init_weights(self.actor,  actor_scales)
        _init_weights(self.critic, critic_scales)

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        """Sample an action from the current policy."""
        actions_mean = self.actor(observations)
        # scale_tril = Cholesky factor: diag(std), NOT diag(std²)
        scale_tril   = torch.diag(self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=scale_tril)

        actions          = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(states if self.asymmetric else observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def act_inference(self, observations):
        """Deterministic action for evaluation (no sampling)."""
        return self.actor(observations)

    def evaluate(self, observations, states, actions):
        """Evaluate log-probabilities and entropy of given actions under the current policy."""
        actions_mean = self.actor(observations)
        # scale_tril = Cholesky factor: diag(std), NOT diag(std²)
        scale_tril   = torch.diag(self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=scale_tril)

        actions_log_prob = distribution.log_prob(actions)
        entropy          = distribution.entropy()
        value            = self.critic(states if self.asymmetric else observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


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
        torch.nn.init.orthogonal_(module.weight, gain=scales[idx])


class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from obs_t to obs_{t+1}.

    Trained on Bob's own rollout data so it learns Bob's kinematics, not
    Alice's.  During BCO relabelling, Alice's object-state trajectory is fed
    through this model to produce kinematically correct Bob actions.
    """

    def __init__(self, obs_shape: tuple, actions_shape: tuple):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_shape[0] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, actions_shape[0]),
        )

    def forward(self, obs_t: torch.Tensor, obs_tplus1: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([obs_t, obs_tplus1], dim=-1))


def get_activation(act_name):

    """Return the nn.Module activation corresponding to act_name."""
    activations = {
        "elu":     nn.ELU(),
        "selu":    nn.SELU(),
        "relu":    nn.ReLU(),
        "crelu":   nn.ReLU(),
        "lrelu":   nn.LeakyReLU(),
        "tanh":    nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if act_name not in activations:
        raise ValueError(f"Unknown activation function: '{act_name}'. Choose from {list(activations)}")
    return activations[act_name]
