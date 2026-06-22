"""
skrl-compatible models for Push-SAC with LSTM.

Architecture matches the Push-PPO baseline (module_push.py) for fair comparison:
  Actor:  obs → 512 → 256 → 128 → LSTM(128→256) → mu(4) + log_std(4)
  Critic: (obs ⊕ action) → 512 → 256 → 128 → Q(1)
"""

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class PushPolicyRNN(GaussianMixin, Model):
    """
    SAC actor with LSTM for push task.

    Continuous Gaussian policy with tanh squashing (handled by skrl's GaussianMixin).
    LSTM provides temporal memory across pushes within an episode.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        reduction="sum",
        num_envs=1,
        num_layers=1,
        hidden_size=256,
        sequence_length=8,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction,
        )

        self._num_envs = num_envs
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._sequence_length = sequence_length

        obs_dim = self.num_observations

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mean_head = nn.Linear(hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self._sequence_length,
                "sizes": [
                    (self._num_layers, self._num_envs, self._hidden_size),
                    (self._num_layers, self._num_envs, self._hidden_size),
                ],
            }
        }

    def compute(self, inputs, role=""):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        rnn_inputs = inputs.get("rnn", None)

        trunk_out = self.trunk(states)

        if rnn_inputs is not None and len(rnn_inputs) == 2:
            hidden = rnn_inputs[0]
            cell = rnn_inputs[1]
        else:
            hidden = torch.zeros(
                self._num_layers, states.shape[0], self._hidden_size,
                device=self.device,
            )
            cell = torch.zeros(
                self._num_layers, states.shape[0], self._hidden_size,
                device=self.device,
            )

        if terminated is not None and torch.any(terminated):
            term_mask = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
            hidden[:, term_mask.bool()] = 0.0
            cell[:, term_mask.bool()] = 0.0

        sequence_length = self._sequence_length
        if trunk_out.dim() == 2:
            batch_size = trunk_out.shape[0]
            if batch_size > self._num_envs and batch_size % sequence_length == 0:
                rnn_in = trunk_out.view(-1, sequence_length, 128)
                if hidden.shape[1] != rnn_in.shape[0]:
                    hidden = torch.zeros(
                        self._num_layers, rnn_in.shape[0], self._hidden_size,
                        device=self.device,
                    )
                    cell = torch.zeros(
                        self._num_layers, rnn_in.shape[0], self._hidden_size,
                        device=self.device,
                    )
                rnn_out, (new_hidden, new_cell) = self.lstm(rnn_in, (hidden, cell))
                rnn_out = rnn_out.reshape(-1, self._hidden_size)
            else:
                rnn_in = trunk_out.unsqueeze(1)
                if hidden.shape[1] != batch_size:
                    hidden = torch.zeros(
                        self._num_layers, batch_size, self._hidden_size,
                        device=self.device,
                    )
                    cell = torch.zeros(
                        self._num_layers, batch_size, self._hidden_size,
                        device=self.device,
                    )
                rnn_out, (new_hidden, new_cell) = self.lstm(rnn_in, (hidden, cell))
                rnn_out = rnn_out.squeeze(1)
        else:
            rnn_out, (new_hidden, new_cell) = self.lstm(trunk_out, (hidden, cell))
            if rnn_out.dim() == 3:
                rnn_out = rnn_out.reshape(-1, self._hidden_size)

        mean_actions = self.mean_head(rnn_out)

        return mean_actions, self.log_std_parameter, {"rnn": [new_hidden.detach(), new_cell.detach()]}


class PushCritic(DeterministicMixin, Model):
    """
    SAC Q-network for push task (no LSTM — feedforward).

    Input: concatenation of observation + action.
    Output: scalar Q-value.
    """

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        input_dim = self.num_observations + self.num_actions

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        final = self.net[-1]
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)

    def compute(self, inputs, role=""):
        states = inputs["states"]
        actions = inputs["taken_actions"]
        x = torch.cat([states, actions], dim=-1)
        return self.net(x), {}
