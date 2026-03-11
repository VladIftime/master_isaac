import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

def build_mlp(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)

class AC(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = build_mlp(52, 14)
        self.critic = build_mlp(52, 1)
        self.log_std = nn.Parameter(torch.zeros(14))
        self.asymmetric = False
    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations)
        scale_tril   = torch.diag(self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=scale_tril)

        actions_log_prob = distribution.log_prob(actions)
        entropy          = distribution.entropy()
        value            = self.critic(states if self.asymmetric else observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

ac = AC()
_o = torch.zeros(25, 52)
_a = torch.zeros(25, 14)
r1, r2, r3, r4, r5 = ac.evaluate(_o, None, _a)
print("r1 shape:", r1.shape, "numel:", r1.numel())
print("r2 shape:", r2.shape, "numel:", r2.numel())
print("r3 shape:", r3.shape, "numel:", r3.numel())
print("r4 shape:", r4.shape, "numel:", r4.numel())
print("r5 shape:", r5.shape, "numel:", r5.numel())

active_mask = torch.ones(25, 1)
test = r1.view(-1, 1) * active_mask
print(test.shape, test.numel())
