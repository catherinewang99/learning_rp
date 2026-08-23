"""ActorCritic: twin-friendly conv encoder + Gaussian policy + value head.

The conv stack mirrors the vgg family used in the AV track (3x3 convs,
GroupNorm, canonical pool spacing, GAP) with layer names conv1..convN so
ProbedModel's Conv2d probing and the 1:1 layer mapping work unchanged. The
value head shares the encoder (project decision): the encoder's update — and
therefore V(X|e) — includes both the policy and value gradients.

Action distribution: TANH-SQUASHED diagonal Gaussian over [forward, turn]
(SquashedNormal, as in the sensory-hierarchy repo). The squash is load-bearing,
not cosmetic: with a clipped unsquashed Gaussian, a mean drifting past +-1 is
an ABSORBING state (every sample executes the same railed action — observed as
circling — and the gradient carries no signal to pull the mean back). A
squashed mean lives in (-1,1) by construction. Entropy uses the base-Gaussian
entropy as the standard proxy (the squashed density has no closed form).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

VGG6_WIDTHS = (32, 64, 128, 128, 256, 256)


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: tuple[int, ...] = VGG6_WIDTHS,
        pool_after: tuple[int, ...] = (1, 2, 4, 6),
        trunk_dim: int = 256,
        action_dim: int = 2,
        groups: int = 8,
    ):
        super().__init__()
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        c = in_channels
        for i, w in enumerate(widths, start=1):
            layers[f"conv{i}"] = nn.Conv2d(c, w, kernel_size=3, padding=1)
            layers[f"norm{i}"] = nn.GroupNorm(min(groups, w), w)
            layers[f"act{i}"] = nn.ReLU()
            if i in pool_after:
                layers[f"pool{i}"] = nn.MaxPool2d(2)
            c = w
        layers["gap"] = nn.AdaptiveAvgPool2d(1)
        layers["flatten"] = nn.Flatten()
        self.encoder = nn.Sequential(layers)
        self.trunk = nn.Sequential(nn.Linear(c, trunk_dim), nn.Tanh())
        self.actor_mean = nn.Linear(trunk_dim, action_dim)
        self.value_head = nn.Linear(trunk_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

    def forward(self, x: torch.Tensor):
        z = self.trunk(self.encoder(x))
        mean = self.actor_mean(z)          # PRE-tanh mean; executed = tanh(sample)
        value = self.value_head(z).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value


_ATANH_EPS = 1e-6


def squashed_logp_entropy(mean, log_std, action):
    """log-prob of an EXECUTED action a = tanh(u) under the squashed Gaussian,
    plus the base-Gaussian entropy proxy. One formula for collection and
    replay (u recovered via atanh) so PPO ratios are exactly consistent."""
    a = action.clamp(-1 + _ATANH_EPS, 1 - _ATANH_EPS)
    u = torch.atanh(a)
    dist = torch.distributions.Normal(mean, log_std.exp())
    logp = dist.log_prob(u).sum(-1) - torch.log1p(-a.pow(2) + _ATANH_EPS).sum(-1)
    return logp, dist.entropy().sum(-1)


def squashed_sample(mean, log_std):
    """(action in (-1,1), logp) — sampling path for rollouts."""
    dist = torch.distributions.Normal(mean, log_std.exp())
    action = torch.tanh(dist.sample())
    logp, _ = squashed_logp_entropy(mean, log_std, action)
    return action, logp
