"""Minimal PPO pieces: GAE over a window with a bootstrap at the boundary."""

from __future__ import annotations

import torch


def gae(rewards: torch.Tensor, values: torch.Tensor, resets: torch.Tensor,
        bootstrap: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
    """rewards/resets (T, B); values (T, B); bootstrap (B,) = V(s_{T}).

    resets[t] = True when the episode ended AT step t (the env auto-reset
    afterwards), so no value bootstraps across the boundary.
    Returns (advantages (T,B), returns (T,B) = adv + values).
    """
    t_len, _ = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(t_len)):
        alive = (~resets[t]).float()
        delta = rewards[t] + gamma * next_value * alive - values[t]
        last = delta + gamma * lam * alive * last
        adv[t] = last
        next_value = values[t]
    return adv, adv + values
