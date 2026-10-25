"""Neural-network components shared by all algorithms.

All methods use identical architectures (two hidden layers, tanh) so that
differences in performance are attributable to the learning rule and not to
model capacity.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim: int, out_dim: int, hidden: Sequence[int] = (64, 64),
        act: type[nn.Module] = nn.Tanh, out_gain: float = 1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy pi_theta(a|s) with state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), log_std_init: float = -0.5):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, hidden, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def dist(self, obs: torch.Tensor) -> Normal:
        mu = self.mu(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 1.0)).expand_as(mu)
        return Normal(mu, std)

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.dist(obs).log_prob(act).sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden=(64, 64)):
        super().__init__()
        self.v = mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class SafetyCritic(nn.Module):
    """Level-3 surrogate Safe_NN(s,a;psi) = sigma(f_psi(s,a)).

    Estimates the probability that no CTL atomic predicate is violated within
    the next k steps after executing action a in state s (Section 5).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128)):
        super().__init__()
        self.f = mlp(obs_dim + act_dim, 1, hidden, act=nn.ReLU)

    def logits(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.f(torch.cat([obs, act], -1)).squeeze(-1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(obs, act))


class RND(nn.Module):
    """Random Network Distillation (Burda et al., 2019) over state-action pairs.

    Explore(s,a) = ||f(s,a) - f_target(s,a)||^2, mapped to [0,1) by
    1 - exp(-e / e_bar) where e_bar is a running mean of the prediction error.
    """

    def __init__(self, obs_dim: int, act_dim: int, feat: int = 64, hidden=(128, 128)):
        super().__init__()
        self.target = mlp(obs_dim + act_dim, feat, hidden, act=nn.ReLU)
        self.pred = mlp(obs_dim + act_dim, feat, hidden, act=nn.ReLU)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.register_buffer("err_mean", torch.tensor(1.0))
        self.register_buffer("count", torch.tensor(0.0))

    def error(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], -1)
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)

    @torch.no_grad()
    def bonus(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        e = self.error(obs, act)
        return 1.0 - torch.exp(-e / (self.err_mean + 1e-8))

    @torch.no_grad()
    def update_stats(self, err: torch.Tensor) -> None:
        n = float(err.numel())
        tot = self.count + n
        self.err_mean.copy_((self.err_mean * self.count + err.sum()) / tot)
        self.count.copy_(torch.clamp(tot, max=1e6))


class RunningMeanStd:
    """Running observation normaliser (Welford)."""

    def __init__(self, shape, eps: float = 1e-4):
        import numpy as np
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x) -> None:
        import numpy as np
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.mean.shape[0])
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x, clip: float = 10.0):
        import numpy as np
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}
