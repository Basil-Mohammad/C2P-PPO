# =============================================================
# ppo_base.py — PPO with Continuous + Discrete Support
# C2P-PPO Project | Safe RL with CTL Constraints
# =============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
from typing import Tuple, Dict, List


# ─────────────────────────────────────────
# Continuous Actor-Critic Network
# ─────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim:     int,
        act_dim:     int,
        hidden:      int  = 64,
        continuous:  bool = True,
        act_low:     float = -10.0,
        act_high:    float =  10.0,
    ):
        super().__init__()
        self.continuous = continuous
        self.act_low    = act_low
        self.act_high   = act_high
        self.act_scale  = (act_high - act_low) / 2.0
        self.act_bias   = (act_high + act_low) / 2.0

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        if continuous:
            # Gaussian policy
            self.actor_mean    = nn.Linear(hidden, act_dim)
            self.actor_log_std = nn.Parameter(
                torch.zeros(act_dim)
            )
        else:
            self.actor = nn.Linear(hidden, act_dim)

        # Critic
        self.critic = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        if self.continuous:
            nn.init.orthogonal_(
                self.actor_mean.weight, gain=0.01
            )
        else:
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        x     = self.backbone(obs)
        value = self.critic(x).squeeze(-1)
        if self.continuous:
            mean = torch.tanh(self.actor_mean(x))
            mean = mean * self.act_scale + self.act_bias
            return mean, value
        else:
            logits = self.actor(x)
            return logits, value

    def get_action(self, obs: torch.Tensor):
        if self.continuous:
            mean, value = self.forward(obs)
            std  = self.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            action   = dist.sample()
            action   = torch.clamp(
                action, self.act_low, self.act_high
            )
            log_prob = dist.log_prob(action).sum(-1)
            entropy  = dist.entropy().sum(-1)
        else:
            logits, value = self.forward(obs)
            dist     = Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ):
        if self.continuous:
            mean, value = self.forward(obs)
            std  = self.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(-1)
            entropy  = dist.entropy().sum(-1)
        else:
            logits, value = self.forward(obs)
            dist     = Categorical(logits=logits)
            log_prob = dist.log_prob(actions)
            entropy  = dist.entropy()
        return log_prob, entropy, value

    def get_action_probs(
        self, obs: torch.Tensor, n_samples: int = 10
    ) -> torch.Tensor:
        """
        For continuous policy: estimate action probabilities
        via sampling (used by shield for safety weighting).
        Returns normalised weights of shape (n_samples,).
        """
        if self.continuous:
            mean, _ = self.forward(obs)
            std  = self.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            samples  = dist.sample((n_samples,))
            log_prob = dist.log_prob(samples).sum(-1)
            probs    = torch.softmax(log_prob, dim=0)
            return probs, samples
        else:
            logits, _ = self.forward(obs)
            return torch.softmax(logits, dim=-1), None


# ─────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.obs:       List = []
        self.actions:   List = []
        self.log_probs: List = []
        self.rewards:   List = []
        self.values:    List = []
        self.dones:     List = []

    def clear(self):
        self.__init__()

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.rewards
        values  = self.values + [last_value.item()]
        dones   = self.dones

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (rewards[t]
                     + gamma * values[t+1] * (1 - dones[t])
                     - values[t])
            gae   = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(
            advantages, dtype=torch.float32
        )
        returns = advantages + torch.tensor(
            self.values, dtype=torch.float32
        )
        return advantages, returns

    def get_tensors(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
        )


# ─────────────────────────────────────────
# Standard PPO Agent (Continuous)
# ─────────────────────────────────────────
class PPOAgent:
    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        self.cfg    = cfg
        self.device = torch.device(
            cfg["experiment"]["device"]
        )

        p = cfg["ppo"]
        self.lr            = p["lr"]
        self.gamma         = p["gamma"]
        self.gae_lambda    = p["gae_lambda"]
        self.clip_epsilon  = p["clip_epsilon"]
        self.epochs        = p["epochs"]
        self.batch_size    = p["batch_size"]
        self.value_coef    = p["value_coef"]
        self.entropy_coef  = p["entropy_coef"]
        self.max_grad_norm = p["max_grad_norm"]

        # Continuous actor-critic
        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.buffer      = RolloutBuffer()
        self.train_stats: Dict = {}

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, entropy, value = \
            self.ac.get_action(obs_t)
        return (
            action.cpu().numpy(),
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
        )

    def update(self, last_obs: np.ndarray) -> Dict:
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        obs_t, actions_t, old_lp_t = self.buffer.get_tensors()
        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        total_loss = total_pg = total_vf = 0.0
        total_ent  = 0.0
        n_updates  = 0
        dataset_size = obs_t.shape[0]

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start: start + self.batch_size]

                log_probs, entropy, values = \
                    self.ac.evaluate_actions(
                        obs_t[idx], actions_t[idx]
                    )

                ratio = torch.exp(log_probs - old_lp_t[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * advantages[idx]

                pg_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss = -entropy.mean()
                loss     = (pg_loss
                            + self.value_coef  * vf_loss
                            + self.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += entropy.mean().item()
                n_updates  += 1

        self.buffer.clear()
        self.train_stats = {
            "loss":    total_loss / n_updates,
            "pg_loss": total_pg   / n_updates,
            "vf_loss": total_vf   / n_updates,
            "entropy": total_ent  / n_updates,
        }
        return self.train_stats

    def save(self, path: str):
        torch.save({
            "model_state":     self.ac.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)
        print(f"[PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[PPO] Checkpoint loaded ← {path}")