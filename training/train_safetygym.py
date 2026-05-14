# =============================================================
# train_safetygym.py — Training on SafetyPointGoal1-v0
# C2P-PPO Project | Safe RL with CTL Constraints
# Uses separate safegym conda environment
# =============================================================

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

from src.ppo_base  import ActorCritic, RolloutBuffer
from src.lagrangian import LagrangianMultiplier
from src.c2p_ppo   import C2PRolloutBuffer
from envs.safety_gym_wrapper import (
    SafetyGymnasiumWrapper,
    PointGoalSafetyMapper,
)


# ─────────────────────────────────────────
# Load Config
# ─────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────
# PointGoal-specific Adaptive Shield
# ─────────────────────────────────────────
class PointGoalShield:
    """
    Adaptive shield for SafetyPointGoal1.
    Uses lidar margin to scale actions.
    """
    def __init__(self, cfg: dict):
        s = cfg["shielding"]
        self.alpha_t     = s["alpha_0"]
        self.beta        = s["beta"]
        self.c_threshold = s["c_threshold"]
        self.window_size = s["window_size"]

        from collections import deque
        self.violation_buffer = deque(maxlen=self.window_size)
        for _ in range(self.window_size):
            self.violation_buffer.append(0.0)
        self.alpha_history = []

    def update_alpha(self, viol_rate: float):
        self.violation_buffer.append(viol_rate)
        c = float(np.mean(self.violation_buffer))
        x = self.beta * (c - self.c_threshold)
        self.alpha_t = float(
            torch.sigmoid(torch.tensor(x)).item()
        )
        self.alpha_history.append(self.alpha_t)
        return self.alpha_t

    def scale_action(
        self,
        action:        np.ndarray,
        safety_margin: float,
    ) -> np.ndarray:
        """
        Scale action toward zero when unsafe.
        For PointGoal: zero action = stop moving.
        """
        if safety_margin > 0.3:
            return np.clip(action, -1.0, 1.0)

        blend    = self.alpha_t * (1.0 - safety_margin / 0.3)
        shielded = (1.0 - blend) * action
        return np.clip(shielded, -1.0, 1.0)


# ─────────────────────────────────────────
# C2P-PPO Agent for PointGoal
# ─────────────────────────────────────────
class C2PPPOPointGoal:
    """C2P-PPO adapted for SafetyPointGoal1-v0."""

    def __init__(
        self,
        cfg:     dict,
        obs_dim: int,
        act_dim: int,
    ):
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

        # Larger network for obs_dim=60
        self.ac = ActorCritic(
            obs_dim, act_dim,
            hidden=128,
            continuous=True,
            act_low=-1.0,
            act_high=1.0,
        ).to(self.device)

        self.optimizer  = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.lagrangian = LagrangianMultiplier(cfg)
        self.shield     = PointGoalShield(cfg)
        self.mapper     = PointGoalSafetyMapper(cfg)
        self.buffer     = C2PRolloutBuffer()

        self.iteration   = 0
        self.train_stats = {}

    @torch.no_grad()
    def select_action(
        self,
        obs:      np.ndarray,
        env_cost: float = 0.0,
    ):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)

        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()

        safety_margin = self.mapper.safety_margin(
            obs, env_cost
        )
        shielded = self.shield.scale_action(
            action_np, safety_margin
        )

        return (
            shielded,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
            log_prob.item(),
            1.0,
        )

    def shape_reward(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        r_env:    float,
        env_cost: float,
    ):
        return self.mapper.shape_reward(
            obs, next_obs, r_env, env_cost
        )

    def update(self, last_obs: np.ndarray) -> dict:
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae_c2p(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        (obs_t, actions_t, old_lp_t,
         next_obs_t, ctl_cost_t) = \
            self.buffer.get_c2p_tensors()

        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)
        ctl_cost_t = ctl_cost_t.to(self.device)

        lambda_t   = self.lagrangian.lambda_t

        total_loss = total_pg = total_vf = 0.0
        total_ent  = total_safety = 0.0
        n_updates  = 0
        dataset_size = obs_t.shape[0]

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(
                0, dataset_size, self.batch_size
            ):
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

                pg_loss     = -torch.min(surr1, surr2).mean()
                vf_loss     = torch.nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss    = -entropy.mean()
                safety_loss = lambda_t * ctl_cost_t[idx].mean()

                loss = (pg_loss
                        + self.value_coef   * vf_loss
                        + self.entropy_coef * ent_loss
                        + safety_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss   += loss.item()
                total_pg     += pg_loss.item()
                total_vf     += vf_loss.item()
                total_ent    += entropy.mean().item()
                total_safety += safety_loss.item()
                n_updates    += 1

        mean_ctl = ctl_cost_t.mean().item()
        self.lagrangian.update(mean_ctl)
        self.buffer.clear()
        self.iteration += 1

        self.train_stats = {
            "loss":        total_loss   / n_updates,
            "pg_loss":     total_pg     / n_updates,
            "vf_loss":     total_vf     / n_updates,
            "entropy":     total_ent    / n_updates,
            "safety_loss": total_safety / n_updates,
            "lambda_t":    self.lagrangian.lambda_t,
            "alpha_t":     self.shield.alpha_t,
            "ctl_cost":    mean_ctl,
        }
        return self.train_stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "iteration":   self.iteration,
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "lambda_t":    self.lagrangian.lambda_t,
            "alpha_t":     self.shield.alpha_t,
        }, path)
        print(f"[C2P-PointGoal] Saved → {path}")


# ─────────────────────────────────────────
# PPO Baseline for PointGoal
# ─────────────────────────────────────────
class PPOPointGoal:
    """Standard PPO for SafetyPointGoal1-v0."""

    def __init__(
        self, cfg: dict, obs_dim: int, act_dim: int
    ):
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

        self.ac = ActorCritic(
            obs_dim, act_dim,
            hidden=128,
            continuous=True,
            act_low=-1.0,
            act_high=1.0,
        ).to(self.device)

        self.optimizer   = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.buffer      = RolloutBuffer()
        self.train_stats = {}

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        return (
            action.cpu().numpy(),
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
        )

    def update(self, last_obs: np.ndarray) -> dict:
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

        obs_t, actions_t, old_lp_t = \
            self.buffer.get_tensors()
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
            for start in range(
                0, dataset_size, self.batch_size
            ):
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
                vf_loss  = torch.nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss = -entropy.mean()
                loss     = (pg_loss
                            + self.value_coef   * vf_loss
                            + self.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
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
            "ctl_cost": 0.0,
        }
        return self.train_stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, path)
        print(f"[PPO-PointGoal] Saved → {path}")


# ─────────────────────────────────────────
# Logger
# ─────────────────────────────────────────
class PointGoalLogger:
    def __init__(self, cfg: dict, name: str):
        self.name = name
        log_dir   = os.path.join(
            cfg["logging"]["log_dir"],
            f"pointgoal_{name}"
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer     = SummaryWriter(log_dir)
        self.rewards    = []
        self.violations = []
        self.lengths    = []
        self.stats_log  = []

    def log_episode(
        self, ep: int, r: float, v: float, l: int
    ):
        self.rewards.append(r)
        self.violations.append(v)
        self.lengths.append(l)
        self.writer.add_scalar("Episode/Reward",    r, ep)
        self.writer.add_scalar("Episode/Violation", v, ep)
        self.writer.add_scalar("Episode/Length",    l, ep)

    def log_train(self, it: int, stats: dict):
        self.stats_log.append({"iteration": it, **stats})
        for k, v in stats.items():
            self.writer.add_scalar(f"Train/{k}", v, it)

    def save_csv(self, table_dir: str):
        os.makedirs(table_dir, exist_ok=True)
        pd.DataFrame({
            "episode":   range(len(self.rewards)),
            "reward":    self.rewards,
            "violation": self.violations,
            "length":    self.lengths,
        }).to_csv(
            os.path.join(
                table_dir,
                f"pointgoal_{self.name}_episodes.csv"
            ), index=False
        )
        if self.stats_log:
            pd.DataFrame(self.stats_log).to_csv(
                os.path.join(
                    table_dir,
                    f"pointgoal_{self.name}_train_stats.csv"
                ), index=False
            )
        print(f"[Logger] Saved → {table_dir}")

    def close(self):
        self.writer.close()


# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────
def train_pointgoal(
    agent_name: str,
    cfg:        dict,
    seed:       int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env    = SafetyGymnasiumWrapper(
        max_episode_steps=500
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"\n{'='*50}")
    print(f"  PointGoal | {agent_name} | Seed {seed}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"{'='*50}\n")

    # Create agent
    if agent_name == "c2p_ppo":
        agent  = C2PPPOPointGoal(cfg, obs_dim, act_dim)
        is_c2p = True
    else:
        # PPO baseline and PPO-Lagrangian use same arch
        # PPO-Lagrangian adds Lagrangian penalty
        agent  = PPOPointGoal(cfg, obs_dim, act_dim)
        is_c2p = False
        if agent_name == "ppo_lagrangian":
            from src.lagrangian import LagrangianMultiplier
            agent.lagrangian = LagrangianMultiplier(cfg)
            agent.is_lagrangian = True
        else:
            agent.is_lagrangian = False

    logger     = PointGoalLogger(cfg, f"{agent_name}_s{seed}")
    ckpt_dir   = cfg["checkpointing"]["save_dir"]
    save_every = cfg["checkpointing"]["save_every"]
    table_dir  = cfg["logging"]["table_dir"]

    n_steps      = cfg["ppo"]["n_steps"]
    max_episodes = 500
    episode      = 0
    iteration    = 0

    obs, _        = env.reset(seed=seed)
    prev_cost     = 0.0
    ep_reward     = 0.0
    ep_violations = 0
    ep_length     = 0

    pbar = tqdm(
        total=max_episodes,
        desc=f"PointGoal {agent_name} s{seed}"
    )

    while episode < max_episodes:
        for _ in range(n_steps):

            if is_c2p:
                (action, log_prob, value,
                 obs_t, pi_old, adv_mask) = \
                    agent.select_action(obs, prev_cost)
            else:
                action, log_prob, value, obs_t = \
                    agent.select_action(obs)

            next_obs, r_env, env_cost, done, info = \
                env.step(action)

            if is_c2p:
                r_total, ctl_cost = agent.shape_reward(
                    obs, next_obs, r_env, env_cost
                )
                next_obs_t = torch.tensor(
                    next_obs, dtype=torch.float32
                )
                agent.buffer.add_c2p(
                    obs_t,
                    torch.tensor(
                        action, dtype=torch.float32
                    ),
                    log_prob,
                    r_total,
                    value,
                    float(done),
                    next_obs_t,
                    ctl_cost,
                    pi_old,
                    adv_mask,
                )
            else:
                ctl_cost = float(env_cost > 0.5)
                agent.buffer.add(
                    obs_t,
                    torch.tensor(
                        action, dtype=torch.float32
                    ),
                    log_prob,
                    float(r_env),
                    value,
                    float(done),
                )

            ep_reward     += float(r_env)
            ep_violations += int(ctl_cost)
            ep_length     += 1
            prev_cost      = env_cost
            obs            = next_obs

            if done:
                viol_rate = ep_violations / max(ep_length, 1)
                logger.log_episode(
                    episode, ep_reward, viol_rate, ep_length
                )
                if is_c2p:
                    agent.shield.update_alpha(viol_rate)

                obs, _        = env.reset()
                prev_cost     = 0.0
                ep_reward     = 0.0
                ep_violations = 0
                ep_length     = 0
                episode      += 1
                pbar.update(1)

                if episode >= max_episodes:
                    break

        if len(agent.buffer.obs) > 0:
            stats = agent.update(obs)
            logger.log_train(iteration, stats)
            iteration += 1

            if iteration % cfg["logging"]["log_every"] == 0:
                recent_r = np.mean(
                    logger.rewards[-20:]
                ) if logger.rewards else 0.0
                recent_v = np.mean(
                    logger.violations[-20:]
                ) if logger.violations else 0.0
                tqdm.write(
                    f"[{agent_name} s{seed} "
                    f"Iter {iteration:4d}] "
                    f"R: {recent_r:7.3f} | "
                    f"V: {recent_v:.3f}"
                    + (f" | λ: {stats['lambda_t']:.4f}"
                       f" | α: {stats['alpha_t']:.4f}"
                       if is_c2p else "")
                )

            if iteration % save_every == 0:
                agent.save(os.path.join(
                    ckpt_dir,
                    f"pointgoal_{agent_name}"
                    f"_s{seed}_iter{iteration:04d}.pt"
                ))

    pbar.close()
    agent.save(os.path.join(
        ckpt_dir,
        f"pointgoal_{agent_name}_s{seed}_final.pt"
    ))
    logger.save_csv(table_dir)
    logger.close()
    env.close()

    print(f"\n[PointGoal {agent_name} s{seed}] "
          f"Complete ✅")
    return logger


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/cartpole_config.yaml",
    )
    parser.add_argument(
        "--agent",
        default="c2p_ppo",
        choices=["c2p_ppo", "ppo_baseline", "ppo_lagrangian", "both"],
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.agent == "both":
        train_pointgoal("ppo_baseline",   cfg, args.seed)
        train_pointgoal("ppo_lagrangian", cfg, args.seed)
        train_pointgoal("c2p_ppo",        cfg, args.seed)
    else:
        train_pointgoal(args.agent, cfg, args.seed)