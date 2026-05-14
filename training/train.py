# =============================================================
# train.py — Complete Training Script (All Agents)
# C2P-PPO Project | Safe RL with CTL Constraints
# =============================================================

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ppo_base  import PPOAgent
from src.c2p_ppo   import C2PPPOAgent
from src.baselines import (
    PPOLagrangianAgent,
    ShieldedPPOAgent,
    CBFPPOAgent,
    LTLPPOAgent,
    CPOAgent,
)
from src.ablations import (
    C2PPPO_NoLag,
    C2PPPO_NoShield,
    C2PPPO_FixedAlpha,
    C2PPPO_NoRND,
)


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict):
    try:
        import safe_control_gym
        from safe_control_gym.utils.registration import make
        env = make(
            "cartpole",
            **{
                "task_config": {
                    "task": cfg["environment"]["task"],
                    "episode_len_sec": 5,
                    "ctrl_freq": 50,
                }
            }
        )
        print("[ENV] Safe-Control-Gym CartPole loaded ✅")
        return env, "safe_control_gym"
    except Exception as e:
        print(f"[ENV] Failed: {e}")
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        print("[ENV] Gymnasium CartPole loaded ✅")
        return env, "gymnasium"


def get_dims(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] \
        if hasattr(env.action_space, 'shape') \
        and len(env.action_space.shape) > 0 \
        else 1
    return obs_dim, act_dim


def env_step(env, action):
    try:
        result = env.step(action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = result
        next_obs = np.array(next_obs, dtype=np.float32).flatten()
        return next_obs, float(reward), bool(done), info
    except AttributeError:
        # Bug in Safe-Control-Gym seed 3072:
        # 'CartPole' has no attribute 'out_of_bounds'
        # Force episode termination and reset
        obs_reset = env.reset()
        if isinstance(obs_reset, tuple):
            next_obs = obs_reset[0]
        else:
            next_obs = obs_reset
        next_obs = np.array(next_obs, dtype=np.float32).flatten()
        return next_obs, 0.0, True, {}


def env_reset(env, seed=None):
    try:
        result = env.reset(seed=seed)
        obs = result[0] if isinstance(result, tuple) else result
    except TypeError:
        obs = env.reset()
    return np.array(obs, dtype=np.float32).flatten()


# ─────────────────────────────────────────
# Logger
# ─────────────────────────────────────────
class TrainingLogger:
    def __init__(self, cfg: dict, agent_name: str):
        self.cfg        = cfg
        self.agent_name = agent_name
        log_dir = os.path.join(
            cfg["logging"]["log_dir"], agent_name
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        self.episode_rewards:    list = []
        self.episode_violations: list = []
        self.episode_lengths:    list = []
        self.train_stats_log:    list = []

    def log_episode(self, episode, reward, violation, length):
        self.episode_rewards.append(reward)
        self.episode_violations.append(violation)
        self.episode_lengths.append(length)
        self.writer.add_scalar(
            "Episode/Reward",    reward,    episode
        )
        self.writer.add_scalar(
            "Episode/Violation", violation, episode
        )
        self.writer.add_scalar(
            "Episode/Length",    length,    episode
        )

    def log_train(self, iteration, stats):
        self.train_stats_log.append(
            {"iteration": iteration, **stats}
        )
        for k, v in stats.items():
            self.writer.add_scalar(f"Train/{k}", v, iteration)

    def save_csv(self):
        table_dir = self.cfg["logging"]["table_dir"]
        os.makedirs(table_dir, exist_ok=True)
        pd.DataFrame({
            "episode":   range(len(self.episode_rewards)),
            "reward":    self.episode_rewards,
            "violation": self.episode_violations,
            "length":    self.episode_lengths,
        }).to_csv(
            os.path.join(
                table_dir, f"{self.agent_name}_episodes.csv"
            ), index=False
        )
        if self.train_stats_log:
            pd.DataFrame(self.train_stats_log).to_csv(
                os.path.join(
                    table_dir,
                    f"{self.agent_name}_train_stats.csv"
                ), index=False
            )
        print(f"[Logger] CSV saved → {table_dir}")

    def close(self):
        self.writer.close()


# ─────────────────────────────────────────
# Generic Training Loop
# ─────────────────────────────────────────
def train_generic(
    agent,
    agent_name: str,
    cfg:        dict,
    seed:       int = 42,
    is_c2p:     bool = False,
) -> TrainingLogger:
    """
    Generic training loop for all agents.
    Handles both standard and C2P-PPO agents.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, _ = make_env(cfg)
    obs_dim, act_dim = get_dims(env)

    logger     = TrainingLogger(cfg, f"{agent_name}_s{seed}")
    ckpt_dir   = cfg["checkpointing"]["save_dir"]
    save_every = cfg["checkpointing"]["save_every"]
    os.makedirs(ckpt_dir, exist_ok=True)

    n_steps      = cfg["ppo"]["n_steps"]
    max_episodes = 1000
    episode      = 0
    iteration    = 0

    obs      = env_reset(env, seed=seed)
    prev_obs = obs.copy()

    ep_reward     = 0.0
    ep_violations = 0
    ep_length     = 0

    print(f"\n{'='*50}")
    print(f"  {agent_name} | Seed {seed}")
    print(f"{'='*50}\n")

    pbar = tqdm(
        total=max_episodes,
        desc=f"{agent_name} s{seed}"
    )

    while episode < max_episodes:
        for _ in range(n_steps):

            # ── Select action ─────────────
            if is_c2p:
                (action, log_prob, value, obs_t,
                 pi_old_prob, adv_mask) = \
                    agent.select_action(obs, prev_obs)
            else:
                action, log_prob, value, obs_t = \
                    agent.select_action(obs)

            next_obs, r_env, done, _ = env_step(env, action)

            # ── Cost / Reward ──────────────
            if is_c2p:
                r_total, ctl_cost = agent.shape_reward(
                    obs, action, next_obs, r_env
                )
            else:
                ctl_cost = 0.0
                r_total  = r_env
                # Compute CTL cost for ALL agents uniformly
                cart_pos   = float(next_obs[0]) if len(next_obs) > 0 else 0.0
                pole_angle = float(next_obs[2]) if len(next_obs) > 2 else 0.0
                ctl_cost   = float(
                    abs(cart_pos)   >= 2.4
                    or abs(pole_angle) >= 0.2
                    )
                if hasattr(agent, 'add_cost'):
                    agent.add_cost(obs, next_obs)

            # ── Store in buffer ───────────
            action_t = torch.tensor(
                action, dtype=torch.float32
            )
            if is_c2p:
                next_obs_t = torch.tensor(
                    next_obs, dtype=torch.float32
                )
                agent.buffer.add_c2p(
                    obs_t, action_t, log_prob,
                    r_total, value, float(done),
                    next_obs_t, ctl_cost,
                    pi_old_prob, adv_mask,
                )
            else:
                agent.buffer.add(
                    obs_t, action_t, log_prob,
                    r_total, value, float(done),
                )

            ep_reward     += r_env
            ep_violations += int(ctl_cost)
            ep_length     += 1
            prev_obs       = obs.copy()
            obs            = next_obs

            if done:
                viol_rate = ep_violations / max(ep_length, 1)
                logger.log_episode(
                    episode, ep_reward, viol_rate, ep_length
                )

                # Update alpha for C2P-PPO
                if is_c2p and hasattr(agent, 'shield'):
                    agent.shield.update_alpha(viol_rate)
                elif is_c2p and hasattr(agent, 'update_alpha'):
                    agent.update_alpha(viol_rate)

                obs      = env_reset(env)
                prev_obs = obs.copy()

                ep_reward     = 0.0
                ep_violations = 0
                ep_length     = 0
                episode      += 1
                pbar.update(1)

                if episode >= max_episodes:
                    break

        # ── Update ────────────────────────
        if len(agent.buffer.obs) > 0:
            stats = agent.update(obs)
            logger.log_train(iteration, stats)
            iteration += 1

            if iteration % cfg["logging"]["log_every"] == 0:
                recent_r = np.mean(
                    logger.episode_rewards[-20:]
                ) if logger.episode_rewards else 0.0
                recent_v = np.mean(
                    logger.episode_violations[-20:]
                ) if logger.episode_violations else 0.0

                msg = (
                    f"[{agent_name} s{seed} "
                    f"Iter {iteration:4d}] "
                    f"R: {recent_r:7.3f} | "
                    f"V: {recent_v:.3f}"
                )
                if "lambda_t" in stats:
                    msg += f" | λ: {stats['lambda_t']:.4f}"
                if "alpha_t" in stats:
                    msg += f" | α: {stats['alpha_t']:.4f}"
                tqdm.write(msg)

            if iteration % save_every == 0:
                agent.save(os.path.join(
                    ckpt_dir,
                    f"{agent_name}_s{seed}"
                    f"_iter{iteration:04d}.pt"
                ))

    pbar.close()
    agent.save(os.path.join(
        ckpt_dir, f"{agent_name}_s{seed}_final.pt"
    ))
    logger.save_csv()
    logger.close()

    print(f"\n[{agent_name} s{seed}] Complete ✅")
    return logger


# ─────────────────────────────────────────
# Agent Factory
# ─────────────────────────────────────────
def make_agent(agent_name: str, cfg: dict, obs_dim: int, act_dim: int):
    agents = {
        "ppo_baseline":  (PPOAgent,            False),
        "ppo_lagrangian":(PPOLagrangianAgent,   False),
        "shielded_ppo":  (ShieldedPPOAgent,     False),
        "cbf_ppo":       (CBFPPOAgent,          False),
        "ltl_ppo":        (LTLPPOAgent,          False),
        "cpo":            (CPOAgent,             False),
        "c2p_ppo":       (C2PPPOAgent,          True),
        "c2p_nolag":         (C2PPPO_NoLag,         True),
        "c2p_noshield":      (C2PPPO_NoShield,      True),
        "c2p_fixedalpha":    (C2PPPO_FixedAlpha,    True),
        "c2p_nornd":         (C2PPPO_NoRND,         True),
    }
    if agent_name not in agents:
        raise ValueError(f"Unknown agent: {agent_name}")
    AgentClass, is_c2p = agents[agent_name]
    return AgentClass(cfg, obs_dim, act_dim), is_c2p


# ─────────────────────────────────────────
# Run Single Experiment
# ─────────────────────────────────────────
def run_experiment(
    agent_name: str,
    cfg:        dict,
    seed:       int = 42,
):
    env, _ = make_env(cfg)
    obs_dim, act_dim = get_dims(env)
    env.close() if hasattr(env, 'close') else None

    # Re-create env inside train_generic
    agent, is_c2p = make_agent(
        agent_name, cfg, obs_dim, act_dim
    )

    return train_generic(
        agent, agent_name, cfg, seed, is_c2p
    )


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
        choices=[
            "c2p_ppo",
            "ppo_baseline",
            "ppo_lagrangian",
            "shielded_ppo",
            "cbf_ppo",
            "ltl_ppo",
            "cpo",
            "c2p_nolag",
            "c2p_noshield",
            "c2p_fixedalpha",
            "c2p_nornd",
            "all",
            "ablations",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.agent == "all":
        agents = [
            "ppo_baseline",
            "ppo_lagrangian",
            "shielded_ppo",
            "cbf_ppo",
            "ltl_ppo",
            "cpo",
            "c2p_ppo",
        ]
    elif args.agent == "ablations":
        agents = [
            "c2p_ppo",
            "c2p_nolag",
            "c2p_noshield",
            "c2p_fixedalpha",
            "c2p_nornd",
        ]
        for agent_name in agents:
            print(f"\n{'#'*50}")
            print(f"# Training: {agent_name}")
            print(f"{'#'*50}")
            run_experiment(agent_name, cfg, seed=args.seed)
    else:
        run_experiment(args.agent, cfg, seed=args.seed)