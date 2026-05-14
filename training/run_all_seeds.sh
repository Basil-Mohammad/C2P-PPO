#!/bin/bash
# ─────────────────────────────────────────
# run_all_seeds.sh — Full Experiment Suite
# C2P-PPO | 7 Agents × 5 Seeds = 35 Runs
# ─────────────────────────────────────────

echo "========================================"
echo "  C2P-PPO Full Experiment Suite"
echo "  7 Agents × 5 Seeds = 35 Experiments"
echo "========================================"

SEEDS=(42 123 456 789 1024 2048 3072 4096 5120 6144)
AGENTS=(
    ppo_baseline
    ppo_lagrangian
    shielded_ppo
    cbf_ppo
    ltl_ppo
    cpo
    c2p_ppo
)

TOTAL=$((${#AGENTS[@]} * ${#SEEDS[@]}))
COUNT=0

for agent in "${AGENTS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "  Agent: $agent"
    echo "----------------------------------------"
    for seed in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        echo "  [$COUNT/$TOTAL] $agent | Seed $seed"
        python training/train.py \
            --agent $agent \
            --seed  $seed
    done
done
echo ""
echo "----------------------------------------"
echo "  True Ablation Studies (5 seeds)"
echo "----------------------------------------"
ABLATIONS=(c2p_nolag c2p_noshield c2p_fixedalpha c2p_nornd)
for agent in "${ABLATIONS[@]}"; do
    echo ""
    echo "  Ablation: $agent"
    for seed in "${SEEDS[@]}"; do
        echo "  → Seed $seed"
        python training/train.py --agent $agent --seed $seed
    done
done
echo ""
echo "----------------------------------------"
echo "  Generating Figures and Tables..."
echo "----------------------------------------"
python notebooks/plot_results.py
echo ""
echo "========================================"
echo "  All 35 Experiments + Plots Complete ✅"
echo "========================================"