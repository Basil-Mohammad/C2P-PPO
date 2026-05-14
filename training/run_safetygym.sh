#!/bin/bash
# ─────────────────────────────────────────
# run_safetygym.sh — SafetyPointGoal Experiments
# 2 Agents × 10 Seeds = 20 Experiments
# ─────────────────────────────────────────

echo "========================================"
echo "  SafetyPointGoal1 Experiments"
echo "  2 Agents × 10 Seeds = 20 Runs"
echo "========================================"

SEEDS=(42 123 456 789 1024 2048 3072 4096 5120 6144)
AGENTS=(ppo_baseline c2p_ppo)

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
        python training/train_safetygym.py \
            --agent $agent \
            --seed  $seed
    done
done

echo ""
echo "========================================"
echo "  SafetyPointGoal Complete ✅"
echo "========================================"
