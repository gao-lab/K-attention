#!/bin/bash
# Experiment C: Ablation - KNET (constrained) vs KNET_uncons (unconstrained) on simulation
# Tests point-to-point constraint benefit across RC and Markov datasets
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

DATASETS=("abs-ran_fix2" "relative_fix2" "random_rand" "markov_1_0_50000")
SEEDS=(0 1 2)
MAX_EPOCHS=500
PATIENCE=20
MAX_LR=1e-4
KS=12
NK=64

echo "=== Experiment C: Constraint Ablation ==="

# Cache datasets
for dst in "${DATASETS[@]}"; do
    python run_bmk.py --model-type KNET --test-config "$dst" \
        --kernel-size "$KS" --num-kernels "$NK" --cache-run 2>/dev/null
done

# Run ablation
for seed in "${SEEDS[@]}"; do
    for dst in "${DATASETS[@]}"; do
        for model in "KNET" "KNET_uncons"; do
            echo "[ExpC] model=$model dst=$dst seed=$seed"
            python run_bmk.py \
                --model-type "$model" \
                --test-config "$dst" \
                --kernel-size "$KS" \
                --num-kernels "$NK" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$MAX_LR" \
                --version "$seed"
        done
    done
done

echo "=== Experiment C Done ==="
