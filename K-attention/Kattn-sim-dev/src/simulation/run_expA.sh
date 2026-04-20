#!/bin/bash
# Experiment A: Hyperparameter scan (kernel_size × num_kernels) for KNET
# Datasets: abs-ran_fix2 (RC Simu7) and markov_1_0_50000 (Markov H=1.0)
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

KERNEL_SIZES=(6 8 10 12 15)
NUM_KERNELS=(16 32 64 128)
DATASETS=("abs-ran_fix2" "markov_1_0_50000")
MAX_EPOCHS=200
PATIENCE=20
MAX_LR=1e-4

echo "=== Experiment A: Hyperparameter Scan ==="

# Pre-cache datasets
echo "--- Caching datasets ---"
for dst in "${DATASETS[@]}"; do
    echo "Caching $dst ..."
    python run_bmk.py --model-type KNET --test-config "$dst" \
        --kernel-size 12 --num-kernels 64 --cache-run
done

# Run sweep
echo "--- Starting hyperparameter sweep ---"
for dst in "${DATASETS[@]}"; do
    for ks in "${KERNEL_SIZES[@]}"; do
        for nk in "${NUM_KERNELS[@]}"; do
            echo "[ExpA] dataset=$dst kernel_size=$ks num_kernels=$nk"
            python run_bmk.py \
                --model-type KNET \
                --test-config "$dst" \
                --kernel-size "$ks" \
                --num-kernels "$nk" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$MAX_LR" \
                --version 0
        done
    done
done

echo "=== Experiment A Done ==="
