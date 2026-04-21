#!/bin/bash
# Experiment A: Hyperparameter scan (kernel_size × num_kernels) for KNET
# RC tasks use KNET_rc (plain); Markov uses KNET (diagonal mask)
# Runs 2 jobs in parallel (GPU: ~19GB / 24.5GB)
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

KERNEL_SIZES=(6 8 10 12 15)
NUM_KERNELS=(16 32 64 128)
MAX_EPOCHS=500
PATIENCE=20
MAX_LR=1e-2
BATCH_SIZE=512
AUROC_THRESHOLD=0.99

echo "=== Experiment A: Hyperparameter Scan ==="

# Pre-cache datasets
echo "--- Caching datasets ---"
python run_bmk.py --model-type KNET_rc --test-config abs-ran_fix2 \
    --kernel-size 12 --num-kernels 64 --cache-run
python run_bmk.py --model-type KNET --test-config markov_1_0_50000 \
    --kernel-size 12 --num-kernels 64 --cache-run

# Helper: run up to 2 jobs in parallel
PIDS=()
flush_pids() {
    if [ ${#PIDS[@]} -ge 2 ]; then
        wait "${PIDS[@]}"
        PIDS=()
    fi
}
wait_all() {
    [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}" && PIDS=()
}

run_job() {
    python run_bmk.py "$@" \
        --max-epochs "$MAX_EPOCHS" \
        --patience "$PATIENCE" \
        --max-lr "$MAX_LR" \
        --batch-size "$BATCH_SIZE" \
        --auroc-threshold "$AUROC_THRESHOLD" \
        --version 0 &
    PIDS+=($!)
    flush_pids
}

# RC sweep: KNET_rc
echo "--- RC Sweep: abs-ran_fix2 ---"
for ks in "${KERNEL_SIZES[@]}"; do
    for nk in "${NUM_KERNELS[@]}"; do
        echo "[ExpA-RC] kernel_size=$ks num_kernels=$nk"
        run_job --model-type KNET_rc --test-config abs-ran_fix2 \
            --kernel-size "$ks" --num-kernels "$nk"
    done
done
wait_all

# Markov sweep: KNET
echo "--- Markov Sweep: markov_1_0_50000 ---"
for ks in "${KERNEL_SIZES[@]}"; do
    for nk in "${NUM_KERNELS[@]}"; do
        echo "[ExpA-Markov] kernel_size=$ks num_kernels=$nk"
        run_job --model-type KNET --test-config markov_1_0_50000 \
            --kernel-size "$ks" --num-kernels "$nk"
    done
done
wait_all

echo "=== Experiment A Done ==="
