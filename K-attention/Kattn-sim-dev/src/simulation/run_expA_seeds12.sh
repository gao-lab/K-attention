#!/bin/bash
# Experiment A extra seeds: run seed=1 and seed=2 for all hyperparameter configs
# RC tasks: KNET_rc (abs-ran_fix2); Markov: KNET (markov_1_0_50000)
# 2 jobs in parallel; runs from: Kattn-sim-dev/src/simulation/

PYTHON=/rd1/liut/miniconda3/envs/kattn-sim/bin/python
source env_setup.sh

KERNEL_SIZES=(6 8 10 12 15)
NUM_KERNELS=(16 32 64 128)
MAX_EPOCHS=500
PATIENCE=20
MAX_LR=1e-2
BATCH_SIZE=512
AUROC_THRESHOLD=0.99

echo "=== Experiment A — extra seeds (1, 2) ==="

# Helper: 2 jobs in parallel
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
    local ver=$1; shift
    $PYTHON run_bmk.py "$@" \
        --max-epochs "$MAX_EPOCHS" \
        --patience "$PATIENCE" \
        --max-lr "$MAX_LR" \
        --batch-size "$BATCH_SIZE" \
        --auroc-threshold "$AUROC_THRESHOLD" \
        --version "$ver" &
    PIDS+=($!)
    flush_pids
}

for SEED in 1 2; do
    echo "--- Seed $SEED ---"

    echo "[RC] abs-ran_fix2 seed=$SEED"
    for ks in "${KERNEL_SIZES[@]}"; do
        for nk in "${NUM_KERNELS[@]}"; do
            echo "  kernel_size=$ks num_kernels=$nk"
            run_job $SEED --model-type KNET_rc --test-config abs-ran_fix2 \
                --kernel-size "$ks" --num-kernels "$nk"
        done
    done
    wait_all

    echo "[Markov] markov_1_0_50000 seed=$SEED"
    for ks in "${KERNEL_SIZES[@]}"; do
        for nk in "${NUM_KERNELS[@]}"; do
            echo "  kernel_size=$ks num_kernels=$nk"
            run_job $SEED --model-type KNET --test-config markov_1_0_50000 \
                --kernel-size "$ks" --num-kernels "$nk"
        done
    done
    wait_all
done

echo "=== Experiment A extra seeds Done ==="
