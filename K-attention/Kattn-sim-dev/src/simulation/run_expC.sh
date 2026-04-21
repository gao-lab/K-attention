#!/bin/bash
# Experiment C: Ablation - constrained vs unconstrained KNET
# RC: KNET_rc vs KNET_uncons_rc; Markov: KNET vs KNET_uncons
# Runs 2 jobs in parallel
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

RC_DSTS=("abs-ran_fix2" "relative_fix2" "random_rand")
MARKOV_DSTS=("markov_1_0_50000")
SEEDS=(0 1 2)
MAX_EPOCHS=500
PATIENCE=20
MAX_LR=1e-2
BATCH_SIZE=512
AUROC_THRESHOLD=0.99
KS=12
NK=64

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

echo "=== Experiment C: Constraint Ablation ==="

# Cache
for dst in "${RC_DSTS[@]}"; do
    python run_bmk.py --model-type KNET_rc --test-config "$dst" \
        --kernel-size "$KS" --num-kernels "$NK" --cache-run 2>/dev/null
done
for dst in "${MARKOV_DSTS[@]}"; do
    python run_bmk.py --model-type KNET --test-config "$dst" \
        --kernel-size "$KS" --num-kernels "$NK" --cache-run 2>/dev/null
done

echo "--- RC Ablation ---"
for seed in "${SEEDS[@]}"; do
    for dst in "${RC_DSTS[@]}"; do
        for model in "KNET_rc" "KNET_uncons_rc"; do
            echo "[ExpC-RC] model=$model dst=$dst seed=$seed"
            python run_bmk.py \
                --model-type "$model" --test-config "$dst" \
                --kernel-size "$KS" --num-kernels "$NK" \
                --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
                --max-lr "$MAX_LR" --batch-size "$BATCH_SIZE" \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed" &
            PIDS+=($!)
            flush_pids
        done
    done
done
wait_all

echo "--- Markov Ablation ---"
for seed in "${SEEDS[@]}"; do
    for dst in "${MARKOV_DSTS[@]}"; do
        for model in "KNET" "KNET_uncons"; do
            echo "[ExpC-Markov] model=$model dst=$dst seed=$seed"
            python run_bmk.py \
                --model-type "$model" --test-config "$dst" \
                --kernel-size "$KS" --num-kernels "$NK" \
                --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
                --max-lr "$MAX_LR" --batch-size "$BATCH_SIZE" \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed" &
            PIDS+=($!)
            flush_pids
        done
    done
done
wait_all

echo "=== Experiment C Done ==="
