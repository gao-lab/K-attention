#!/bin/bash
# Experiment B: Learning curves (data size vs AUROC) for all models
# RC: subsample abs-ran_fix2 at multiple sizes × seeds
# Markov: use pre-split markov_1_0_{5000,10000,20000,50000,100000}
# Runs 2 jobs in parallel (GPU: ~19GB / 24.5GB)
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

RC_MODELS=("KNET_rc" "cnn" "transformer_cls" "transformer_cls_kmer" "mha" "cnn_transformer_pm")
MARKOV_MODELS=("KNET" "cnn" "transformer_cls" "transformer_cls_kmer" "mha" "cnn_transformer_pm")
SEEDS=(0 1 2)
MAX_EPOCHS=500
PATIENCE=20
BATCH_SIZE=512
AUROC_THRESHOLD=0.99

RC_DST="abs-ran_fix2"
RC_SIZES=(5000 10000 20000 50000 100000)
MARKOV_DSTS=("markov_1_0_5000" "markov_1_0_10000" "markov_1_0_20000" "markov_1_0_50000" "markov_1_0_100000")

get_lr() {
    case "$1" in
        transformer*) echo "1e-5" ;;
        KNET*) echo "1e-2" ;;
        *) echo "1e-4" ;;
    esac
}

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

echo "=== Experiment B: Learning Curves ==="

# Pre-cache
echo "--- Caching datasets ---"
python run_bmk.py --model-type KNET_rc --test-config "$RC_DST" \
    --kernel-size 12 --num-kernels 64 --cache-run
for dst in "${MARKOV_DSTS[@]}"; do
    python run_bmk.py --model-type KNET --test-config "$dst" \
        --kernel-size 12 --num-kernels 64 --cache-run
done

# --- RC Learning Curves ---
echo "--- RC Learning Curves ---"
for seed in "${SEEDS[@]}"; do
    for n in "${RC_SIZES[@]}"; do
        for model in "${RC_MODELS[@]}"; do
            lr=$(get_lr "$model")
            echo "[ExpB-RC] model=$model n=$n seed=$seed"
            python run_bmk.py \
                --model-type "$model" \
                --test-config "$RC_DST" \
                --sample-size "$n" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$lr" \
                --batch-size "$BATCH_SIZE" \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed" &
            PIDS+=($!)
            flush_pids
        done
    done
done
wait_all

# --- Markov Learning Curves ---
echo "--- Markov Learning Curves ---"
for seed in "${SEEDS[@]}"; do
    for dst in "${MARKOV_DSTS[@]}"; do
        for model in "${MARKOV_MODELS[@]}"; do
            lr=$(get_lr "$model")
            echo "[ExpB-Markov] model=$model dst=$dst seed=$seed"
            python run_bmk.py \
                --model-type "$model" \
                --test-config "$dst" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$lr" \
                --batch-size "$BATCH_SIZE" \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed" &
            PIDS+=($!)
            flush_pids
        done
    done
done
wait_all

echo "=== Experiment B Done ==="
