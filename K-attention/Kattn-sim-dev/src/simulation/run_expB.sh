#!/bin/bash
# Experiment B: Learning curves (data size vs AUROC) for all models
# RC: subsample abs-ran_fix2 at multiple sizes × seeds
# Markov: use pre-split markov_1_0_{5000,10000,20000,50000,100000}
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

MODELS=("KNET" "cnn" "transformer_cls" "transformer_cls_kmer" "mha" "cnn_transformer_pm")
SEEDS=(0 1 2)
MAX_EPOCHS=500
PATIENCE=20

# RC subsample sizes (from 100k dataset)
RC_DST="abs-ran_fix2"
RC_SIZES=(5000 10000 20000 50000 100000)

# Markov pre-split datasets (H=1.0)
MARKOV_DSTS=("markov_1_0_5000" "markov_1_0_10000" "markov_1_0_20000" "markov_1_0_50000" "markov_1_0_100000")

get_lr() {
    local m=$1
    case "$m" in
        transformer*) echo "1e-5" ;;
        cnn*) echo "1e-4" ;;
        mha) echo "1e-4" ;;
        KNET*) echo "1e-4" ;;
        *) echo "1e-4" ;;
    esac
}

echo "=== Experiment B: Learning Curves ==="

# Pre-cache RC dataset (full size, individual sizes use --sample-size so same cache key)
echo "--- Caching RC dataset ---"
python run_bmk.py --model-type KNET --test-config "$RC_DST" \
    --kernel-size 12 --num-kernels 64 --cache-run

# Pre-cache Markov datasets
echo "--- Caching Markov datasets ---"
for dst in "${MARKOV_DSTS[@]}"; do
    echo "Caching $dst ..."
    python run_bmk.py --model-type KNET --test-config "$dst" \
        --kernel-size 12 --num-kernels 64 --cache-run
done

# --- RC Learning Curves ---
echo "--- RC Learning Curves: $RC_DST ---"
for seed in "${SEEDS[@]}"; do
    for n in "${RC_SIZES[@]}"; do
        for model in "${MODELS[@]}"; do
            lr=$(get_lr "$model")
            echo "[ExpB-RC] model=$model n=$n seed=$seed"
            python run_bmk.py \
                --model-type "$model" \
                --test-config "$RC_DST" \
                --sample-size "$n" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$lr" \
                --version "$seed"
        done
    done
done

# --- Markov Learning Curves ---
echo "--- Markov Learning Curves ---"
for seed in "${SEEDS[@]}"; do
    for dst in "${MARKOV_DSTS[@]}"; do
        for model in "${MODELS[@]}"; do
            lr=$(get_lr "$model")
            echo "[ExpB-Markov] model=$model dst=$dst seed=$seed"
            python run_bmk.py \
                --model-type "$model" \
                --test-config "$dst" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$lr" \
                --version "$seed"
        done
    done
done

echo "=== Experiment B Done ==="
