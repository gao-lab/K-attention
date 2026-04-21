#!/bin/bash
# Resume Exp B: transformer RC runs (sequential, batch=128) + all Markov runs
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

SEEDS=(0 1 2)
MAX_EPOCHS=500
PATIENCE=20
AUROC_THRESHOLD=0.99
RC_DST="abs-ran_fix2"
RC_SIZES=(5000 10000 20000 50000 100000)
MARKOV_MODELS=("KNET" "cnn" "transformer_cls" "transformer_cls_kmer" "mha" "cnn_transformer_pm")
MARKOV_DSTS=("markov_1_0_5000" "markov_1_0_10000" "markov_1_0_20000" "markov_1_0_50000" "markov_1_0_100000")

get_lr() {
    case "$1" in
        transformer*) echo "1e-5" ;;
        KNET*) echo "1e-2" ;;
        *) echo "1e-4" ;;
    esac
}

# Parallel helper for light models (batch=512)
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

echo "=== Exp B Resume ==="

# ---- RC: transformer models only, sequential, batch=128 ----
echo "--- RC Transformers (sequential, batch=128) ---"
for seed in "${SEEDS[@]}"; do
    for n in "${RC_SIZES[@]}"; do
        for model in "transformer_cls" "transformer_cls_kmer"; do
            # Skip already-done runs
            if [ "$model" = "transformer_cls_kmer" ] && [ "$n" = "5000" ] && [ "$seed" = "2" ]; then
                echo "[skip] $model n=$n seed=$seed already done"; continue
            fi
            if [ "$model" = "transformer_cls_kmer" ] && [ "$n" = "5000" ] && [ "$seed" = "0" ]; then
                echo "[skip] $model n=$n seed=$seed already done"; continue
            fi
            if [ "$model" = "transformer_cls_kmer" ] && [ "$n" = "10000" ] && [ "$seed" = "0" ]; then
                echo "[skip] $model n=$n seed=$seed already done"; continue
            fi
            if [ "$model" = "transformer_cls" ] && [ "$n" = "5000" ] && [ "$seed" = "0" ]; then
                echo "[skip] $model n=$n seed=$seed already done"; continue
            fi
            if [ "$model" = "transformer_cls" ] && [ "$n" = "10000" ] && [ "$seed" = "0" ]; then
                echo "[skip] $model n=$n seed=$seed already done"; continue
            fi
            lr=$(get_lr "$model")
            echo "[ExpB-RC-TF] model=$model n=$n seed=$seed"
            python run_bmk.py \
                --model-type "$model" \
                --test-config "$RC_DST" \
                --sample-size "$n" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$lr" \
                --batch-size 128 \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed"
        done
    done
done

# ---- Markov: all models ----
# Light models (KNET, cnn, mha, cnn_transformer_pm) run in parallel pairs
# Transformer models run sequentially with batch=128
echo "--- Markov Learning Curves (transformer models only; light models already done) ---"
for seed in "${SEEDS[@]}"; do
    for dst in "${MARKOV_DSTS[@]}"; do
        # Transformer models: sequential, batch=128
        for model in "transformer_cls" "transformer_cls_kmer"; do
            # Skip transformer_cls_kmer markov_1_0_5000 seed=0 (already done)
            if [ "$model" = "transformer_cls_kmer" ] && [ "$dst" = "markov_1_0_5000" ] && [ "$seed" = "0" ]; then
                echo "[skip] $model dst=$dst seed=$seed already done"
                continue
            fi
            lr=$(get_lr "$model")
            echo "[ExpB-Markov-TF] model=$model dst=$dst seed=$seed"
            python run_bmk.py \
                --model-type "$model" \
                --test-config "$dst" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "$lr" \
                --batch-size 128 \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed"
        done
    done
done

echo "=== Exp B Resume Done ==="
