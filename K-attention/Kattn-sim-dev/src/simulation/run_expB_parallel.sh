#!/bin/bash
# Exp B: run transformer models with up to 3 parallel jobs (batch=128, ~5.9GB each, safe on RTX 4090)
# Picks up from where run_expB_resume.sh left off — skips all already-done runs.
# Run from: Kattn-sim-dev/src/simulation/
# Usage: bash run_expB_parallel.sh > /tmp/expB_parallel.log 2>&1 &

source env_setup.sh

MAX_JOBS=3
MAX_EPOCHS=500
PATIENCE=20
AUROC_THRESHOLD=0.99
RC_DST="abs-ran_fix2"
RC_SIZES=(5000 10000 20000 50000 100000)
MARKOV_DSTS=("markov_1_0_5000" "markov_1_0_10000" "markov_1_0_20000" "markov_1_0_50000" "markov_1_0_100000")

PIDS=()

# Submit a job; drain pool when it reaches MAX_JOBS using wait -n
submit() {
    python run_bmk.py "$@" &
    PIDS+=($!)
    if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
        wait -n
        # Remove finished PIDs from array
        live=()
        for p in "${PIDS[@]}"; do
            kill -0 "$p" 2>/dev/null && live+=("$p")
        done
        PIDS=("${live[@]}")
    fi
}

wait_all() {
    [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
    PIDS=()
}

echo "=== Exp B Parallel (transformer models, 3-parallel) ==="

# ---- RC: transformer models ----
# Already done: transformer_cls n=5000,10000 seed=0; transformer_cls_kmer n=5000 seeds=0,2; n=10000 seed=0
echo "--- RC Transformers ---"
for seed in 0 1 2; do
    for n in 5000 10000 20000 50000 100000; do
        for model in "transformer_cls" "transformer_cls_kmer"; do
            # Skip already-done
            [[ $model == "transformer_cls"      && $n == 5000  && $seed == 0 ]] && continue
            [[ $model == "transformer_cls"      && $n == 10000 && $seed == 0 ]] && continue
            [[ $model == "transformer_cls_kmer" && $n == 5000  && $seed == 0 ]] && continue
            [[ $model == "transformer_cls_kmer" && $n == 5000  && $seed == 2 ]] && continue
            [[ $model == "transformer_cls_kmer" && $n == 10000 && $seed == 0 ]] && continue
            [[ $model == "transformer_cls"      && $n == 20000 && $seed == 0 ]] && continue
            [[ $model == "transformer_cls_kmer" && $n == 20000 && $seed == 0 ]] && continue
            echo "[ExpB-RC-TF] model=$model n=$n seed=$seed"
            submit \
                --model-type "$model" \
                --test-config "$RC_DST" \
                --sample-size "$n" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "1e-5" \
                --batch-size 128 \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed"
        done
    done
done
wait_all

# ---- Markov: transformer models only (light models already done) ----
# Already done: transformer_cls_kmer markov_1_0_5000 seed=0
echo "--- Markov Transformers ---"
for seed in 0 1 2; do
    for dst in "${MARKOV_DSTS[@]}"; do
        for model in "transformer_cls" "transformer_cls_kmer"; do
            [[ $model == "transformer_cls_kmer" && $dst == "markov_1_0_5000" && $seed == 0 ]] && continue
            echo "[ExpB-Markov-TF] model=$model dst=$dst seed=$seed"
            submit \
                --model-type "$model" \
                --test-config "$dst" \
                --max-epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE" \
                --max-lr "1e-5" \
                --batch-size 128 \
                --auroc-threshold "$AUROC_THRESHOLD" \
                --version "$seed"
        done
    done
done
wait_all

echo "=== Exp B Parallel Done ==="
