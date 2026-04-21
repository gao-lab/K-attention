#!/bin/bash
# Quick validation: verify KNET achieves AUROC~1.0 with lr=1e-2
# Uses 20000 samples for fast testing
# Run from: Kattn-sim-dev/src/simulation/

source env_setup.sh

KS=12
NK=128
MAX_EPOCHS=500
PATIENCE=30
MAX_LR=1e-2
BATCH_SIZE=512
AUROC_THRESHOLD=0.99
N=20000                       # fast test size
MARKOV_DST="markov_1_0_20000" # pre-split Markov 20k

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

echo "=== Validation (n=20000): KNET with lr=1e-2 ==="

# Cache datasets first
echo "--- Caching ---"
python run_bmk.py --model-type KNET_rc --test-config abs-ran_fix2 \
    --kernel-size "$KS" --num-kernels "$NK" --sample-size "$N" --cache-run
python run_bmk.py --model-type KNET --test-config "$MARKOV_DST" \
    --kernel-size "$KS" --num-kernels "$NK" --cache-run

echo "[1] KNET_rc on abs-ran_fix2 n=20k (RC plain, Simu7)"
python run_bmk.py \
    --model-type KNET_rc --test-config abs-ran_fix2 \
    --sample-size "$N" \
    --kernel-size "$KS" --num-kernels "$NK" \
    --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
    --max-lr "$MAX_LR" --batch-size "$BATCH_SIZE" \
    --auroc-threshold "$AUROC_THRESHOLD" --version 0 &
PIDS+=($!)

echo "[2] KNET on abs-ran_fix2 n=20k (RC masked, compare)"
python run_bmk.py \
    --model-type KNET --test-config abs-ran_fix2 \
    --sample-size "$N" \
    --kernel-size "$KS" --num-kernels "$NK" \
    --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
    --max-lr "$MAX_LR" --batch-size "$BATCH_SIZE" \
    --auroc-threshold "$AUROC_THRESHOLD" --version 0 &
PIDS+=($!)

flush_pids

echo "[3] KNET on markov_1_0_20000 (Markov masked)"
python run_bmk.py \
    --model-type KNET --test-config "$MARKOV_DST" \
    --kernel-size "$KS" --num-kernels "$NK" \
    --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
    --max-lr "$MAX_LR" --batch-size "$BATCH_SIZE" \
    --auroc-threshold "$AUROC_THRESHOLD" --version 0 &
PIDS+=($!)

echo "[4] KNET_uncons_rc on abs-ran_fix2 n=20k (RC ablation baseline)"
python run_bmk.py \
    --model-type KNET_uncons_rc --test-config abs-ran_fix2 \
    --sample-size "$N" \
    --kernel-size "$KS" --num-kernels "$NK" \
    --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
    --max-lr "$MAX_LR" --batch-size "$BATCH_SIZE" \
    --auroc-threshold "$AUROC_THRESHOLD" --version 0 &
PIDS+=($!)

flush_pids
wait_all

echo "=== Validation Done ==="
echo "--- Results ---"
cat ../../results/exp_results.csv
