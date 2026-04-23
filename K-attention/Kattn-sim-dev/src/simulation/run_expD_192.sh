#!/bin/bash
# Exp D (192.168.3.17): KNET_rc on Simu16 (random_rand) learning curve
# 5 sample sizes x 3 seeds = 15 runs, 3-parallel
# Run from: Kattn-sim-dev/src/simulation/
# Usage: bash run_expD_192.sh > /tmp/expD_knet.log 2>&1 &

source env_setup.sh
PYTHON=/rd1/liut/miniconda3/envs/kattn-sim/bin/python

MAX_JOBS=3
MAX_EPOCHS=500
PATIENCE=20
AUROC_THRESHOLD=0.99
SIZES=(5000 10000 20000 50000 100000)

PIDS=()
submit() {
    $PYTHON run_bmk.py "$@" &
    PIDS+=($!)
    if [ ${#PIDS[@]} -ge $MAX_JOBS ]; then
        wait -n
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

echo "=== Exp D: KNET_rc on Simu16 (random_rand) ==="
for seed in 0 1 2; do
    for n in "${SIZES[@]}"; do
        echo "[ExpD-KNET_rc] n=$n seed=$seed"
        submit \
            --model-type KNET_rc \
            --test-config random_rand \
            --sample-size "$n" \
            --max-epochs "$MAX_EPOCHS" \
            --patience "$PATIENCE" \
            --max-lr 1e-2 \
            --batch-size 512 \
            --auroc-threshold "$AUROC_THRESHOLD" \
            --version "$seed"
    done
done
wait_all
echo "=== Exp D KNET_rc Done ==="
