#!/bin/bash
# Exp C small-sample ablation: constrained vs unconstrained (same nk and param-matched nk=5)
# RC: abs-ran_fix2, n=1k,2k,5k; Markov: markov_1_0_5000 (n=2k,5k)
# 2 jobs in parallel on 192.168.3.17
# Run from: Kattn-sim-dev/src/simulation/
# Usage: nohup bash run_expC_small.sh > /tmp/expC_small.log 2>&1 &

PYTHON=/rd1/liut/miniconda3/envs/kattn-sim/bin/python
source env_setup.sh

SEEDS=(0 1 2)
MAX_EPOCHS=500
PATIENCE=20
MAX_LR=1e-2
BATCH_SIZE=512
AUROC_THRESHOLD=0.99
RC_DST="abs-ran_fix2"
MK_BASE="markov_1_0_5000"

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
    $PYTHON run_bmk.py "$@" \
        --max-epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
        --max-lr "$MAX_LR" --batch-size "$BATCH_SIZE" \
        --auroc-threshold "$AUROC_THRESHOLD" &
    PIDS+=($!)
    flush_pids
}

echo "=== Exp C small-sample ablation (constrained vs unconstrained) ==="
echo "  Wattn params: constrained nk=64 → 37,632; uncons nk=64 → 451,584; uncons nk=5 → 35,280"

# ── Pre-cache ──
echo "--- Caching ---"
$PYTHON run_bmk.py --model-type KNET_rc --test-config "$RC_DST" \
    --sample-size 1000 --cache-run 2>/dev/null
$PYTHON run_bmk.py --model-type KNET_rc --test-config "$RC_DST" \
    --sample-size 2000 --cache-run 2>/dev/null
$PYTHON run_bmk.py --model-type KNET --test-config "$MK_BASE" \
    --sample-size 2000 --cache-run 2>/dev/null

# ═══════════════════════════════════════════════
# RC task
# ═══════════════════════════════════════════════
echo "--- RC: KNET_rc (constrained, nk=64) n=1k,2k ---"
for seed in "${SEEDS[@]}"; do
    for n in 1000 2000; do
        run_job --model-type KNET_rc --test-config "$RC_DST" \
            --kernel-size 12 --num-kernels 64 --sample-size "$n" --version "$seed"
    done
done
wait_all

echo "--- RC: KNET_uncons_rc (nk=64, 12× params) n=1k,2k,5k ---"
for seed in "${SEEDS[@]}"; do
    for n in 1000 2000 5000; do
        run_job --model-type KNET_uncons_rc --test-config "$RC_DST" \
            --kernel-size 12 --num-kernels 64 --sample-size "$n" --version "$seed"
    done
done
wait_all

echo "--- RC: KNET_uncons_rc (nk=5, param-matched) n=1k,2k,5k ---"
for seed in "${SEEDS[@]}"; do
    for n in 1000 2000 5000; do
        run_job --model-type KNET_uncons_rc --test-config "$RC_DST" \
            --kernel-size 12 --num-kernels 5 --sample-size "$n" --version "$seed"
    done
done
wait_all

# ═══════════════════════════════════════════════
# Markov task
# ═══════════════════════════════════════════════
echo "--- Markov: KNET (constrained, nk=64) n=2k ---"
for seed in "${SEEDS[@]}"; do
    run_job --model-type KNET --test-config "$MK_BASE" \
        --kernel-size 12 --num-kernels 64 --sample-size 2000 --version "$seed"
done
wait_all

echo "--- Markov: KNET_uncons (nk=64, 12× params) n=2k,5k ---"
for seed in "${SEEDS[@]}"; do
    for n in 2000 5000; do
        run_job --model-type KNET_uncons --test-config "$MK_BASE" \
            --kernel-size 12 --num-kernels 64 --sample-size "$n" --version "$seed"
    done
done
wait_all

echo "--- Markov: KNET_uncons (nk=5, param-matched) n=2k,5k ---"
for seed in "${SEEDS[@]}"; do
    for n in 2000 5000; do
        run_job --model-type KNET_uncons --test-config "$MK_BASE" \
            --kernel-size 12 --num-kernels 5 --sample-size "$n" --version "$seed"
    done
done
wait_all

echo "=== Exp C small-sample ablation Done ==="
