#!/bin/bash
# Exp C ablation (revised): constrained vs unconstrained at 2k/5k/10k, 3 seeds
# RC: hardest task = random_rand (Simu16, 0% PWM)
# Markov: markov_1_0_5000 / markov_1_0_10000
# 2 jobs in parallel on 192.168.3.17
# Usage: nohup bash run_expC_ablation.sh > /tmp/expC_ablation.log 2>&1 &

PYTHON=/rd1/liut/miniconda3/envs/kattn-sim/bin/python
cd /rd1/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation || exit 1
source env_setup.sh

ARGS_BASE="--kernel-size 12 --max-epochs 500 --patience 20 --max-lr 1e-2 --batch-size 512 --auroc-threshold 0.99"

PIDS=()
flush_pids() {
    if [ ${#PIDS[@]} -ge 2 ]; then wait "${PIDS[@]}"; PIDS=(); fi
}
wait_all() {
    [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"; PIDS=()
}
run() {
    $PYTHON run_bmk.py $ARGS_BASE "$@" &
    PIDS+=($!)
    flush_pids
}

echo "=== Exp C Ablation: constrained vs unconstrained (2k/5k/10k, 3 seeds) ==="
echo "    RC task: random_rand (Simu16)   |   Markov task: markov_1_0"

# ── Pre-cache new configs ──
echo "--- Caching ---"
$PYTHON run_bmk.py --model-type KNET_rc      --test-config random_rand      --sample-size 2000 --cache-run 2>/dev/null
$PYTHON run_bmk.py --model-type KNET         --test-config markov_1_0_5000  --sample-size 2000 --cache-run 2>/dev/null
echo "Cache done."

# ════════════════════════════════════════════
# RC task: random_rand
# ════════════════════════════════════════════

echo "--- [RC] KNET_rc (constrained, nk=64) n=2k seeds 0-2 ---"
for seed in 0 1 2; do
    run --model-type KNET_rc --test-config random_rand --num-kernels 64 --sample-size 2000 --version $seed
done
wait_all

echo "--- [RC] KNET_uncons_rc (nk=64) n=2k,5k,10k seeds 0-2 ---"
for seed in 0 1 2; do
    for n in 2000 5000 10000; do
        run --model-type KNET_uncons_rc --test-config random_rand --num-kernels 64 --sample-size $n --version $seed
    done
done
wait_all

echo "--- [RC] KNET_uncons_rc (nk=5, param-matched) n=2k,5k,10k seeds 0-2 ---"
for seed in 0 1 2; do
    for n in 2000 5000 10000; do
        run --model-type KNET_uncons_rc --test-config random_rand --num-kernels 5 --sample-size $n --version $seed
    done
done
wait_all

# ════════════════════════════════════════════
# Markov task
# ════════════════════════════════════════════

echo "--- [Markov] KNET (constrained, nk=64) n=2k seeds 0-2 ---"
for seed in 0 1 2; do
    run --model-type KNET --test-config markov_1_0_5000 --num-kernels 64 --sample-size 2000 --version $seed
done
wait_all

echo "--- [Markov] KNET_uncons (nk=64) n=2k seeds 0-2; n=5k,10k seeds 1-2 ---"
for seed in 0 1 2; do
    run --model-type KNET_uncons --test-config markov_1_0_5000  --num-kernels 64 --sample-size 2000 --version $seed
done
wait_all
for seed in 1 2; do
    run --model-type KNET_uncons --test-config markov_1_0_5000  --num-kernels 64 --version $seed
    run --model-type KNET_uncons --test-config markov_1_0_10000 --num-kernels 64 --version $seed
done
wait_all

echo "--- [Markov] KNET_uncons (nk=5, param-matched) n=2k seeds 0-2; n=5k,10k seeds 1-2 ---"
for seed in 0 1 2; do
    run --model-type KNET_uncons --test-config markov_1_0_5000  --num-kernels 5 --sample-size 2000 --version $seed
done
wait_all
for seed in 1 2; do
    run --model-type KNET_uncons --test-config markov_1_0_5000  --num-kernels 5 --version $seed
    run --model-type KNET_uncons --test-config markov_1_0_10000 --num-kernels 5 --version $seed
done
wait_all

echo "=== Exp C Ablation Done ==="
