#!/bin/bash
# submit_rc_tasks.sh — RC Task 1/2/3 学习曲线实验提交脚本
#
# Task 1: random_fix2 (40% PWM) — 6 models × 6 sizes × 5 seeds = 180 runs
# Task 2: random_fix1 (20% PWM) — 6 models × 6 sizes × 5 seeds = 180 runs
# Task 3: random_rand (0%  PWM) — 补全至 5 seeds                = 148 runs
#
# Model-specific resources:
#   KNET*             → --time=16:00:00 --mem=64G
#   transformer*      → --time=12:00:00 --mem=48G
#   cnn, mha, cnn_*   → --time=08:00:00 --mem=32G
#
# Usage (on luminary, from Kattn-sim-dev/src/simulation/):
#   bash submit_rc_tasks.sh [--dry-run]
#
# With --dry-run: prints sbatch commands without submitting.

DRY_RUN=0
[[ "$1" == "--dry-run" ]] && DRY_RUN=1

LUSTRE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
SIMDIR=$LUSTRE/src/simulation
LOGDIR=/lustre/grp/gglab/liut/logs
JOB_SCRIPT=$SIMDIR/job_rc_tasks.sh

export KATTN_SRC_DIR=$LUSTRE/src
export KATTN_RESOURCES_DIR=$LUSTRE/resources
export HF_DATASETS_CACHE=$LUSTRE/.hf_cache
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'Kattn-sim-dev/src' | tr '\n' ':')

cd "$SIMDIR" || exit 1

# ── Log subdirectories ──────────────────────────────────────────────────────
mkdir -p "$LOGDIR/rc_task1" "$LOGDIR/rc_task2" "$LOGDIR/rc_task3"

# ── Helpers ─────────────────────────────────────────────────────────────────
TOTAL=0
SKIP=0

RESULTS_CSV=$LUSTRE/results/exp_results.csv

is_done() {
    # is_done MODEL CONFIG SAMPLE_SIZE SEED
    # Returns 0 (true) if a matching row exists in exp_results.csv
    local model=$1 config=$2 sample_size=$3 seed=$4
    if [ ! -f "$RESULTS_CSV" ]; then
        return 1
    fi
    # CSV columns: timestamp,model_type,test_config,kernel_size,num_kernels,sample_size,max_lr,version,val_auroc
    grep -qE "^[^,]+,${model},${config},[^,]+,[^,]+,${sample_size},[^,]+,${seed}," \
         "$RESULTS_CSV" 2>/dev/null
}

get_lr() {
    case "$1" in
        transformer*) echo "1e-5" ;;
        KNET*)        echo "1e-2" ;;
        *)            echo "1e-4" ;;   # cnn, mha, cnn_transformer_pm
    esac
}

get_batch() {
    case "$1" in
        transformer*) echo "128" ;;
        *)            echo "512" ;;
    esac
}

get_time() {
    case "$1" in
        KNET*)        echo "16:00:00" ;;
        transformer*) echo "12:00:00" ;;
        *)            echo "08:00:00" ;;
    esac
}

get_mem() {
    case "$1" in
        KNET*)        echo "64G" ;;
        transformer*) echo "48G" ;;
        *)            echo "32G" ;;
    esac
}

submit_job() {
    local task=$1 model=$2 config=$3 sample_size=$4 seed=$5
    if is_done "$model" "$config" "$sample_size" "$seed"; then
        echo "  SKIP $model/$config/n=$sample_size/seed=$seed (已完成)"
        SKIP=$((SKIP + 1))
        return
    fi
    local lr=$(get_lr "$model")
    local batch=$(get_batch "$model")
    local time=$(get_time "$model")
    local mem=$(get_mem "$model")
    local outfile="$LOGDIR/$task/%j.out"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] sbatch --partition=gpu2,gpu32 --time=$time --mem=$mem --output=$outfile $JOB_SCRIPT $model $config $sample_size $seed $lr $batch"
    else
        sbatch --partition=gpu2,gpu32 \
               --time="$time" \
               --mem="$mem" \
               --output="$outfile" \
               "$JOB_SCRIPT" \
               "$model" "$config" "$sample_size" "$seed" "$lr" "$batch"
    fi
    TOTAL=$((TOTAL + 1))
}

RC_MODELS=(KNET_rc cnn cnn_transformer_pm mha transformer_cls transformer_cls_kmer)
ALL_SEEDS=(0 1 2 3 4)
NEW_SEEDS=(3 4)
ALL_SIZES=(1000 2000 5000 20000 50000 100000)

# ── Pre-cache datasets ───────────────────────────────────────────────────────
echo "=== Pre-caching datasets (random_fix2, random_fix1) ==="
for config in random_fix2 random_fix1; do
    for n in "${ALL_SIZES[@]}"; do
        echo "  caching $config n=$n ..."
        python run_bmk.py --model-type KNET_rc --test-config "$config" \
            --sample-size "$n" --cache-run 2>&1 | tail -1
    done
done
# random_rand caches already exist; run full dataset cache just in case
python run_bmk.py --model-type KNET_rc --test-config random_rand \
    --sample-size 1000 --cache-run 2>&1 | tail -1
echo "Cache done."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# RC Task 1 — random_fix2 (40% PWM) — nearly complete (177/180)
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Submitting Task 1: random_fix2 (180 runs) ==="
for n in "${ALL_SIZES[@]}"; do
    for seed in "${ALL_SEEDS[@]}"; do
        for model in "${RC_MODELS[@]}"; do
            submit_job "rc_task1" "$model" "random_fix2" "$n" "$seed"
        done
    done
done
echo "Task 1: submitted."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# RC Task 2 — random_fix1 (20% PWM) — COMPLETE (180/180), expect all skipped
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Submitting Task 2: random_fix1 (180 runs, expect all skipped) ==="
for n in "${ALL_SIZES[@]}"; do
    for seed in "${ALL_SEEDS[@]}"; do
        for model in "${RC_MODELS[@]}"; do
            submit_job "rc_task2" "$model" "random_fix1" "$n" "$seed"
        done
    done
done
echo "Task 2: submitted."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# RC Task 3 — random_rand (0% PWM) — supplement to 5 seeds
# Existing: KNET_rc seeds 0-2 (partial), cnn_transformer_pm seeds 0-4 (partial)
#           transformer_cls/kmer/cnn/mha: incomplete
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Submitting Task 3: random_rand (148 runs, filling gaps) ==="

# KNET_rc: needs 1K (5 seeds, has only seeds 3,4) + seeds 3,4 for 5K,20K,50K,100K
echo "--- KNET_rc ---"
for seed in "${ALL_SEEDS[@]}"; do
    submit_job "rc_task3" "KNET_rc" "random_rand" "1000" "$seed"
done
for n in 5000 20000 50000 100000; do
    for seed in "${NEW_SEEDS[@]}"; do
        submit_job "rc_task3" "KNET_rc" "random_rand" "$n" "$seed"
    done
done

# cnn_transformer_pm: needs 1K (5 seeds) + seeds 3,4 for 2K,5K,20K,50K,100K
echo "--- cnn_transformer_pm ---"
for seed in "${ALL_SEEDS[@]}"; do
    submit_job "rc_task3" "cnn_transformer_pm" "random_rand" "1000" "$seed"
done
for n in 2000 5000 20000 50000 100000; do
    for seed in "${NEW_SEEDS[@]}"; do
        submit_job "rc_task3" "cnn_transformer_pm" "random_rand" "$n" "$seed"
    done
done

# transformer_cls, transformer_cls_kmer, cnn, mha: no or incomplete data → all 6 sizes × 5 seeds
echo "--- transformer_cls, transformer_cls_kmer, cnn, mha ---"
for model in "transformer_cls" "transformer_cls_kmer" "cnn" "mha"; do
    for n in "${ALL_SIZES[@]}"; do
        for seed in "${ALL_SEEDS[@]}"; do
            submit_job "rc_task3" "$model" "random_rand" "$n" "$seed"
        done
    done
done

echo "Task 3: submitted."
echo ""
echo "=== Total submitted: $TOTAL jobs (skip=$SKIP) ==="
[[ $DRY_RUN -eq 0 ]] && squeue -u liut | tail -5
