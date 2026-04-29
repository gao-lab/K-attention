#!/bin/bash
# submit_markov_tasks.sh — Markov Task 1/2/3 学习曲线实验提交脚本
#
# Task 1: markov_0_75 (entropy=0.75, easiest) — 6 models × 6 sizes × 5 seeds = 180 runs
# Task 2: markov_1_0  (entropy=1.0, medium)   — ALREADY COMPLETE, skipped
# Task 3: markov_1_25 (entropy=1.25, hardest) — 6 models × 6 sizes × 5 seeds = 180 runs
#
# Sample sizes: 2K, 5K, 10K, 20K, 50K, 100K
# Each size maps to a (test_config, sample_size) pair — see get_config_args() below.
#
# Usage (on luminary, from Kattn-sim-dev/src/simulation/):
#   bash submit_markov_tasks.sh [--dry-run]

DRY_RUN=0
[[ "$1" == "--dry-run" ]] && DRY_RUN=1

LUSTRE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
SIMDIR=$LUSTRE/src/simulation
LOGDIR=/lustre/grp/gglab/liut/logs
JOB_SCRIPT=$SIMDIR/job_markov_tasks.sh

export KATTN_SRC_DIR=$LUSTRE/src
export KATTN_RESOURCES_DIR=$LUSTRE/resources
export HF_DATASETS_CACHE=$LUSTRE/.hf_cache
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$(echo $PYTHONPATH | tr ':' '\n' | grep -v 'Kattn-sim-dev/src' | tr '\n' ':')

cd "$SIMDIR" || exit 1

# ── Log subdirectories ──────────────────────────────────────────────────────
mkdir -p "$LOGDIR/markov_task1" "$LOGDIR/markov_task3"

# ── Helpers ─────────────────────────────────────────────────────────────────
PARTITIONS=(gpu2 gpu32)
JOB_IDX=0
TOTAL=0

get_lr() {
    case "$1" in
        transformer*) echo "1e-5" ;;
        KNET*)        echo "1e-2" ;;
        *)            echo "1e-4" ;;
    esac
}

get_batch() {
    case "$1" in
        transformer*) echo "128" ;;
        *)            echo "512" ;;
    esac
}

# Returns "TEST_CONFIG SAMPLE_SIZE" for markov_0_75 at given data size
get_0_75_args() {
    case "$1" in
        2000)   echo "markov_0_75_5000 2000" ;;
        5000)   echo "markov_0_75_5000 -1" ;;
        10000)  echo "markov_0_75_50000 10000" ;;
        20000)  echo "markov_0_75_50000 20000" ;;
        50000)  echo "markov_0_75_50000 -1" ;;
        100000) echo "markov_0_75_100000 -1" ;;
    esac
}

# Returns "TEST_CONFIG SAMPLE_SIZE" for markov_1_25 at given data size
get_1_25_args() {
    case "$1" in
        2000)   echo "markov_1_25_5000 2000" ;;
        5000)   echo "markov_1_25_5000 -1" ;;
        10000)  echo "markov_1_25_20000 10000" ;;
        20000)  echo "markov_1_25_20000 -1" ;;
        50000)  echo "markov_1_25_50000 -1" ;;
        100000) echo "markov_1_25_100000 -1" ;;
    esac
}

submit_job() {
    local task=$1 model=$2 config=$3 sample_size=$4 seed=$5
    local lr=$(get_lr "$model")
    local batch=$(get_batch "$model")
    local part=${PARTITIONS[$((JOB_IDX % 2))]}
    local outfile="$LOGDIR/$task/%j.out"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] sbatch --partition=$part --output=$outfile $JOB_SCRIPT $model $config $sample_size $seed $lr $batch"
    else
        sbatch --partition="$part" \
               --output="$outfile" \
               "$JOB_SCRIPT" \
               "$model" "$config" "$sample_size" "$seed" "$lr" "$batch"
    fi
    JOB_IDX=$((JOB_IDX + 1))
    TOTAL=$((TOTAL + 1))
}

MK_MODELS=(KNET cnn cnn_transformer_pm mha transformer_cls transformer_cls_kmer)
ALL_SEEDS=(0 1 2 3 4)
ALL_SIZES=(2000 5000 10000 20000 50000 100000)

# ── Pre-cache datasets ───────────────────────────────────────────────────────
echo "=== Pre-caching datasets (markov_0_75, markov_1_25) ==="
for n in "${ALL_SIZES[@]}"; do
    args_0_75=$(get_0_75_args "$n")
    cfg=$(echo $args_0_75 | cut -d' ' -f1)
    ss=$(echo $args_0_75 | cut -d' ' -f2)
    echo "  caching $cfg n=$ss ..."
    python run_bmk.py --model-type KNET --test-config "$cfg" \
        --sample-size "$ss" --cache-run 2>&1 | tail -1

    args_1_25=$(get_1_25_args "$n")
    cfg=$(echo $args_1_25 | cut -d' ' -f1)
    ss=$(echo $args_1_25 | cut -d' ' -f2)
    echo "  caching $cfg n=$ss ..."
    python run_bmk.py --model-type KNET --test-config "$cfg" \
        --sample-size "$ss" --cache-run 2>&1 | tail -1
done
echo "Cache done."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Markov Task 1 — markov_0_75 (entropy=0.75, easiest)
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Submitting Markov Task 1: markov_0_75 (180 runs) ==="
for n in "${ALL_SIZES[@]}"; do
    read -r cfg ss <<< "$(get_0_75_args "$n")"
    for seed in "${ALL_SEEDS[@]}"; do
        for model in "${MK_MODELS[@]}"; do
            submit_job "markov_task1" "$model" "$cfg" "$ss" "$seed"
        done
    done
done
echo "Markov Task 1: submitted."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Markov Task 2 — markov_1_0 — ALREADY COMPLETE, skipped
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Markov Task 2 (markov_1_0): already complete, skipped ==="
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Markov Task 3 — markov_1_25 (entropy=1.25, hardest)
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Submitting Markov Task 3: markov_1_25 (180 runs) ==="
for n in "${ALL_SIZES[@]}"; do
    read -r cfg ss <<< "$(get_1_25_args "$n")"
    for seed in "${ALL_SEEDS[@]}"; do
        for model in "${MK_MODELS[@]}"; do
            submit_job "markov_task3" "$model" "$cfg" "$ss" "$seed"
        done
    done
done
echo "Markov Task 3: submitted."
echo ""

echo "=== Total submitted: $TOTAL jobs ==="
[[ $DRY_RUN -eq 0 ]] && squeue -u liut | tail -5
