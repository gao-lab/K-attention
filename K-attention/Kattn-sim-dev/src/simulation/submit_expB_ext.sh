#!/bin/bash
# submit_expB_ext.sh — Exp B extension: new sizes (1k,2k RC; 2k Markov) + seeds 3&4 for all sizes
# Total: 210 new runs
# Usage (from Kattn-sim-dev/src/simulation/ on luminary, kattn-sim env active):
#   bash submit_expB_ext.sh
#
# Partitions: alternates gpu2 / gpu32

LUSTRE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
SIMDIR=$LUSTRE/src/simulation
LOGDIR=/lustre/grp/gglab/liut/logs
mkdir -p "$LOGDIR"

export KATTN_SRC_DIR=$LUSTRE/src
export KATTN_RESOURCES_DIR=$LUSTRE/resources
export HF_DATASETS_CACHE=$LUSTRE/.hf_cache
export PYTHONPATH=$KATTN_SRC_DIR:$(echo $PYTHONPATH | tr ':' '\n' | grep -v 'Kattn-sim-dev/src' | tr '\n' ':')
cd "$SIMDIR" || exit 1

PARTITIONS=(gpu2 gpu32)
JOB_IDX=0
TOTAL=0

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

submit_job() {
    local model=$1 config=$2 sample_size=$3 seed=$4
    local lr=$(get_lr "$model")
    local batch=$(get_batch "$model")
    local part=${PARTITIONS[$((JOB_IDX % 2))]}
    sbatch --partition="$part" \
        "$SIMDIR/job_expB_ext.sh" \
        "$model" "$config" "$sample_size" "$seed" "$lr" "$batch"
    JOB_IDX=$((JOB_IDX + 1))
    TOTAL=$((TOTAL + 1))
}

RC_MODELS=(KNET_rc cnn cnn_transformer_pm mha transformer_cls transformer_cls_kmer)
MK_MODELS=(KNET   cnn cnn_transformer_pm mha transformer_cls transformer_cls_kmer)

RC_DST="abs-ran_fix2"
MK_BASE="markov_1_0_5000"   # used for Markov n=2k via --sample-size

RC_OLD_SIZES=(5000 10000 20000 50000 100000)
MK_OLD_DSTS=(markov_1_0_5000 markov_1_0_10000 markov_1_0_20000 markov_1_0_50000 markov_1_0_100000)
NEW_SEEDS=(3 4)
ALL_SEEDS=(0 1 2 3 4)

echo "=== Pre-caching new datasets ==="
python run_bmk.py --model-type KNET_rc --test-config "$RC_DST" \
    --sample-size 1000 --cache-run 2>&1 | tail -2
python run_bmk.py --model-type KNET_rc --test-config "$RC_DST" \
    --sample-size 2000 --cache-run 2>&1 | tail -2
python run_bmk.py --model-type KNET --test-config "$MK_BASE" \
    --sample-size 2000 --cache-run 2>&1 | tail -2
echo "Cache done."

echo ""
echo "=== Submitting Exp B extension ==="

# ── Block 1: RC new sizes (n=1k, 2k), all 5 seeds ──
echo "--- RC new sizes (1k,2k) × 5 seeds × 6 models = 60 runs ---"
for n in 1000 2000; do
    for seed in "${ALL_SEEDS[@]}"; do
        for model in "${RC_MODELS[@]}"; do
            submit_job "$model" "$RC_DST" "$n" "$seed"
        done
    done
done

# ── Block 2: RC existing sizes, new seeds only ──
echo "--- RC existing sizes × seeds 3,4 × 6 models = 60 runs ---"
for n in "${RC_OLD_SIZES[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        for model in "${RC_MODELS[@]}"; do
            submit_job "$model" "$RC_DST" "$n" "$seed"
        done
    done
done

# ── Block 3: Markov new size n=2k, all 5 seeds ──
echo "--- Markov n=2k × 5 seeds × 6 models = 30 runs ---"
for seed in "${ALL_SEEDS[@]}"; do
    for model in "${MK_MODELS[@]}"; do
        submit_job "$model" "$MK_BASE" "2000" "$seed"
    done
done

# ── Block 4: Markov existing sizes, new seeds only ──
echo "--- Markov existing sizes × seeds 3,4 × 6 models = 60 runs ---"
for dst in "${MK_OLD_DSTS[@]}"; do
    for seed in "${NEW_SEEDS[@]}"; do
        for model in "${MK_MODELS[@]}"; do
            # Markov existing uses full dataset (sample_size=-1)
            submit_job "$model" "$dst" "-1" "$seed"
        done
    done
done

echo ""
echo "=== Submitted $TOTAL jobs total ==="
squeue -u liut | head -30
