#!/bin/bash
# submit_crispr_cnn_tf.sh — CRISPR CNN-TF Slurm 提交脚本
# 11 datasets × 2 models (cnn_transformer, cnn_transformer_pm) × 5 folds = 110 runs
# LR: 1e-3（见 job_crispr_cnn_tf.sh）
#
# Usage (on luminary, from Kattn-sim-dev/src/crispr/):
#   bash submit_crispr_cnn_tf.sh [--dry-run]

DRY_RUN=0
[[ "$1" == "--dry-run" ]] && DRY_RUN=1

CODE_BASE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
DATA_BASE=/lustre/grp/gglab/liut/Kattn-sim-dev
SCRIPT_DIR=$CODE_BASE/src/crispr
JOB_SCRIPT=$SCRIPT_DIR/job_crispr_cnn_tf.sh
LOGDIR=/lustre/grp/gglab/liut/logs

mkdir -p "$LOGDIR/crispr_cnn_tf"

DATASETS=(
    chari2015Train293T
    doench2014-Hs
    doench2014-Mm
    doench2016_hg19
    hart2016-Hct1162lib1Avg
    hart2016-HelaLib1Avg
    hart2016-HelaLib2Avg
    hart2016-Rpe1Avg
    morenoMateos2015
    xu2015TrainHl60
    xu2015TrainKbm7
)

MODELS=(cnn_transformer cnn_transformer_pm)
SETS=(set0 set1 set2 set3 set4)
VERSION=0

TOTAL=0
SKIP=0

for DS in "${DATASETS[@]}"; do
    for SET in "${SETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            RESULT=$DATA_BASE/results/Crispr/$DS/$MODEL/$SET/$VERSION/test_metrics.csv
            if [ -f "$RESULT" ]; then
                echo "  SKIP $DS/$SET/$MODEL (已完成)"
                SKIP=$((SKIP + 1))
                continue
            fi

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "[DRY] sbatch --partition=gpu2,gpu32 --output=$LOGDIR/crispr_cnn_tf/%j.out $JOB_SCRIPT $DS $MODEL $SET $VERSION"
            else
                sbatch --partition=gpu2,gpu32 \
                       --output="$LOGDIR/crispr_cnn_tf/%j.out" \
                       "$JOB_SCRIPT" \
                       "$DS" "$MODEL" "$SET" "$VERSION"
            fi
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo ""
echo "=== 提交完成: submitted=$TOTAL skip=$SKIP ==="
[[ $DRY_RUN -eq 0 ]] && squeue -u liut | tail -5
