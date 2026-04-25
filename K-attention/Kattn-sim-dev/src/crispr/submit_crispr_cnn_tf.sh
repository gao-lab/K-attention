#!/bin/bash
# 提交 cnn_transformer × 11 datasets × 5 folds × version=0 的 Slurm array job
#
# 使用方式（在 luminary 上执行）：
#   cd /lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev/src/crispr
#   bash submit_crispr_cnn_tf.sh

CODE_BASE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
DATA_BASE=/lustre/grp/gglab/liut/Kattn-sim-dev
SCRIPT_DIR=$CODE_BASE/src/crispr
TASK_LIST=$SCRIPT_DIR/crispr_task_list.txt
LOGDIR=/lustre/grp/gglab/liut/logs

mkdir -p "$LOGDIR"

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

# ── Step 1: 生成任务列表（跳过已完成）──
echo "生成任务列表..."
> "$TASK_LIST"
for DS in "${DATASETS[@]}"; do
    for SET in "${SETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            RESULT=$DATA_BASE/results/Crispr/$DS/$MODEL/$SET/$VERSION/test_metrics.csv
            if [ -f "$RESULT" ]; then
                echo "  SKIP $DS $SET $MODEL (已完成)"
            else
                echo "$DS $SET $MODEL" >> "$TASK_LIST"
            fi
        done
    done
done

N=$(wc -l < "$TASK_LIST")
echo "待运行任务数：$N"
if [ "$N" -eq 0 ]; then
    echo "所有任务已完成，无需提交"
    exit 0
fi

ARRAY_END=$((N - 1))
HALF=$((N / 2))

# ── Step 2: 分两半投到 gpu2 / gpu32 ──
echo "提交 Slurm array job..."

sbatch --partition=gpu2 \
       --array=0-$((HALF - 1)) \
       "$SCRIPT_DIR/job_crispr_cnn_tf.sh"

sbatch --partition=gpu32 \
       --array=${HALF}-${ARRAY_END} \
       "$SCRIPT_DIR/job_crispr_cnn_tf.sh"

echo ""
echo "已提交。查看状态：squeue -u liut"
echo "日志目录：$LOGDIR"
