#!/bin/bash
#SBATCH --job-name=crispr_cnn_tf
#SBATCH --output=/lustre/grp/gglab/liut/logs/crispr_cnn_tf_%j.log
#SBATCH --error=/lustre/grp/gglab/liut/logs/crispr_cnn_tf_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# CRISPR CNN-Transformer baseline — 5-fold CV on luminary
# 11 datasets × 2 models × 5 sets × version=0 = 110 runs
# 单 GPU 节点串行执行（约 15-20 GPU-hours），已完成自动跳过

BASE=/lustre/grp/gglab/liut/Kattn-sim-dev
export KATTN_SRC_DIR=$BASE/src
export KATTN_RESOURCES_DIR=$BASE/resources
export HF_DATASETS_CACHE=$BASE/.hf_cache
export HF_HOME=$HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$PYTHONPATH

SCRIPT_DIR=$BASE/src/crispr
RESULT_BASE=$BASE/results/Crispr

source /lustre/grp/gglab/liut/mambaforge/etc/profile.d/conda.sh
conda activate kattn-sim

cd "$SCRIPT_DIR"

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
FAIL=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for DS in "${DATASETS[@]}"; do
    DATA_DIR=$KATTN_RESOURCES_DIR/$DS
    if [ ! -d "$DATA_DIR" ]; then
        log "SKIP $DS — 数据目录不存在"
        continue
    fi

    for SET in "${SETS[@]}"; do
        # 检查 HuggingFace cache（其他模型已跑过，应已存在）
        CACHE_DIR=$SCRIPT_DIR/_cache_dsts/$DS/$SET
        if [ ! -d "$CACHE_DIR" ]; then
            log "Cache $DS/$SET ..."
            python run_Crispr_Regulation.py -m KNET_Crispr -c "$DS" -s $SET \
                -v $VERSION --cache-run 2>&1 | tail -3
        fi

        for MODEL in "${MODELS[@]}"; do
            TOTAL=$((TOTAL + 1))
            RESULT=$RESULT_BASE/$DS/$MODEL/$SET/$VERSION/test_metrics.csv
            if [ -f "$RESULT" ]; then
                log "SKIP $DS/$SET/$MODEL — 已完成"
                SKIP=$((SKIP + 1))
                continue
            fi

            log "RUN  $DS / $SET / $MODEL ..."
            if python run_Crispr_Regulation.py \
                -m "$MODEL" -c "$DS" -s $SET \
                -v $VERSION --max-epochs 200 --patience 20 --max-lr 1e-3 \
                2>&1 | tail -5; then
                log "DONE $DS/$SET/$MODEL"
            else
                log "FAIL $DS/$SET/$MODEL"
                FAIL=$((FAIL + 1))
            fi
        done
    done
done

log "=== 完成 === total=$TOTAL skip=$SKIP fail=$FAIL done=$((TOTAL - SKIP - FAIL))"
