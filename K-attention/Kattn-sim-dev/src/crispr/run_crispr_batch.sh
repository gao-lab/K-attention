#!/bin/bash
# Phase 1: CRISPR 全数据集批量实验
# 11 datasets × 4 models × set0 × seed 0 (~30 GPU-hours)
# 已完成的 run 自动跳过

set -e

# ---- 环境变量 ----
BASE=/rd1/liut/K-attention/K-attention
export KATTN_SRC_DIR=$BASE/Kattn-sim-dev/src
export KATTN_RESOURCES_DIR=$BASE/Kattn-sim-dev/resources
export HF_DATASETS_CACHE=$BASE/Kattn-sim-dev/.hf_cache
export HF_HOME=$HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$PYTHONPATH

SCRIPT_DIR=$BASE/Kattn-sim-dev/src/crispr
RESULT_BASE=$BASE/Kattn-sim-dev/results/Crispr

cd "$SCRIPT_DIR"

# ---- 数据集与模型列表 ----
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

MODELS=(KNET_Crispr cnn_transformer cnn_transformer_pm CRISPRon_base)

SET=set0
VERSION=0
TOTAL=0
SKIP=0
FAIL=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for DS in "${DATASETS[@]}"; do
    # 检查数据目录是否存在
    DATA_DIR=$KATTN_RESOURCES_DIR/$DS
    if [ ! -d "$DATA_DIR" ]; then
        log "SKIP $DS — 数据目录不存在: $DATA_DIR"
        continue
    fi

    # 先预缓存（只需一次；若 cache 已存在自动跳过）
    CACHE_FLAG=$HF_DATASETS_CACHE/cached_$DS
    if [ ! -f "$CACHE_FLAG" ]; then
        log "Cache $DS ..."
        python run_Crispr_Regulation.py -m KNET_Crispr -c "$DS" -s $SET \
            -v $VERSION --cache-run 2>&1 | tail -3
        touch "$CACHE_FLAG"
    fi

    for MODEL in "${MODELS[@]}"; do
        TOTAL=$((TOTAL + 1))
        RESULT=$RESULT_BASE/$DS/$MODEL/$SET/$VERSION/test_metrics.csv
        if [ -f "$RESULT" ]; then
            log "SKIP $DS/$MODEL — 已完成"
            SKIP=$((SKIP + 1))
            continue
        fi

        log "RUN  $DS / $MODEL ..."
        if python run_Crispr_Regulation.py \
            -m "$MODEL" -c "$DS" -s $SET \
            -v $VERSION --max-epochs 200 --patience 20 --max-lr 1e-3 \
            2>&1 | tee -a /tmp/crispr_batch.log | tail -5; then
            log "DONE $DS/$MODEL"
        else
            log "FAIL $DS/$MODEL"
            FAIL=$((FAIL + 1))
        fi
    done
done

log "=== 完成 === total=$TOTAL skip=$SKIP fail=$FAIL done=$((TOTAL - SKIP - FAIL))"
