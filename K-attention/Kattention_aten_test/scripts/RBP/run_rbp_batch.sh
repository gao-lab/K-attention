#!/bin/bash
# Phase 2: RBP 全量 CNN-TF 参数匹配版批量实验
# 172 RBPs × cnn_transformer_pm × 3 seeds × adamw (~130 GPU-hours)
# 已完成的 run 自动跳过（基于 Report_*.pkl 存在判断）

BASE=/rd1/liut/K-attention/K-attention
HDF5_ROOT=$BASE/Kattention_aten_test/external/RBP/HDF5
SCRIPT_DIR=$BASE/Kattention_aten_test/scripts/RBP
RESULT_ROOT=$BASE/Kattention_aten_test/result/RBP/HDF5

MODEL=cnn_transformer_pm
OPT=adamw
KERNEL_LEN=12
KERNEL_NUM=64
SEEDS=(666 1996 212)

cd "$SCRIPT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TOTAL=0
SKIP=0
FAIL=0

for RBP_DIR in "$HDF5_ROOT"/*/; do
    RBP=$(basename "$RBP_DIR")
    # 检查 train/test.hdf5 是否存在
    if [ ! -f "$RBP_DIR/train.hdf5" ] || [ ! -f "$RBP_DIR/test.hdf5" ]; then
        log "SKIP $RBP — HDF5 文件不完整"
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        RESULT_PKL=$RESULT_ROOT/$RBP/$MODEL/Report_KernelNum-${KERNEL_NUM}kernel_size-${KERNEL_LEN}_seed-${SEED}_opkl-${OPT}.pkl
        if [ -f "$RESULT_PKL" ]; then
            log "SKIP $RBP seed=$SEED — 已完成"
            SKIP=$((SKIP + 1))
            continue
        fi

        log "RUN  $RBP seed=$SEED ..."
        if python main.py "$RBP_DIR" $KERNEL_LEN $KERNEL_NUM $SEED $MODEL $OPT \
            2>&1 | tee -a /tmp/rbp_batch.log | tail -3; then
            log "DONE $RBP seed=$SEED"
        else
            log "FAIL $RBP seed=$SEED"
            FAIL=$((FAIL + 1))
        fi
    done
done

log "=== 完成 === total=$TOTAL skip=$SKIP fail=$FAIL done=$((TOTAL - SKIP - FAIL))"
