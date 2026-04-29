#!/bin/bash
# run_expC_markov_nomask2.sh
#
# Exp C 补充：Markov 任务 P2P约束✅ + mask❌ 条件（完成2x2消融）
# KNET_nomask = 有 groups (P2P约束) + 无 band mask
#
# 在 192.168.3.17 上运行：
#   cd /rd1/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation
#   bash run_expC_markov_nomask2.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE=/rd1/liut/K-attention/K-attention/Kattn-sim-dev
SRC=$BASE/src
RESOURCES=$BASE/resources
HF_CACHE=$BASE/.hf_cache

MODEL=KNET_nomask
KS=12
NK=64
LR=1e-2
SEEDS=(0 1 2)
CONFIGS=("markov_1_0_5000" "markov_1_0_10000" "markov_1_0_50000")

DONE=0
SKIP=0

for SEED in "${SEEDS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        echo "[RUN ] config=$CONFIG seed=$SEED"
        conda run -n kattn-sim --no-capture-output \
            env KATTN_SRC_DIR="$SRC" \
                KATTN_RESOURCES_DIR="$RESOURCES" \
                HF_DATASETS_CACHE="$HF_CACHE" \
                HF_HOME="$HF_CACHE" \
                HF_DATASETS_OFFLINE=1 \
                PYTHONPATH="$SRC" \
            python "$SCRIPT_DIR/run_bmk.py" \
                --model-type  "$MODEL" \
                --test-config "$CONFIG" \
                --kernel-size "$KS" \
                --num-kernels "$NK" \
                --max-lr      "$LR" \
                --version     "$SEED"
        ((DONE++)) || true
        echo "[DONE] config=$CONFIG seed=$SEED"
    done
done

echo ""
echo "Finished: $DONE runs."
