#!/bin/bash
# run_expC_markov_nomask.sh
#
# Exp C 补充：Markov 任务真正无约束基线
# KNET_uncons_nomask = 无 groups + 无 band mask（完全无约束）
#
# 在 192.168.3.17 上运行：
#   cd /rd1/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation
#   bash run_expC_markov_nomask.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE=/rd1/liut/K-attention/K-attention/Kattn-sim-dev
SRC=$BASE/src
RESOURCES=$BASE/resources
HF_CACHE=$BASE/.hf_cache

MODEL=KNET_uncons_nomask
KS=12
NK=64
LR=1e-2
SEEDS=(0 1 2)

# n=2k: markov_1_0_5000 subsampled to 2000
# n=5k: markov_1_0_5000 full (-1)
# n=10k: markov_1_0_10000 full (-1)
CONFIGS=("markov_1_0_5000" "markov_1_0_5000" "markov_1_0_10000")
SS_LIST=(2000 -1 -1)
LABELS=(2k 5k 10k)

DONE=0
SKIP=0

for SEED in "${SEEDS[@]}"; do
    for IDX in 0 1 2; do
        CONFIG=${CONFIGS[$IDX]}
        SS=${SS_LIST[$IDX]}
        LABEL=${LABELS[$IDX]}

        # 结果路径
        if [ "$SS" = "-1" ]; then
            RESULT=$BASE/results/${CONFIG}/${MODEL}/ks${KS}_nk${NK}_ss-1/v${SEED}/test_metrics.csv
        else
            RESULT=$BASE/results/${CONFIG}/${MODEL}/ks${KS}_nk${NK}_ss${SS}/v${SEED}/test_metrics.csv
        fi

        if [ -f "$RESULT" ]; then
            echo "[SKIP] n=${LABEL} seed=${SEED}"
            ((SKIP++)) || true
            continue
        fi

        echo "[RUN ] n=${LABEL} seed=${SEED}  (config=${CONFIG} ss=${SS})"

        SS_ARG=""
        [ "$SS" != "-1" ] && SS_ARG="--sample-size $SS"

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
                --version     "$SEED" \
                $SS_ARG

        ((DONE++)) || true
        echo "[DONE] n=${LABEL} seed=${SEED}"
    done
done

echo ""
echo "Finished: $DONE runs, $SKIP skipped."
