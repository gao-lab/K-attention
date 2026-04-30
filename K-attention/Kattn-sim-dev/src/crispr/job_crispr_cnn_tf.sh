#!/bin/bash
#SBATCH --job-name=crispr_cnn_tf
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# NOTE: --output is set dynamically by submit_crispr_cnn_tf.sh

# Positional args: DS MODEL SET VERSION
DS=$1
MODEL=$2
SET=$3
VERSION=${4:-0}

CODE_BASE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
DATA_BASE=/lustre/grp/gglab/liut/Kattn-sim-dev

SCRIPT_DIR=$CODE_BASE/src/crispr
RESULT_BASE=$DATA_BASE/results/Crispr

source /lustre/grp/gglab/liut/mambaforge/etc/profile.d/conda.sh
conda activate kattn-sim

# 必须在 conda activate 之后覆盖，否则 conda env 内置路径会覆盖这里的设置
export KATTN_SRC_DIR=$CODE_BASE/src
export KATTN_RESOURCES_DIR=$DATA_BASE/resources
export HF_DATASETS_CACHE=$DATA_BASE/.hf_cache
export HF_HOME=$HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$PYTHONPATH

echo "[$(date '+%H:%M:%S')] START $DS / $SET / $MODEL"

# 检查结果是否已存在
RESULT=$RESULT_BASE/$DS/$MODEL/$SET/$VERSION/test_metrics.csv
if [ -f "$RESULT" ]; then
    echo "SKIP $DS/$SET/$MODEL — 已完成"
    exit 0
fi

cd "$SCRIPT_DIR"

# 检查 HuggingFace cache
CACHE_DIR=$SCRIPT_DIR/_cache_dsts/$DS/$SET
if [ ! -d "$CACHE_DIR" ]; then
    echo "Cache $DS/$SET ..."
    python run_Crispr_Regulation.py -m KNET_Crispr -c "$DS" -s $SET \
        -v $VERSION --cache-run 2>&1 | tail -3
fi

python run_Crispr_Regulation.py \
    -m "$MODEL" -c "$DS" -s $SET \
    -v $VERSION --max-epochs 200 --patience 20 --max-lr 1e-3

echo "[$(date '+%H:%M:%S')] DONE $DS/$SET/$MODEL exit=$?"
