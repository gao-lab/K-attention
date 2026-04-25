#!/bin/bash
#SBATCH --job-name=crispr_cnn_tf
#SBATCH --output=/lustre/grp/gglab/liut/logs/crispr_cnn_tf_%A_%a.log
#SBATCH --error=/lustre/grp/gglab/liut/logs/crispr_cnn_tf_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# array index 由 submit_crispr_cnn_tf.sh 传入 --array=0-N

CODE_BASE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
DATA_BASE=/lustre/grp/gglab/liut/Kattn-sim-dev

SCRIPT_DIR=$CODE_BASE/src/crispr
RESULT_BASE=$DATA_BASE/results/Crispr
TASK_LIST=$SCRIPT_DIR/crispr_task_list.txt

source /lustre/grp/gglab/liut/mambaforge/etc/profile.d/conda.sh
conda activate kattn-sim

# 必须在 conda activate 之后覆盖，否则 conda env 内置路径会覆盖这里的设置
export KATTN_SRC_DIR=$CODE_BASE/src
export KATTN_RESOURCES_DIR=$DATA_BASE/resources
export HF_DATASETS_CACHE=$DATA_BASE/.hf_cache
export HF_HOME=$HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$PYTHONPATH

# 读取第 SLURM_ARRAY_TASK_ID 行（格式：dataset set model）
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")
if [ -z "$LINE" ]; then
    echo "ERROR: empty task at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

DS=$(echo "$LINE" | awk '{print $1}')
SET=$(echo "$LINE" | awk '{print $2}')
MODEL=$(echo "$LINE" | awk '{print $3}')
VERSION=0

echo "[$(date '+%H:%M:%S')] START $DS / $SET / $MODEL (array_id=$SLURM_ARRAY_TASK_ID)"

# 检查结果是否已存在
RESULT=$RESULT_BASE/$DS/$MODEL/$SET/$VERSION/test_metrics.csv
if [ -f "$RESULT" ]; then
    echo "SKIP $DS/$SET/$MODEL — 已完成"
    exit 0
fi

# 检查 HuggingFace cache
CACHE_DIR=$SCRIPT_DIR/_cache_dsts/$DS/$SET
if [ ! -d "$CACHE_DIR" ]; then
    echo "Cache $DS/$SET ..."
    cd "$SCRIPT_DIR"
    python run_Crispr_Regulation.py -m KNET_Crispr -c "$DS" -s $SET \
        -v $VERSION --cache-run 2>&1 | tail -3
fi

cd "$SCRIPT_DIR"
python run_Crispr_Regulation.py \
    -m "$MODEL" -c "$DS" -s $SET \
    -v $VERSION --max-epochs 200 --patience 20 --max-lr 1e-3

echo "[$(date '+%H:%M:%S')] DONE $DS/$SET/$MODEL exit=$?"
