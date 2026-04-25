#!/bin/bash
#SBATCH --job-name=rbp_cnn_tf
#SBATCH --output=/lustre/grp/gglab/liut/logs/rbp_cnn_tf_%A_%a.log
#SBATCH --error=/lustre/grp/gglab/liut/logs/rbp_cnn_tf_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# array index 由 submit_rbp_cnn_tf.sh 传入 --array=0-N

CODE_BASE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattention_aten_test
DATA_BASE=/lustre/grp/gglab/liut/Kattention_aten_test
HDF5_ROOT=$DATA_BASE/external/RBP/HDF5
RESULT_ROOT=$DATA_BASE/result/RBP/HDF5
SCRIPT_DIR=$CODE_BASE/scripts/RBP
RBP_LIST_FILE=$SCRIPT_DIR/rbp_list.txt

MODEL=cnn_transformer
KERNEL_LEN=12
KERNEL_NUM=64
SEED=666
OPT=adamw

# 读取第 SLURM_ARRAY_TASK_ID 行对应的 RBP 名
RBP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RBP_LIST_FILE")
if [ -z "$RBP" ]; then
    echo "ERROR: empty RBP at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

RBP_DIR=$HDF5_ROOT/$RBP

# 检查数据完整性
if [ ! -f "$RBP_DIR/train.hdf5" ] || [ ! -f "$RBP_DIR/test.hdf5" ]; then
    echo "SKIP $RBP — HDF5 不完整"
    exit 0
fi

# 检查结果是否已存在
RESULT_PKL=$RESULT_ROOT/$RBP/$MODEL/Report_KernelNum-${KERNEL_NUM}kernel_size-${KERNEL_LEN}_seed-${SEED}_opkl-${OPT}.pkl
if [ -f "$RESULT_PKL" ]; then
    echo "SKIP $RBP — 已完成"
    exit 0
fi

mkdir -p "$RESULT_ROOT/$RBP/$MODEL"

echo "[$(date '+%H:%M:%S')] START $RBP (array_id=$SLURM_ARRAY_TASK_ID)"

cd "$SCRIPT_DIR"
conda activate pytorch

python -u main.py \
    "$RBP_DIR" \
    $KERNEL_LEN \
    $KERNEL_NUM \
    $SEED \
    $MODEL \
    $OPT

echo "[$(date '+%H:%M:%S')] DONE $RBP exit=$?"
