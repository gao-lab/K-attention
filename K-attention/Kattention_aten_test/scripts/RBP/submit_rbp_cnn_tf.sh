#!/bin/bash
# submit_rbp_cnn_tf.sh
# 提交 cnn_transformer × 172 RBP × seed=666 的 Slurm array job
#
# 使用方式（在集群 luminary 上，于脚本所在目录执行）：
#   cd /lustre/grp/gglab/liut/Kattention_aten_test/scripts/RBP
#   bash submit_rbp_cnn_tf.sh
#
# 前置要求：
#   1. conda activate pytorch（或含 torch/h5py/einops/sklearn 的环境）
#   2. 若缺依赖：pip install h5py einops scikit-learn

CODE_BASE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattention_aten_test
DATA_BASE=/lustre/grp/gglab/liut/Kattention_aten_test
HDF5_ROOT=$DATA_BASE/external/RBP/HDF5
SCRIPT_DIR=$CODE_BASE/scripts/RBP
LOGDIR=/lustre/grp/gglab/liut/logs
RBP_LIST_FILE=$SCRIPT_DIR/rbp_list.txt

mkdir -p "$LOGDIR"

# ── Step 1: 生成 RBP 列表（仅保留 HDF5 完整的条目）──
echo "生成 RBP 列表..."
> "$RBP_LIST_FILE"
for RBP_DIR in "$HDF5_ROOT"/*/; do
    RBP=$(basename "$RBP_DIR")
    if [ -f "$RBP_DIR/train.hdf5" ] && [ -f "$RBP_DIR/test.hdf5" ]; then
        echo "$RBP" >> "$RBP_LIST_FILE"
    else
        echo "  SKIP $RBP (HDF5 不完整)"
    fi
done

N=$(wc -l < "$RBP_LIST_FILE")
echo "有效 RBP 数：$N"
if [ "$N" -eq 0 ]; then
    echo "ERROR: 无有效 RBP，退出"
    exit 1
fi

ARRAY_END=$((N - 1))

# ── Step 2: 轮流投到 gpu2 / gpu32（各一半）──
HALF=$((N / 2))
echo ""
echo "提交 Slurm array job (总 $N 个 RBP)..."

sbatch --partition=gpu2 \
       --array=0-$((HALF - 1)) \
       "$SCRIPT_DIR/job_rbp_cnn_tf.sh"

sbatch --partition=gpu32 \
       --array=${HALF}-${ARRAY_END} \
       "$SCRIPT_DIR/job_rbp_cnn_tf.sh"

echo ""
echo "已提交。查看状态：squeue -u liut"
echo "日志目录：$LOGDIR"
echo "结果目录：$DATA_BASE/result/RBP/HDF5/<RBP>/cnn_transformer/"
