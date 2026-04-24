#!/bin/bash
#SBATCH --job-name=expB-ext
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/grp/gglab/liut/logs/expB_ext_%j.out

# Args: MODEL TEST_CONFIG SAMPLE_SIZE SEED LR BATCH
MODEL=$1
TEST_CONFIG=$2
SAMPLE_SIZE=$3
SEED=$4
LR=$5
BATCH=$6

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate kattn-sim

LUSTRE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
export KATTN_SRC_DIR=$LUSTRE/src
export KATTN_RESOURCES_DIR=$LUSTRE/resources
export HF_DATASETS_CACHE=$LUSTRE/.hf_cache
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$(echo $PYTHONPATH | tr ':' '\n' | grep -v 'Kattn-sim-dev/src' | tr '\n' ':')

cd "$KATTN_SRC_DIR/simulation" || exit 1

echo "[ExpB-ext] model=$MODEL config=$TEST_CONFIG n=$SAMPLE_SIZE seed=$SEED lr=$LR start=$(date)"
python run_bmk.py \
    --model-type   "$MODEL" \
    --test-config  "$TEST_CONFIG" \
    --sample-size  "$SAMPLE_SIZE" \
    --max-epochs   500 \
    --patience     20 \
    --max-lr       "$LR" \
    --batch-size   "$BATCH" \
    --auroc-threshold 0.99 \
    --version      "$SEED"
echo "[ExpB-ext] model=$MODEL config=$TEST_CONFIG n=$SAMPLE_SIZE seed=$SEED done=$(date)"
