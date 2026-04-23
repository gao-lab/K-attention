#!/bin/bash
#SBATCH --job-name=expD-tf
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/grp/gglab/liut/logs/expD_%A_%a.out

# Usage: called by submit_expD_cluster.sh
# Args: MODEL N SEED
MODEL=$1
N=$2
SEED=$3

SIMDIR=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev/src/simulation
cd "$SIMDIR" || exit 1

# Fix environment paths for lustre
export KATTN_SRC_DIR=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev/src
export KATTN_RESOURCES_DIR=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev/resources
export HF_DATASETS_CACHE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev/.hf_cache
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$PYTHONPATH

conda run -n kattn-sim python run_bmk.py \
    --model-type "$MODEL" \
    --test-config random_rand \
    --sample-size "$N" \
    --max-epochs 500 \
    --patience 20 \
    --max-lr 1e-5 \
    --batch-size 128 \
    --auroc-threshold 0.99 \
    --version "$SEED"
