#!/bin/bash
#SBATCH --job-name=expD-tf
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/grp/gglab/liut/logs/expD_%j.out

MODEL=$1
N=$2
SEED=$3

# Activate conda environment (must come before env var overrides)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate kattn-sim

# Override paths AFTER activation so our values take precedence
LUSTRE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
export KATTN_SRC_DIR=$LUSTRE/src
export KATTN_RESOURCES_DIR=$LUSTRE/resources
export HF_DATASETS_CACHE=$LUSTRE/.hf_cache
export HF_DATASETS_OFFLINE=1
# Prepend our src dir to ensure new kattn overrides any old editable install
export PYTHONPATH=$KATTN_SRC_DIR:$(echo $PYTHONPATH | tr ':' '\n' | grep -v 'Kattn-sim-dev/src' | tr '\n' ':')

cd "$KATTN_SRC_DIR/simulation" || exit 1

echo "[ExpD] model=$MODEL n=$N seed=$SEED start=$(date)"
python run_bmk.py \
    --model-type "$MODEL" \
    --test-config random_rand \
    --sample-size "$N" \
    --max-epochs 500 \
    --patience 20 \
    --max-lr 1e-5 \
    --batch-size 128 \
    --auroc-threshold 0.99 \
    --version "$SEED"
echo "[ExpD] model=$MODEL n=$N seed=$SEED done=$(date)"
