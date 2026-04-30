#!/bin/bash
#SBATCH --job-name=markov-tasks
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
# NOTE: --output, --time, --mem can be overridden by submit_markov_tasks.sh via sbatch CLI

set -euo pipefail

# Positional args: MODEL TEST_CONFIG SAMPLE_SIZE SEED LR BATCH
MODEL=${1:?Usage: $0 MODEL TEST_CONFIG SAMPLE_SIZE SEED LR BATCH}
TEST_CONFIG=${2:?}
SAMPLE_SIZE=${3:?}
SEED=${4:?}
LR=${5:?}
BATCH=${6:?}

echo "============================================"
echo "[markov-tasks] model=$MODEL config=$TEST_CONFIG n=$SAMPLE_SIZE seed=$SEED lr=$LR batch=$BATCH"
echo "[markov-tasks] start=$(date -Iseconds)"
echo "[markov-tasks] host=$(hostname)"
echo "============================================"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate kattn-sim 2>&1 | tail -1 || {
    echo "FATAL: conda activate kattn-sim failed"
    exit 1
}

LUSTRE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
export KATTN_SRC_DIR=$LUSTRE/src
export KATTN_RESOURCES_DIR=$LUSTRE/resources
export HF_DATASETS_CACHE=$LUSTRE/.hf_cache
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$KATTN_SRC_DIR:$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'Kattn-sim-dev/src' | tr '\n' ':')

cd "$KATTN_SRC_DIR/simulation" || exit 1

python run_bmk.py \
    --model-type      "$MODEL" \
    --test-config     "$TEST_CONFIG" \
    --sample-size     "$SAMPLE_SIZE" \
    --max-epochs      500 \
    --patience        20 \
    --max-lr          "$LR" \
    --batch-size      "$BATCH" \
    --auroc-threshold 0.99 \
    --version         "$SEED"

RC=$?
if [ $RC -eq 0 ]; then
    echo "[markov-tasks] SUCCESS model=$MODEL config=$TEST_CONFIG n=$SAMPLE_SIZE seed=$SEED done=$(date -Iseconds)"
else
    echo "[markov-tasks] FAILED model=$MODEL config=$TEST_CONFIG n=$SAMPLE_SIZE seed=$SEED exit=$RC time=$(date -Iseconds)"
fi
exit $RC
