#!/bin/bash
# Submit Exp D transformer jobs to Slurm cluster
# Usage: conda activate kattn-sim && bash submit_expD_cluster.sh

LUSTRE=/lustre/grp/gglab/liut/K-attention/K-attention/Kattn-sim-dev
SIMDIR=$LUSTRE/src/simulation
mkdir -p /lustre/grp/gglab/liut/logs

# Override paths so new kattn is used (not old editable install)
export KATTN_SRC_DIR=$LUSTRE/src
export KATTN_RESOURCES_DIR=$LUSTRE/resources
export HF_DATASETS_CACHE=$LUSTRE/.hf_cache
export PYTHONPATH=$KATTN_SRC_DIR:$(echo $PYTHONPATH | tr ':' '\n' | grep -v 'Kattn-sim-dev/src' | tr '\n' ':')

cd "$SIMDIR" || exit 1

# Pre-cache random_rand dataset
echo "Pre-caching random_rand..."
python run_bmk.py --model-type cnn_transformer_pm --test-config random_rand --cache-run 2>&1 | tail -3
echo "Cache done."

# Submit 30 jobs
for model in cnn_transformer_pm cnn_transformer; do
    for n in 5000 10000 20000 50000 100000; do
        for seed in 0 1 2; do
            echo "Submitting: model=$model n=$n seed=$seed"
            sbatch job_expD_cluster.sh "$model" "$n" "$seed"
        done
    done
done

echo "All 30 jobs submitted."
squeue -u liut | head -20
