#!/bin/bash
# Submit Exp D transformer jobs to Slurm cluster
# Run from: Kattn-sim-dev/src/simulation/ on luminary
# Usage: bash submit_expD_cluster.sh

mkdir -p /lustre/grp/gglab/liut/logs

SIZES=(5000 10000 20000 50000 100000)
MODELS=("cnn_transformer_pm" "cnn_transformer")
SEEDS=(0 1 2)

for model in "${MODELS[@]}"; do
    for n in "${SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Submitting: model=$model n=$n seed=$seed"
            sbatch job_expD_cluster.sh "$model" "$n" "$seed"
        done
    done
done

echo "All 30 jobs submitted. Monitor with: squeue -u liut"
