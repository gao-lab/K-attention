#!/bin/bash
# Source this to fix all environment paths for running outside Docker
BASE="/home/mnt/liut/K-attention/K-attention/Kattn-sim-dev"
export KATTN_BASE_DIR="$BASE"
export KATTN_SRC_DIR="$BASE/src"
export KATTN_RESOURCES_DIR="$BASE/resources"
export KATTN_RESULTS_DIR="$BASE/results"
export PYTHONPATH="$BASE/src:$PYTHONPATH"
export HF_DATASETS_CACHE="$BASE/.hf_cache"
export HF_DATASETS_OFFLINE=1
