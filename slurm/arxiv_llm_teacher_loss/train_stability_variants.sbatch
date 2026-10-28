#!/bin/bash

#SBATCH --job-name=iris-stability-train
#SBATCH --account=engelhardlab
#SBATCH --partition=engelhardlab-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-49%4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/work/jx198/iris_arxiv/logs/stability-train-%A_%a.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/stability-train-%A_%a.err

set -eo pipefail

module purge
module load Anaconda3/2024.02
module load CUDA/12.4

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

export HF_HOME=/work/jx198/iris_arxiv/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
EXPERIMENT_CONFIG=experiments/arxiv_llm_teacher_loss/config.stability.toml
IRIS_CONFIG=config.arxiv.dcc.train.stability.toml
VARIANTS=(baseline teacher_w0025 teacher_w005 teacher_w010 teacher_w020 teacher_w040 teacher_w080 teacher_w160 teacher_w320 teacher_w640)
SEEDS=(42 123 2026 3407 10007)

NUM_SEEDS=${#SEEDS[@]}
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
VARIANT="${VARIANTS[${VARIANT_INDEX}]}"
TRAINING_SEED="${SEEDS[${SEED_INDEX}]}"

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p /work/jx198/iris_arxiv/checkpoints/teacher_loss_v2_stability
mkdir -p /work/jx198/iris_arxiv/results/teacher_loss_v2_stability

cd /hpc/home/jx198/iris_arxiv

echo "Host: $(hostname)"
echo "Variant: ${VARIANT}"
echo "Training seed: ${TRAINING_SEED}"
echo "Fixed split seed: 42"
echo "SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/run_training.py \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --iris-config "${IRIS_CONFIG}" \
    --variant "${VARIANT}" \
    --training-seed "${TRAINING_SEED}"

echo "Completed: ${VARIANT}, seed=${TRAINING_SEED}"
