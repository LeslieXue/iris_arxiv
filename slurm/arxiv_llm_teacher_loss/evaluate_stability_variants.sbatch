#!/bin/bash

#SBATCH --job-name=iris-stability-eval
#SBATCH --account=engelhardlab
#SBATCH --partition=engelhardlab-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-49%4
#SBATCH --output=/work/jx198/iris_arxiv/logs/stability-eval-%A_%a.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/stability-eval-%A_%a.err

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
RUN_NAME="${VARIANT}.seed-${TRAINING_SEED}"

CHECKPOINT=/work/jx198/iris_arxiv/checkpoints/teacher_loss_v2_stability/${RUN_NAME}.pt
RESULT_DIR=/work/jx198/iris_arxiv/results/teacher_loss_v2_stability

mkdir -p /work/jx198/iris_arxiv/logs "${RESULT_DIR}"
cd /hpc/home/jx198/iris_arxiv

echo "Host: $(hostname)"
echo "Variant: ${VARIANT}"
echo "Training seed: ${TRAINING_SEED}"
nvidia-smi

"${IRIS_PYTHON}" scripts/evaluate_arxiv_checkpoint.py \
    --config "${IRIS_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${RESULT_DIR}/${RUN_NAME}.test_metrics.json"

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/evaluate_alignment.py \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --iris-config "${IRIS_CONFIG}" \
    --variant "${VARIANT}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${RESULT_DIR}/${RUN_NAME}.teacher_alignment.json"

echo "Evaluated: ${VARIANT}, seed=${TRAINING_SEED}"
