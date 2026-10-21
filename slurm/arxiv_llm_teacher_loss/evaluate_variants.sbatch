#!/bin/bash

#SBATCH --job-name=iris-teacher-eval
#SBATCH --account=engelhardlab
#SBATCH --partition=engelhardlab-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/work/jx198/iris_arxiv/logs/teacher-eval-%A_%a.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/teacher-eval-%A_%a.err

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
EXPERIMENT_CONFIG=experiments/arxiv_llm_teacher_loss/config.toml
IRIS_CONFIG=config.arxiv.dcc.train.toml
VARIANTS=(baseline teacher_w001 teacher_w005 teacher_w010)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
CHECKPOINT=/work/jx198/iris_arxiv/checkpoints/teacher_loss_v1/${VARIANT}.pt
RESULT_DIR=/work/jx198/iris_arxiv/results/teacher_loss_v1

mkdir -p /work/jx198/iris_arxiv/logs "${RESULT_DIR}"
cd /hpc/home/jx198/iris_arxiv

echo "Host: $(hostname)"
echo "Variant: ${VARIANT}"
nvidia-smi

"${IRIS_PYTHON}" scripts/evaluate_arxiv_checkpoint.py \
    --config "${IRIS_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${RESULT_DIR}/${VARIANT}.test_metrics.json"

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/evaluate_alignment.py \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --iris-config "${IRIS_CONFIG}" \
    --variant "${VARIANT}" \
    --output "${RESULT_DIR}/${VARIANT}.teacher_alignment.json"

echo "Teacher-loss evaluation variant completed: ${VARIANT}"
