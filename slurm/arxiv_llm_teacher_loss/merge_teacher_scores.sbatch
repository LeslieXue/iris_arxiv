#!/bin/bash

#SBATCH --job-name=merge-teacher-scores
#SBATCH --account=engelhardlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/merge-teacher-scores-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/merge-teacher-scores-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
REPO_ROOT=/hpc/home/jx198/iris_arxiv
CONFIG=experiments/arxiv_llm_teacher_loss/config.toml

mkdir -p /work/jx198/iris_arxiv/logs
cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Config: ${CONFIG}"

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/merge_teacher_scores.py \
    --config "${CONFIG}"

ls -lh \
    /work/jx198/iris_arxiv/experiments/arxiv_llm_teacher_loss/v1/teacher_scores.parquet \
    /work/jx198/iris_arxiv/experiments/arxiv_llm_teacher_loss/v1/teacher_scores.metadata.json

echo "Teacher-score merge completed."
