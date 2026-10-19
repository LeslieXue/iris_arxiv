#!/bin/bash

#SBATCH --job-name=export-all-teacher-chunks
#SBATCH --account=engelhardlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/export-all-teacher-chunks-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/export-all-teacher-chunks-%j.err

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

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/export_all_chunks.py \
    --config "${CONFIG}"

echo "Exported artifacts:"
ls -lh /work/jx198/iris_arxiv/experiments/arxiv_llm_teacher_loss/v1

echo "Full teacher-input export completed."
