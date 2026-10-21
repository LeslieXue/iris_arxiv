#!/bin/bash

#SBATCH --job-name=iris-teacher-summary
#SBATCH --account=engelhardlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/teacher-summary-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/teacher-summary-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
cd /hpc/home/jx198/iris_arxiv

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/summarize_results.py \
    --experiment-config experiments/arxiv_llm_teacher_loss/config.toml

RESULT_DIR=/work/jx198/iris_arxiv/results/teacher_loss_v1

export MPLCONFIGDIR=/work/jx198/iris_arxiv/matplotlib-cache
mkdir -p "${MPLCONFIGDIR}"

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/plot_results.py \
    --comparison "${RESULT_DIR}/comparison.csv" \
    --output-dir "${RESULT_DIR}"

echo "Teacher-loss result summary and plotting completed."
