#!/bin/bash

#SBATCH --job-name=iris-one-third-summary
#SBATCH --account=engelhardlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/one-third-summary-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/one-third-summary-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
cd /hpc/home/jx198/iris_arxiv

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/summarize_results.py \
    --experiment-config \
    experiments/arxiv_llm_teacher_loss/config.one_third.toml

echo "One-third teacher-loss result summary completed."
