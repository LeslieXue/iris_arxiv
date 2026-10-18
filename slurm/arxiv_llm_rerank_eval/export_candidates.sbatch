#!/bin/bash

#SBATCH --job-name=iris-export-candidates
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/export-candidates-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/export-candidates-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
REPO_ROOT=/hpc/home/jx198/iris_arxiv
EXPERIMENT_CONFIG=experiments/arxiv_llm_rerank_eval/config.toml
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v1

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"
echo "Output directory: ${OUTPUT_DIR}"

"${IRIS_PYTHON}" --version

if [[ ! -f /work/jx198/iris_arxiv/checkpoints/arxiv11-best.pt ]]; then
    echo "Missing checkpoint: /work/jx198/iris_arxiv/checkpoints/arxiv11-best.pt" >&2
    exit 1
fi

if [[ ! -d /work/jx198/iris_arxiv/data/db ]]; then
    echo "Missing database directory: /work/jx198/iris_arxiv/data/db" >&2
    exit 1
fi

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/export_candidates.py \
    --config "${EXPERIMENT_CONFIG}"

echo
echo "Generated artifacts:"
ls -lh "${OUTPUT_DIR}"

echo
echo "Candidate export job completed."