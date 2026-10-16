#!/bin/bash

#SBATCH --job-name=iris-arxiv-eval
#SBATCH --account=biostat
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/evaluate-arxiv11-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/evaluate-arxiv11-%j.err

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
CONFIG_PATH="${1:-config.arxiv.dcc.train.toml}"
OUTPUT_PATH="${2:-/work/jx198/iris_arxiv/results/arxiv11-test-metrics.json}"

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p /work/jx198/iris_arxiv/results

cd /hpc/home/jx198/iris_arxiv

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_PATH}"

nvidia-smi

"${IRIS_PYTHON}" scripts/evaluate_arxiv_checkpoint.py \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_PATH}"

echo "IRIS arXiv test evaluation completed."
