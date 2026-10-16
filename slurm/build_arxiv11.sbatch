#!/bin/bash

#SBATCH --job-name=iris-arxiv-build
#SBATCH --account=biostat
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/build-arxiv11-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/build-arxiv11-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02
module load CUDA/12.4

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

export HF_HOME=/work/jx198/iris_arxiv/hf_cache
export HF_DATASETS_CACHE=/work/jx198/iris_arxiv/hf_cache/datasets
export PIP_CACHE_DIR=/work/jx198/iris_arxiv/pip-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
CONFIG_PATH="${1:-config.arxiv.dcc.build.toml}"

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p /work/jx198/iris_arxiv/data/processed
mkdir -p /work/jx198/iris_arxiv/data/db

cd /hpc/home/jx198/iris_arxiv

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Config: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

nvidia-smi

"${IRIS_PYTHON}" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the GPU job")

print("GPU:", torch.cuda.get_device_name(0))
PY

"${IRIS_PYTHON}" -m iris --config "${CONFIG_PATH}"

echo "Processed dataset size:"
du -sh /work/jx198/iris_arxiv/data/processed || true

echo "Database size:"
du -sh /work/jx198/iris_arxiv/data/db || true

echo "IRIS arXiv build completed."
