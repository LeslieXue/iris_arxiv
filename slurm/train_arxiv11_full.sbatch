#!/bin/bash

#SBATCH --job-name=iris-arxiv-full
#SBATCH --account=biostat
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/train-arxiv11-full-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/train-arxiv11-full-%j.err

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

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p /work/jx198/iris_arxiv/checkpoints
mkdir -p /work/jx198/iris_arxiv/results

cd /hpc/home/jx198/iris_arxiv

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Config: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

nvidia-smi

"${IRIS_PYTHON}" - "${CONFIG_PATH}" <<'PY'
import sys

import duckdb
import lancedb
import torch

from iris.config import get_config

config_path = sys.argv[1]
cfg = get_config(config_path)

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the GPU job")

print("GPU:", torch.cuda.get_device_name(0))

db = lancedb.connect(cfg.database.path)
table = db.open_table("arxiv11")
duckdb.register("chunks", table.to_lance())
summary = duckdb.sql(
    """
    SELECT
        COUNT(*) AS chunks,
        COUNT(DISTINCT id) AS documents,
        COUNT(DISTINCT label) AS classes
    FROM chunks
    """
).fetchone()
print("Database summary (chunks, documents, classes):", summary)
PY

"${IRIS_PYTHON}" -m iris --config "${CONFIG_PATH}"

"${IRIS_PYTHON}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import torch

from iris.config import get_config

cfg = get_config(sys.argv[1])
checkpoint = Path(cfg.trainer.checkpoint)

if not checkpoint.is_file():
    raise FileNotFoundError(f"Training checkpoint was not created: {checkpoint}")

state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
print("Best checkpoint:", checkpoint)
print("Checkpoint size bytes:", checkpoint.stat().st_size)
print("Query shape:", tuple(state_dict["queries"].shape))
print("Classifier shape:", tuple(state_dict["fc.weight"].shape))
PY

echo "Full IRIS arXiv training completed."
