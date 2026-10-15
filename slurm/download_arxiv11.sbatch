#!/bin/bash

#SBATCH --job-name=arxiv-download
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/download-arxiv11-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/download-arxiv11-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc

export HF_HOME=/work/jx198/iris_arxiv/hf_cache
export HF_DATASETS_CACHE=/work/jx198/iris_arxiv/hf_cache/datasets
export PYTHONUNBUFFERED=1

cd /hpc/home/jx198/iris_arxiv

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc/bin/python

echo "Python executable: ${IRIS_PYTHON}"
"${IRIS_PYTHON}" --version
"${IRIS_PYTHON}" -c 'import datasets; print("datasets:", datasets.__version__)'

"${IRIS_PYTHON}" scripts/download_arxiv11.py \
    --output /work/jx198/iris_arxiv/data/raw/arxiv11/data \
    --dataset-config default

echo "Downloaded text files:"
find /work/jx198/iris_arxiv/data/raw/arxiv11/data \
    -type f -name "*.txt" | wc -l

echo "Raw data size:"
du -sh /work/jx198/iris_arxiv/data/raw/arxiv11