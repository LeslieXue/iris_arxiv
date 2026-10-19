#!/bin/bash

#SBATCH --job-name=qwen-score-one-third
#SBATCH --account=engelhardlab
#SBATCH --partition=engelhardlab-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%7
#SBATCH --output=/work/jx198/iris_arxiv/logs/one-third-score-%A_%a.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/one-third-score-%A_%a.err

set -eo pipefail

module purge
module load Anaconda3/2024.02
module load CUDA/12.4
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

export HF_HOME=/work/jx198/iris_arxiv/hf_cache
export HF_DATASETS_CACHE=/work/jx198/iris_arxiv/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
REPO_ROOT=/hpc/home/jx198/iris_arxiv
CONFIG=experiments/arxiv_llm_teacher_loss/config.one_third.toml
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_teacher_loss/v1_one_third
NUM_SHARDS=7
WIDTH=3
SHARD_ID=$(printf "%0${WIDTH}d" "${SLURM_ARRAY_TASK_ID}")
SHARD_TOTAL=$(printf "%0${WIDTH}d" "${NUM_SHARDS}")
INPUT="${OUTPUT_DIR}/teacher_inputs.shard-${SHARD_ID}-of-${SHARD_TOTAL}.jsonl"
OUTPUT="${OUTPUT_DIR}/qwen_scores.target_evidence_v2.shard-${SHARD_ID}-of-${SHARD_TOTAL}.jsonl"

mkdir -p /work/jx198/iris_arxiv/logs
cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
echo "Input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Configured batch size: 12"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

if [[ ! -f "${INPUT}" ]]; then
    echo "Missing teacher-input shard: ${INPUT}" >&2
    exit 1
fi

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/score_target_candidates.py \
    --config "${CONFIG}" \
    --input "${INPUT}" \
    --output "${OUTPUT}"

echo "One-third Qwen scoring completed: ${SLURM_ARRAY_TASK_ID}"
