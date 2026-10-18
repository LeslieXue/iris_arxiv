#!/bin/bash

#SBATCH --job-name=qwen-top32-pilot
#SBATCH --account=engelhardlab
#SBATCH --partition=engelhardlab-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/qwen-top32-pilot-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/qwen-top32-pilot-%j.err

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
EXPERIMENT_CONFIG=experiments/arxiv_llm_rerank_eval/config.target.toml
PILOT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v2_target_conditioned/pilot_top32

mkdir -p /work/jx198/iris_arxiv/logs
cd "${REPO_ROOT}"

PROMPT_VERSION=$(
    "${IRIS_PYTHON}" -c \
    'import tomllib; print(tomllib.load(open("experiments/arxiv_llm_rerank_eval/config.target.toml", "rb"))["teacher"]["prompt_version"])'
)

export PROMPT_VERSION
export PILOT_INPUT="${PILOT_DIR}/target_teacher_inputs.pilot.jsonl"
export PILOT_OUTPUT="${PILOT_DIR}/qwen_scores.${PROMPT_VERSION}.pilot.jsonl"
export PILOT_MANIFEST="${PILOT_DIR}/pilot_manifest.parquet"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"
echo "Prompt version: ${PROMPT_VERSION}"
echo "Pilot input: ${PILOT_INPUT}"
echo "Pilot output: ${PILOT_OUTPUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

"${IRIS_PYTHON}" --version
nvidia-smi

for path in "${PILOT_INPUT}" "${PILOT_MANIFEST}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing top-32 pilot artifact: ${path}" >&2
        exit 1
    fi
done

"${IRIS_PYTHON}" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the GPU job")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GB:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
)
PY

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/score_target_candidates.py \
    --config "${EXPERIMENT_CONFIG}" \
    --input "${PILOT_INPUT}" \
    --output "${PILOT_OUTPUT}" \
    --overwrite

"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

output_path = Path(os.environ["PILOT_OUTPUT"])
input_path = Path(os.environ["PILOT_INPUT"])
manifest_path = Path(os.environ["PILOT_MANIFEST"])
prompt_version = os.environ["PROMPT_VERSION"]

manifest = pl.read_parquet(manifest_path)
with input_path.open("r", encoding="utf-8") as handle:
    inputs = [json.loads(line) for line in handle if line.strip()]
with output_path.open("r", encoding="utf-8") as handle:
    scores = [json.loads(line) for line in handle if line.strip()]

error_path = output_path.with_suffix(".errors.jsonl")
metadata_path = output_path.with_suffix(".metadata.json")

def key(row):
    return (row["document_id"], row["query_id"], row["chunk_id"])

manifest_keys = {key(row) for row in manifest.iter_rows(named=True)}
input_keys = {key(row) for row in inputs}
score_keys = {key(row) for row in scores}

assert manifest.height == 32
assert len(inputs) == 32
assert len(scores) == 32
assert len(manifest_keys) == len(input_keys) == len(score_keys) == 32
assert manifest_keys == input_keys == score_keys
assert {row["target_label"] for row in scores} == {"cs.AI"}

score_fields = (
    "target_evidence",
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
)
for score in scores:
    for field in score_fields:
        value = score[field]
        assert isinstance(value, int), (field, value)
        assert 0 <= value <= 4, (field, value)
    assert score["brief_reason"].strip()
    assert len(score["brief_reason"].split()) <= 40

assert error_path.exists() and error_path.stat().st_size == 0
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
assert metadata["prompt_version"] == prompt_version
assert metadata["successful_records"] == 32
assert metadata["failed_records"] == 0

print("Top-32 pilot scores validated:", len(scores))
print("Document:", manifest["document_id"][0])
print("Target:", scores[0]["target_label"])
print("Prompt version:", prompt_version)
PY

echo
echo "Top-32 scoring artifacts:"
ls -lh "${PILOT_DIR}"/qwen_scores."${PROMPT_VERSION}".pilot*

echo
echo "Qwen top-32 pilot scoring completed."
