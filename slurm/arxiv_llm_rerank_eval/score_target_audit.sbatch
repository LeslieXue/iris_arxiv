#!/bin/bash

#SBATCH --job-name=qwen-target-audit
#SBATCH --account=engelhardlab
#SBATCH --partition=engelhardlab-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/qwen-target-audit-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/qwen-target-audit-%j.err

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
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v2_target_conditioned

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

PROMPT_VERSION=$(
    "${IRIS_PYTHON}" -c \
    'import tomllib; print(tomllib.load(open("experiments/arxiv_llm_rerank_eval/config.target.toml", "rb"))["teacher"]["prompt_version"])'
)

export PROMPT_VERSION
export AUDIT_INPUT="${OUTPUT_DIR}/target_teacher_inputs.audit.jsonl"
export AUDIT_OUTPUT="${OUTPUT_DIR}/qwen_scores.${PROMPT_VERSION}.audit.jsonl"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"
echo "Prompt version: ${PROMPT_VERSION}"
echo "Audit input: ${AUDIT_INPUT}"
echo "Audit output: ${AUDIT_OUTPUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

"${IRIS_PYTHON}" --version
nvidia-smi

if [[ ! -f "${AUDIT_INPUT}" ]]; then
    echo "Missing target-conditioned audit input: ${AUDIT_INPUT}" >&2
    exit 1
fi

"${IRIS_PYTHON}" - <<'PY'
import os
import tomllib
from pathlib import Path

import torch

config_path = Path(
    "experiments/arxiv_llm_rerank_eval/config.target.toml"
)
with config_path.open("rb") as handle:
    config = tomllib.load(handle)

prompt_version = config["teacher"]["prompt_version"]
assert prompt_version == os.environ["PROMPT_VERSION"]

prompt_path = config_path.parent / "prompts" / f"{prompt_version}.txt"
if not prompt_path.exists():
    raise FileNotFoundError(prompt_path)

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Teacher model:", config["teacher"]["model_id"])
print("Prompt path:", prompt_path)

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
    --input "${AUDIT_INPUT}" \
    --output "${AUDIT_OUTPUT}" \
    --overwrite

"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

output_path = Path(os.environ["AUDIT_OUTPUT"])
prompt_version = os.environ["PROMPT_VERSION"]
manifest = pl.read_parquet(output_path.parent / "audit_manifest.parquet")

with output_path.open("r", encoding="utf-8") as handle:
    scores = [json.loads(line) for line in handle if line.strip()]

error_path = output_path.with_suffix(".errors.jsonl")
metadata_path = output_path.with_suffix(".metadata.json")

assert manifest.height == 36
assert len(scores) == 36

score_keys = {
    (row["document_id"], row["query_id"], row["chunk_id"])
    for row in scores
}
manifest_keys = {
    (row["document_id"], row["query_id"], row["chunk_id"])
    for row in manifest.iter_rows(named=True)
}
assert len(score_keys) == 36
assert score_keys == manifest_keys

score_fields = (
    "target_evidence",
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
)
for score in scores:
    assert score["target_label"] in {"cs.AI", "cs.DS", "cs.PL"}
    for field in score_fields:
        value = score[field]
        assert isinstance(value, int), (field, value)
        assert 0 <= value <= 4, (field, value)
    assert score["brief_reason"].strip()
    assert len(score["brief_reason"].split()) <= 40

assert error_path.exists()
assert error_path.stat().st_size == 0

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
assert metadata["prompt_version"] == prompt_version
assert metadata["task"] == "target_conditioned_explanatory_evidence"
assert metadata["successful_records"] == 36
assert metadata["failed_records"] == 0

print("Validated target-conditioned scores:", len(scores))
print("Prompt version:", prompt_version)
print("Target-conditioned score validation succeeded.")
PY

echo
echo "Target-conditioned scoring artifacts:"
ls -lh "${OUTPUT_DIR}"/qwen_scores."${PROMPT_VERSION}".audit*

echo
echo "Qwen target-conditioned audit scoring completed."
