#!/bin/bash

#SBATCH --job-name=analyze-llm-audit
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/analyze-llm-audit-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/analyze-llm-audit-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-2}"

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
REPO_ROOT=/hpc/home/jx198/iris_arxiv
EXPERIMENT_CONFIG=experiments/arxiv_llm_rerank_eval/config.toml
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v1

mkdir -p /work/jx198/iris_arxiv/logs

cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"

PROMPT_VERSION="$(
    "${IRIS_PYTHON}" - "${EXPERIMENT_CONFIG}" <<'PY'
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
with config_path.open("rb") as handle:
    config = tomllib.load(handle)

print(config["teacher"]["prompt_version"])
PY
)"

SCORE_PATH="${OUTPUT_DIR}/qwen_scores.${PROMPT_VERSION}.audit.jsonl"
METRICS_PATH="${OUTPUT_DIR}/audit_metrics.${PROMPT_VERSION}.json"
SCORED_PATH="${OUTPUT_DIR}/audit_scored.${PROMPT_VERSION}.parquet"
REVIEW_PATH="${OUTPUT_DIR}/audit_review.${PROMPT_VERSION}.md"

echo "Prompt version: ${PROMPT_VERSION}"

if [[ ! -f "${OUTPUT_DIR}/audit_manifest.parquet" ]]; then
    echo "Missing audit manifest" >&2
    exit 1
fi

if [[ ! -f "${OUTPUT_DIR}/teacher_inputs.audit.jsonl" ]]; then
    echo "Missing audit teacher inputs" >&2
    exit 1
fi

if [[ ! -f "${SCORE_PATH}" ]]; then
    echo "Missing audit scores: ${SCORE_PATH}" >&2
    exit 1
fi

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/analyze_audit.py \
    --config "${EXPERIMENT_CONFIG}"

PROMPT_VERSION="${PROMPT_VERSION}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

root = Path(os.environ["OUTPUT_DIR"])
prompt_version = os.environ["PROMPT_VERSION"]

metrics_path = root / f"audit_metrics.{prompt_version}.json"
scored_path = root / f"audit_scored.{prompt_version}.parquet"
review_path = root / f"audit_review.{prompt_version}.md"

metrics = json.loads(
    metrics_path.read_text(encoding="utf-8")
)
scored = pl.read_parquet(scored_path)

assert metrics["prompt_version"] == prompt_version
assert metrics["overall"]["records"] == 36
assert scored.height == 36
assert review_path.exists()
assert review_path.stat().st_size > 0

print("Analysis outputs validated.")
print("Scored rows:", scored.height)
print("Review report bytes:", review_path.stat().st_size)
PY

echo
echo "Analysis artifacts:"
ls -lh \
    "${METRICS_PATH}" \
    "${SCORED_PATH}" \
    "${REVIEW_PATH}"

echo
echo "${PROMPT_VERSION} audit analysis job completed."
