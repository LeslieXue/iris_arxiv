#!/bin/bash

#SBATCH --job-name=analyze-target-audit
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/analyze-target-audit-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/analyze-target-audit-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc-v2

export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-2}"

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
REPO_ROOT=/hpc/home/jx198/iris_arxiv
EXPERIMENT_CONFIG=experiments/arxiv_llm_rerank_eval/config.target.toml
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v2_target_conditioned

mkdir -p /work/jx198/iris_arxiv/logs
cd "${REPO_ROOT}"

PROMPT_VERSION=$(
    "${IRIS_PYTHON}" -c \
    'import tomllib; print(tomllib.load(open("experiments/arxiv_llm_rerank_eval/config.target.toml", "rb"))["teacher"]["prompt_version"])'
)

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Prompt version: ${PROMPT_VERSION}"

for path in \
    "${OUTPUT_DIR}/audit_manifest.parquet" \
    "${OUTPUT_DIR}/target_teacher_inputs.audit.jsonl" \
    "${OUTPUT_DIR}/qwen_scores.${PROMPT_VERSION}.audit.jsonl"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing target audit artifact: ${path}" >&2
        exit 1
    fi
done

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/analyze_target_audit.py \
    --config "${EXPERIMENT_CONFIG}"

PROMPT_VERSION="${PROMPT_VERSION}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

root = Path(os.environ["OUTPUT_DIR"])
version = os.environ["PROMPT_VERSION"]
metrics_path = root / f"target_audit_metrics.{version}.json"
scored_path = root / f"target_audit_scored.{version}.parquet"
review_path = root / f"target_audit_review.{version}.md"

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
scored = pl.read_parquet(scored_path)

assert metrics["prompt_version"] == version
assert metrics["overall"]["records"] == 36
assert scored.height == 36
assert set(scored["target_label"].unique()) == {
    "cs.AI", "cs.DS", "cs.PL"
}
assert review_path.exists() and review_path.stat().st_size > 0

print("Target audit analysis validated.")
print("Scored rows:", scored.height)
print("Review report bytes:", review_path.stat().st_size)
PY

echo
echo "Target audit analysis artifacts:"
ls -lh \
    "${OUTPUT_DIR}/target_audit_metrics.${PROMPT_VERSION}.json" \
    "${OUTPUT_DIR}/target_audit_scored.${PROMPT_VERSION}.parquet" \
    "${OUTPUT_DIR}/target_audit_review.${PROMPT_VERSION}.md"

echo
echo "Target-conditioned audit analysis completed."
