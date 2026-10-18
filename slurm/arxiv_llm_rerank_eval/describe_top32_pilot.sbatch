#!/bin/bash

#SBATCH --job-name=describe-top32
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/describe-top32-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/describe-top32-%j.err

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
PILOT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v2_target_conditioned/pilot_top32

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
    "${PILOT_DIR}/pilot_manifest.parquet" \
    "${PILOT_DIR}/pilot_metadata.json" \
    "${PILOT_DIR}/qwen_scores.${PROMPT_VERSION}.pilot.jsonl"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing top-32 pilot artifact: ${path}" >&2
        exit 1
    fi
done

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/describe_top32_pilot.py \
    --config "${EXPERIMENT_CONFIG}"

PROMPT_VERSION="${PROMPT_VERSION}" \
PILOT_DIR="${PILOT_DIR}" \
"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

root = Path(os.environ["PILOT_DIR"])
version = os.environ["PROMPT_VERSION"]
summary_path = root / f"pilot_comparison.{version}.json"
report_path = root / f"pilot_comparison.{version}.md"
scored_path = root / f"pilot_scored.{version}.parquet"

summary = json.loads(summary_path.read_text(encoding="utf-8"))
scored = pl.read_parquet(scored_path)

assert summary["candidate_records"] == 32
assert summary["queries"] == 8
assert len(summary["query_comparisons"]) == 8
assert scored.height == 32
assert scored.filter(pl.col("original_iris_selection")).height == 8
assert scored.filter(pl.col("qwen_selected")).height == 8
assert report_path.exists() and report_path.stat().st_size > 0

print("Top-32 descriptive comparison validated.")
print("Changed queries:", summary["changed_queries"])
print(
    "Queries with tied Qwen maximum:",
    summary["queries_with_tied_qwen_maximum"],
)
print(
    "Queries with multiple score levels:",
    summary["queries_with_multiple_score_levels"],
)
PY

echo
echo "Top-32 descriptive artifacts:"
ls -lh \
    "${PILOT_DIR}/pilot_comparison.${PROMPT_VERSION}.json" \
    "${PILOT_DIR}/pilot_comparison.${PROMPT_VERSION}.md" \
    "${PILOT_DIR}/pilot_scored.${PROMPT_VERSION}.parquet"

echo
echo "Top-32 descriptive comparison completed."
