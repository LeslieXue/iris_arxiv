#!/bin/bash

#SBATCH --job-name=compare-llm-prompts
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/compare-llm-prompts-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/compare-llm-prompts-%j.err

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
BASELINE_VERSION="${1:-teacher_v2}"
CANDIDATE_VERSION="${2:-teacher_v3}"
COMPARISON_NAME="${BASELINE_VERSION}_vs_${CANDIDATE_VERSION}"

mkdir -p /work/jx198/iris_arxiv/logs
cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Baseline prompt: ${BASELINE_VERSION}"
echo "Candidate prompt: ${CANDIDATE_VERSION}"

for version in "${BASELINE_VERSION}" "${CANDIDATE_VERSION}"; do
    path="${OUTPUT_DIR}/audit_scored.${version}.parquet"
    if [[ ! -f "${path}" ]]; then
        echo "Missing scored audit: ${path}" >&2
        exit 1
    fi
done

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/compare_prompt_versions.py \
    --config "${EXPERIMENT_CONFIG}" \
    --baseline "${BASELINE_VERSION}" \
    --candidate "${CANDIDATE_VERSION}"

BASELINE_VERSION="${BASELINE_VERSION}" \
CANDIDATE_VERSION="${CANDIDATE_VERSION}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

root = Path(os.environ["OUTPUT_DIR"])
baseline = os.environ["BASELINE_VERSION"]
candidate = os.environ["CANDIDATE_VERSION"]
name = f"{baseline}_vs_{candidate}"

metrics_path = root / f"prompt_comparison.{name}.json"
paired_path = root / f"prompt_comparison.{name}.parquet"
review_path = root / f"prompt_comparison.{name}.md"

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
paired = pl.read_parquet(paired_path)

assert metrics["baseline_prompt_version"] == baseline
assert metrics["candidate_prompt_version"] == candidate
assert metrics["overall"]["records"] == 36
assert paired.height == 36
assert review_path.exists() and review_path.stat().st_size > 0

print("Comparison outputs validated.")
print("Paired rows:", paired.height)
print("Review report bytes:", review_path.stat().st_size)
PY

echo
echo "Comparison artifacts:"
ls -lh \
    "${OUTPUT_DIR}/prompt_comparison.${COMPARISON_NAME}.json" \
    "${OUTPUT_DIR}/prompt_comparison.${COMPARISON_NAME}.parquet" \
    "${OUTPUT_DIR}/prompt_comparison.${COMPARISON_NAME}.md"

echo
echo "Prompt comparison job completed."
