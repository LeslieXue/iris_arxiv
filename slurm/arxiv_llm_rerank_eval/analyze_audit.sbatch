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

if [[ ! -f "${OUTPUT_DIR}/audit_manifest.parquet" ]]; then
    echo "Missing audit manifest" >&2
    exit 1
fi

if [[ ! -f "${OUTPUT_DIR}/teacher_inputs.audit.jsonl" ]]; then
    echo "Missing audit teacher inputs" >&2
    exit 1
fi

if [[ ! -f "${OUTPUT_DIR}/qwen_scores.teacher_v2.audit.jsonl" ]]; then
    echo "Missing teacher-v2 audit scores" >&2
    exit 1
fi

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/analyze_audit.py \
    --config "${EXPERIMENT_CONFIG}"

"${IRIS_PYTHON}" - <<'PY'
import json
from pathlib import Path

import polars as pl

root = Path(
    "/work/jx198/iris_arxiv/"
    "experiments/arxiv_llm_rerank_eval/v1"
)

metrics_path = root / "audit_metrics.json"
scored_path = root / "audit_scored.parquet"
review_path = root / "audit_review.md"

metrics = json.loads(
    metrics_path.read_text(encoding="utf-8")
)
scored = pl.read_parquet(scored_path)

assert metrics["prompt_version"] == "teacher_v2"
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
    "${OUTPUT_DIR}/audit_metrics.json" \
    "${OUTPUT_DIR}/audit_scored.parquet" \
    "${OUTPUT_DIR}/audit_review.md"

echo
echo "Teacher-v2 audit analysis job completed."