#!/bin/bash

#SBATCH --job-name=build-target-audit
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/build-target-audit-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/build-target-audit-%j.err

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
SOURCE_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v1
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v2_target_conditioned

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"
echo "Source directory: ${SOURCE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

for path in \
    "${SOURCE_DIR}/audit_manifest.parquet" \
    "${SOURCE_DIR}/teacher_inputs.audit.jsonl"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing source audit artifact: ${path}" >&2
        exit 1
    fi
done

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/build_target_audit.py \
    --config "${EXPERIMENT_CONFIG}"

OUTPUT_DIR="${OUTPUT_DIR}" \
"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

root = Path(os.environ["OUTPUT_DIR"])
manifest = pl.read_parquet(root / "audit_manifest.parquet")

with (
    root / "target_teacher_inputs.audit.jsonl"
).open("r", encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]

safe_keys = {
    "document_id",
    "query_id",
    "chunk_id",
    "target_label",
    "target_description",
    "chunk_text",
}

assert manifest.height == 36
assert manifest["document_id"].n_unique() == 36
assert len(records) == 36
assert {record["target_label"] for record in records} == {
    "cs.AI",
    "cs.DS",
    "cs.PL",
}

for record in records:
    assert set(record) == safe_keys
    assert record["target_description"].strip()
    assert record["chunk_text"].strip()

counts = {}
for record in records:
    label = record["target_label"]
    counts[label] = counts.get(label, 0) + 1

assert counts == {"cs.AI": 12, "cs.DS": 12, "cs.PL": 12}

print("Validated target-conditioned records:", len(records))
print("Target distribution:", counts)
print("Target-conditioned input validation succeeded.")
PY

echo
echo "Target-conditioned audit artifacts:"
ls -lh \
    "${OUTPUT_DIR}/audit_manifest.parquet" \
    "${OUTPUT_DIR}/target_teacher_inputs.audit.jsonl" \
    "${OUTPUT_DIR}/target_audit_metadata.json"

echo
echo "Target-conditioned audit build completed."
