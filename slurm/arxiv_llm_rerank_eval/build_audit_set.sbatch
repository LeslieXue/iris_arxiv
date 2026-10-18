#!/bin/bash

#SBATCH --job-name=build-llm-audit
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/build-llm-audit-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/build-llm-audit-%j.err

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
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"
echo "Output directory: ${OUTPUT_DIR}"

"${IRIS_PYTHON}" --version

if [[ ! -f "${OUTPUT_DIR}/candidates.parquet" ]]; then
    echo "Missing candidate file: ${OUTPUT_DIR}/candidates.parquet" >&2
    exit 1
fi

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/build_audit_set.py \
    --config "${EXPERIMENT_CONFIG}" \
    --per-class-per-band 4

"${IRIS_PYTHON}" - <<'PY'
import json
from pathlib import Path

import polars as pl

root = Path(
    "/work/jx198/iris_arxiv/"
    "experiments/arxiv_llm_rerank_eval/v1"
)

manifest = pl.read_parquet(
    root / "audit_manifest.parquet"
)

assert manifest.height == 36
assert manifest["document_id"].n_unique() == 36
assert set(manifest["split"].unique().to_list()) == {
    "validation"
}

distribution = (
    manifest.group_by(["label_name", "rank_band"])
    .len()
    .sort(["label_name", "rank_band"])
)

assert set(distribution["len"].to_list()) == {4}

safe_keys = {
    "document_id",
    "query_id",
    "chunk_id",
    "chunk_text",
}

with (
    root / "teacher_inputs.audit.jsonl"
).open("r", encoding="utf-8") as handle:
    records = [
        json.loads(line)
        for line in handle
        if line.strip()
    ]

assert len(records) == 36

for record in records:
    assert set(record) == safe_keys
    assert record["chunk_text"].strip()

print("Validated audit records:", len(records))
print(distribution)
print("Audit set validation succeeded.")
PY

echo
echo "Audit artifacts:"
ls -lh \
    "${OUTPUT_DIR}"/audit_* \
    "${OUTPUT_DIR}"/teacher_inputs.audit.jsonl

echo
echo "Prompt-development audit set completed."