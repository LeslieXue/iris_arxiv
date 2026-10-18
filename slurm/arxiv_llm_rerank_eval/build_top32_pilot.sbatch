#!/bin/bash

#SBATCH --job-name=build-top32-pilot
#SBATCH --account=biostat
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/build-top32-pilot-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/build-top32-pilot-%j.err

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
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v2_target_conditioned/pilot_top32

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/build_top32_pilot.py \
    --config "${EXPERIMENT_CONFIG}"

OUTPUT_DIR="${OUTPUT_DIR}" \
"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import polars as pl

root = Path(os.environ["OUTPUT_DIR"])
manifest = pl.read_parquet(root / "pilot_manifest.parquet")
with (root / "target_teacher_inputs.pilot.jsonl").open(
    "r", encoding="utf-8"
) as handle:
    inputs = [json.loads(line) for line in handle if line.strip()]

assert manifest.height == 32
assert manifest["document_id"].n_unique() == 1
assert manifest["query_id"].n_unique() == 8
assert manifest["chunk_id"].n_unique() == 32
assert manifest.filter(pl.col("original_iris_selection")).height == 8
assert len(inputs) == 32

counts = manifest.group_by("query_id").len().sort("query_id")
assert counts["len"].to_list() == [4] * 8

safe_keys = {
    "document_id",
    "query_id",
    "chunk_id",
    "target_label",
    "target_description",
    "chunk_text",
}
assert all(set(record) == safe_keys for record in inputs)

metadata = json.loads(
    (root / "pilot_metadata.json").read_text(encoding="utf-8")
)
assert metadata["candidate_records"] == 32
assert metadata["purpose"] == "descriptive_top32_reranking_pilot"

print("Top-32 pilot validation succeeded.")
print("Document:", metadata["document_id"])
print("Target:", metadata["target_label"])
print("Source:", metadata["source_path"])
print(counts)
PY

echo
echo "Top-32 pilot artifacts:"
ls -lh \
    "${OUTPUT_DIR}/pilot_manifest.parquet" \
    "${OUTPUT_DIR}/target_teacher_inputs.pilot.jsonl" \
    "${OUTPUT_DIR}/pilot_metadata.json"

echo
echo "Top-32 pilot build completed."
