#!/bin/bash

#SBATCH --job-name=qwen-score-smoke
#SBATCH --account=biostat
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/qwen-score-smoke-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/qwen-score-smoke-%j.err

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
EXPERIMENT_CONFIG=experiments/arxiv_llm_rerank_eval/config.toml
OUTPUT_DIR=/work/jx198/iris_arxiv/experiments/arxiv_llm_rerank_eval/v1

export SCORE_PATH="${OUTPUT_DIR}/qwen_scores.smoke.jsonl"

mkdir -p /work/jx198/iris_arxiv/logs
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"
echo "Experiment config: ${EXPERIMENT_CONFIG}"
echo "Smoke output: ${SCORE_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

"${IRIS_PYTHON}" --version
nvidia-smi

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

if [[ ! -f "${OUTPUT_DIR}/teacher_inputs.jsonl" ]]; then
    echo "Missing teacher inputs: ${OUTPUT_DIR}/teacher_inputs.jsonl" >&2
    exit 1
fi

"${IRIS_PYTHON}" \
    experiments/arxiv_llm_rerank_eval/score_candidates.py \
    --config "${EXPERIMENT_CONFIG}" \
    --output "${SCORE_PATH}" \
    --limit 5 \
    --overwrite

"${IRIS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

score_path = Path(os.environ["SCORE_PATH"])

with score_path.open("r", encoding="utf-8") as handle:
    scores = [
        json.loads(line)
        for line in handle
        if line.strip()
    ]

assert len(scores) == 5, f"Expected 5 scores, found {len(scores)}"

score_fields = [
    "ai_evidence",
    "ds_evidence",
    "pl_evidence",
    "substantive_relevance",
    "specificity",
    "context_completeness",
    "direct_label_disclosure",
    "keyword_only_risk",
]

for score in scores:
    for field in score_fields:
        value = score[field]
        assert isinstance(value, int), (field, value)
        assert 0 <= value <= 4, (field, value)

    assert score["brief_reason"].strip()

print("Validated teacher scores:", len(scores))

for score in scores:
    print(
        json.dumps(
            {
                key: value
                for key, value in score.items()
                if key != "raw_output"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
PY

echo
echo "Smoke-test artifacts:"
ls -lh \
    "${OUTPUT_DIR}"/qwen_scores.smoke*

echo
echo "Qwen teacher scoring smoke test completed."