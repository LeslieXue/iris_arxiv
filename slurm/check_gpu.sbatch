#!/bin/bash

#SBATCH --job-name=iris-gpu-check
#SBATCH --account=biostat
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/jx198/iris_arxiv/logs/gpu-check-%j.out
#SBATCH --error=/work/jx198/iris_arxiv/logs/gpu-check-%j.err

set -eo pipefail

module purge
module load Anaconda3/2024.02

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/jx198/iris_arxiv/conda/envs/iris-dcc

export HF_HOME=/work/jx198/iris_arxiv/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

IRIS_PYTHON=/work/jx198/iris_arxiv/conda/envs/iris-dcc/bin/python

echo "Host: $(hostname)"
echo "Python: ${IRIS_PYTHON}"

nvidia-smi

"${IRIS_PYTHON}" - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "answerdotai/ModernBERT-base"

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

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="auto",
).to("cuda")
model.eval()

batch = tokenizer(
    [
        "This is a short arXiv GPU test.",
        "IRIS retrieves informative text chunks.",
    ],
    padding=True,
    return_tensors="pt",
).to("cuda")

with torch.inference_mode():
    output = model(**batch)

print("Output shape:", tuple(output.last_hidden_state.shape))
print("Model device:", next(model.parameters()).device)
print("GPU smoke test succeeded.")
PY