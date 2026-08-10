#!/bin/bash
# Build the conda env for Qwen3 fine-tuning with LLaMA-Factory.
#
# The previous env (/scratch/wx2356/env/Finetune) was gutted by the /scratch
# purge -- its stdlib (codecs.py, os.py, site.py) and every torch .so are gone,
# so its interpreter can't boot. This builds a fresh one alongside it rather
# than trying to repair it.
#
# Version pins are not arbitrary:
#   * transformers 4.51.3 -- Qwen3 support landed in 4.51.0, and
#     LLaMA-Factory 0.9.3.dev0's requirements.txt caps it at <=4.52.4
#     (excluding 4.52.0). 4.51.3 sits safely inside both bounds.
#   * torch cu124 -- H200 is sm_90; the CUDA 12.x wheels cover it.
#   * The rest follow LLaMA-Factory's own requirements.txt upper bounds;
#     numpy<2 in particular is required by it.
#
# Caches are forced onto /scratch: $HOME is at ~87% of its 30k-inode quota and
# a torch install would blow straight through the remainder.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/wx2356/env/Qwen3Finetune}"
LLAMA_FACTORY="${LLAMA_FACTORY:-/scratch/wx2356/LLaMA-Factory}"
PY_VERSION="${PY_VERSION:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/wx2356/.conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/wx2356/.cache/pip}"
export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

echo "=== [1/5] creating env at $ENV_PREFIX (python $PY_VERSION) ==="
# --override-channels -c conda-forge on purpose: the default repo.anaconda.com
# channels are gated behind a Terms-of-Service acceptance on this install
# (CondaToSNonInteractiveError), which would otherwise have to be accepted
# account-wide. conda-forge carries no such gate and supplies python+pip fine.
conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
    "python=${PY_VERSION}" pip

conda activate "$ENV_PREFIX"
python -VV
python -m pip install --upgrade pip setuptools wheel

echo "=== [2/5] torch (cu124, for H200 sm_90) ==="
pip install --index-url "$TORCH_INDEX" torch torchvision torchaudio

echo "=== [3/5] transformers stack (LLaMA-Factory-compatible pins) ==="
pip install \
    "transformers==4.51.3" \
    "datasets>=2.16.0,<=3.6.0" \
    "accelerate>=0.34.0,<=1.7.0" \
    "peft>=0.14.0,<=0.15.2" \
    "trl>=0.8.6,<=0.9.6" \
    "tokenizers>=0.19.0,<=0.21.1" \
    "numpy<2.0.0" \
    "pydantic<=2.10.6" \
    sentencepiece tiktoken protobuf einops scipy \
    "matplotlib>=3.7.0" "pandas>=2.0.0" pyyaml packaging fire omegaconf "tyro<0.9.0"

echo "=== [4/5] deepspeed (ZeRO-3) ==="
pip install "deepspeed==0.16.5"

echo "=== [5/5] llamafactory (editable, deps already pinned above) ==="
pip install --no-deps -e "$LLAMA_FACTORY"

echo
echo "=== verification ==="
python - <<'PY'
import importlib
from packaging import version

ok = True
for pkg in ["torch", "transformers", "deepspeed", "accelerate", "peft", "trl", "datasets", "llamafactory"]:
    try:
        m = importlib.import_module(pkg)
        print("%-16s %s" % (pkg, getattr(m, "__version__", "?")))
    except Exception as exc:
        print("%-16s FAILED: %s" % (pkg, exc))
        ok = False

import torch, transformers
print()
print("torch CUDA build :", torch.version.cuda)
print("cuda available   :", torch.cuda.is_available(), "(False is expected on a login node)")

tv = version.parse(transformers.__version__)
print("Qwen3 support    :", "YES" if tv >= version.parse("4.51.0") else "NO")
from transformers.models.qwen3 import Qwen3ForCausalLM  # noqa: F401
print("Qwen3ForCausalLM : importable")

from llamafactory.data.template import TEMPLATES
t = TEMPLATES["qwen3"]
print("qwen3 template   :", type(t).__name__, t.thought_words)
assert type(t).__name__ == "ReasoningTemplate", "qwen3 must be a ReasoningTemplate"
print()
print("ALL CHECKS PASSED" if ok else "SOME IMPORTS FAILED")
PY
