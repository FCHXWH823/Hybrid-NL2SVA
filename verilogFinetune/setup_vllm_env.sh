#!/bin/bash
# Build a SEPARATE conda env for vLLM inference.
#
# Deliberately not installed into /scratch/wx2356/env/Qwen3Finetune: vLLM pins
# its own torch build and would likely drag that env's torch 2.6.0+cu124 with
# it. Qwen3Finetune is the training environment, it was already destroyed once
# by the /scratch purge and rebuilt, and nothing about inference is worth
# risking it. Two envs, one job each.
#
# Python 3.12 here rather than 3.10: the 3.10 pin existed only because
# LLaMA-Factory requires numpy<2, which has no cp313 wheels. vLLM has no such
# constraint.
#
# Must run on a compute node -- the login node's 3 GiB cgroup cap SIGKILLs
# conda's repodata solve (exit 137), same failure as the training env build.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/wx2356/env/vLLM}"
PY_VERSION="${PY_VERSION:-3.12}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/wx2356/.conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/wx2356/.cache/pip}"
export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

echo "=== [1/3] creating env at $ENV_PREFIX (python $PY_VERSION) ==="
conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
    "python=${PY_VERSION}" pip

conda activate "$ENV_PREFIX"
python -VV
python -m pip install --upgrade pip setuptools wheel

echo "=== [2/3] vllm (pulls its own matching torch) ==="
pip install vllm

echo "=== [3/3] verification ==="
python - <<'PY'
import vllm, torch, transformers
from packaging import version
print("vllm         :", vllm.__version__)
print("torch        :", torch.__version__, "| cuda build:", torch.version.cuda)
print("transformers :", transformers.__version__)
# Qwen3 support landed in vLLM 0.8.5
ok = version.parse(vllm.__version__.split("+")[0]) >= version.parse("0.8.5")
print("Qwen3 support:", "YES" if ok else "NO -- too old")
from vllm import LLM, SamplingParams  # noqa: F401
print("LLM/SamplingParams importable: yes")
print()
print("ALL CHECKS PASSED" if ok else "VERSION TOO OLD")
PY
