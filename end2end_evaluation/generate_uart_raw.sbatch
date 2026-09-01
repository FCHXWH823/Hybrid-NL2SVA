#!/bin/bash -x
#SBATCH --output=uart_raw_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=uart_raw --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Env-driven for the same reason as generate_vllm_fveval.sbatch: sbatch
# silently dropped script arguments earlier in this project.
#
#   GEN_MODEL      HF id or local path   (required)
#   GEN_MODEL_TAG  short name for output filenames (required)
#   GEN_RTL_DIR    AssertLLM's rtl/uart  (required)

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${VLLM_ENV:-/scratch/wx2356/env/vLLM}"

# The anaconda module's libstdc++ (GCC 12) shadows this env's own (GCC 15),
# and vLLM's libicui18n needs CXXABI_1.3.15 from the newer one.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

# Keep every vLLM cache off $HOME -- its torch.compile cache once wrote 1313
# files there and blew the 30k-inode quota, killing engine startup.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wx2356/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wx2356/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wx2356/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wx2356/.cache}"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /scratch/wx2356/Hybrid-NL2SVA
python ./end2end_evaluation/generate_uart_raw.py \
    --model "$GEN_MODEL" \
    --model-tag "$GEN_MODEL_TAG" \
    --uart-rtl-dir "$GEN_RTL_DIR" \
    --max-new-tokens "${GEN_MAXNEW:-31000}" \
    --temperature "${GEN_TEMP:-0.8}" \
    --top-p "${GEN_TOPP:-0.95}" \
    --n "${GEN_N:-1}"

conda deactivate
