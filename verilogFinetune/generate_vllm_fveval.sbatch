#!/bin/bash -x
#SBATCH --output=vllm_gen_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=vllm_gen --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Env-driven, not argv-driven: sbatch silently dropped script arguments for
# prefetch_qwen3.sbatch earlier in this project.
#
# One H200 per job. 14B in bf16 is ~28GB of 141GB; the rest is KV cache, which
# is what lets vLLM keep many 32k-context sequences resident at once.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${VLLM_ENV:-/scratch/wx2356/env/vLLM}"

# The anaconda3/2025.06 module exports LD_LIBRARY_PATH=/share/apps/anaconda3/
# 2025.06/lib, whose libstdc++.so.6.0.29 (GCC 12) shadows this env's own
# libstdc++.so.6.0.35 (GCC 15). vLLM's libicui18n.so.78 needs CXXABI_1.3.15,
# which only the newer one provides -- without this line even `import sqlite3`
# dies inside vLLM's structured-output import chain.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

# Every cache vLLM writes must live on /scratch, not $HOME. vLLM's
# torch.compile cache alone wrote 1313 files under ~/.cache/vllm and pushed
# $HOME past its 30k-inode quota, after which engine startup died with
# "OSError: [Errno 122] Disk quota exceeded" -- 4 jobs failed that way before
# these lines existed. $HOME is only 30k inodes; /scratch has millions.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wx2356/.cache/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wx2356/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wx2356/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wx2356/.cache}"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
python ./generate_vllm_fveval.py \
    --model "$GEN_MODEL" \
    --task "$GEN_TASK" \
    --data "$GEN_DATA" \
    --output "$GEN_OUT" \
    --sva-output "$GEN_SVA_OUT" \
    --max-new-tokens "${GEN_MAXNEW:-32768}" \
    --temperature "${GEN_TEMP:-0.8}" \
    --top-p "${GEN_TOPP:-0.95}" \
    --n "${GEN_N:-1}"

conda deactivate
