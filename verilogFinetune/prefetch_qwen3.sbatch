#!/bin/bash -x
#SBATCH --output=prefetch_qwen3_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=cpu_short
#SBATCH --job-name=prefetch_qwen3 --account=torch_pr_67_general
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Downloads Qwen3-8B (~16GB) and Qwen3-14B (~28GB) into HF_HOME on a CPU node,
# so the H200 training jobs start from a warm cache instead of spending their
# allocation on the transfer.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${FINETUNE_ENV:-/scratch/wx2356/env/Qwen3Finetune}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"
# Faster multi-part transfers; harmless if the accelerator isn't installed.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
python ./prefetch_qwen3.py

conda deactivate
