#!/bin/bash -x
#SBATCH --output=upload_qwen3_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=cpu_short
#SBATCH --job-name=upload_qwen3 --account=torch_pr_67_general
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Pushes the fine-tuned checkpoint to the Hub from a compute node. Not the
# login node: a 15GB transfer there would run for many minutes under a 3GiB
# cgroup cap and against an interactive session that can drop, which would
# leave a half-populated repo on the Hub.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${FINETUNE_ENV:-/scratch/wx2356/env/Qwen3Finetune}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
python ./upload_qwen3_to_hub.py "$@"

conda deactivate
