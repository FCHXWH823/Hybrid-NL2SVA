#!/bin/bash -x
#SBATCH --output=qwen3_8b_think_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=qwen3_8b_think --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wx2356@nyu.edu

# NOTE: the old sbatch files in this directory load anaconda3/2020.07 and
# source /share/apps/anaconda3/2020.07/..., neither of which exists on this
# cluster any more (only anaconda3/2025.06 is available). They also cd into
# /scratch/wx2356/verilogFinetune, a stale copy that has neither this dataset
# nor its train script. Both are corrected below.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
# /scratch/wx2356/env/Finetune was destroyed by the scratch purge (gutted
# stdlib, zero torch .so files); Qwen3Finetune is the rebuilt replacement,
# created by setup_finetune_env.sh.
conda activate "${FINETUNE_ENV:-/scratch/wx2356/env/Qwen3Finetune}"

# Keep model downloads on /scratch, not the 30k-inode $HOME. This is the
# account's real, pre-existing HF cache -- where prefetch_qwen3.sbatch put the
# Qwen3 weights -- so the model load is a cache hit. Named explicitly rather
# than left to the inherited value: Slurm propagates the submitting shell's
# environment by default, but a job submitted with --export=NONE would
# otherwise fall back to an empty directory and re-download 44GB.
export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune

export QWEN3_SIZE=8B
export NPROC_PER_NODE=4

bash ./train_qwen3_think.sh

conda deactivate
