#!/bin/bash -x
#SBATCH --output=fveval_all_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=fveval_all --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Full 73-record inference sweep over FVEval-Verified nl2sva_human.
# One GPU: 8B in bf16 is ~16GB of the H200's 141GB, leaving ample room for
# a batch of 8 sequences at ~1-2k prompt tokens each.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${FINETUNE_ENV:-/scratch/wx2356/env/Qwen3Finetune}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
python ./generate_fveval_outputs.py "$@"

conda deactivate
