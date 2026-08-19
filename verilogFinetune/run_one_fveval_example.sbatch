#!/bin/bash -x
#SBATCH --output=fveval_one_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=fveval_one --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Single-datapoint inference. One GPU is plenty: 8B in bf16 is ~16GB against
# the H200's 141GB, so no sharding is needed.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${FINETUNE_ENV:-/scratch/wx2356/env/Qwen3Finetune}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
python ./run_one_fveval_example.py "$@"

conda deactivate
