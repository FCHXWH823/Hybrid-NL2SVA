#!/bin/bash -x
#SBATCH --output=codev_gen_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=codev_gen --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Driven by environment, not argv: sbatch silently dropped script arguments
# for prefetch_qwen3.sbatch earlier in this project, so anything that must
# reach the script is passed with --export instead.
#
# One GPU is enough for both sizes: 8B is ~16GB and 14B ~28GB in bf16, against
# the H200's 141GB, leaving room for the KV cache of a batch at 8k tokens.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${FINETUNE_ENV:-/scratch/wx2356/env/Qwen3Finetune}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
python ./generate_codev_fveval.py \
    --model "$CODEV_MODEL" \
    --data "$CODEV_DATA" \
    --output "$CODEV_OUT" \
    --sva-output "$CODEV_SVA_OUT" \
    --batch-size "${CODEV_BATCH:-8}" \
    --max-new-tokens "${CODEV_MAXNEW:-32768}"

conda deactivate
