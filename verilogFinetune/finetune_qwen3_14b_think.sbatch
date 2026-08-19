#!/bin/bash -x
#SBATCH --output=qwen3_14b_think_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=512GB
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=qwen3_14b_think --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wx2356@nyu.edu

# See finetune_qwen3_8b_think.sbatch for why the module/cd lines differ from
# the older sbatch files in this directory.
#
# 4 GPUs, not 8 -- the allocation caps a single job at 4. This still fits:
# full fine-tuning holds ~16 bytes/param (2 bf16 weights + 2 bf16 grads +
# 8 fp32 AdamW moments + 4 fp32 master), so Qwen3-14B's ~14.8B params need
# ~237GB of ZeRO-3-sharded state = ~59GB per GPU against the H200's 141GB.
# The remaining ~80GB/GPU covers activations (~7-15GB at 16k context with
# gradient checkpointing, which LLaMA-Factory leaves on by default) plus
# ZeRO-3's transient all-gather buffers. Comfortable, but it IS full
# fine-tuning at the edge of one node -- if it ever OOMs, in order of
# preference: drop CUTOFF_LEN to 8192 (p99 of this dataset is ~4.8k tokens,
# so almost nothing gets truncated), or switch DS_CONFIG_PATH to
# examples/deepspeed/ds_z3_offload_config.json to spill optimizer state to
# the node's 2TB of RAM.
#
# GRAD_ACCUM=4 keeps the effective batch at 4 GPUs x 1 x 4 = 16, the same as
# the 8-GPU x accum-2 configuration this replaces -- halving the GPU count
# should not silently halve the batch and change training dynamics.

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${FINETUNE_ENV:-/scratch/wx2356/env/Qwen3Finetune}"

export HF_HOME="${HF_HOME:-/scratch/wx2356/.huggingface}"

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune

export QWEN3_SIZE=14B
export NPROC_PER_NODE=4
export GRAD_ACCUM=4

bash ./train_qwen3_think.sh

conda deactivate
