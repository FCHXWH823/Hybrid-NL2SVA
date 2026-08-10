#!/bin/bash -x
#SBATCH --output=env_setup_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --partition=cpu_short
#SBATCH --job-name=qwen3_env_setup --account=torch_pr_67_tandon_advanced
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

# Runs setup_finetune_env.sh on a compute node rather than the login node.
# The login node enforces a 3 GiB cgroup memory cap (memory.max =
# 3221225472); conda's conda-forge repodata parse exceeds it and gets
# SIGKILLed mid-solve (exit 137). Compute nodes have no such cap and, as
# verified, still reach pypi.org and conda.anaconda.org.

cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
bash ./setup_finetune_env.sh
