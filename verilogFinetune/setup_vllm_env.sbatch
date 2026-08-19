#!/bin/bash -x
#SBATCH --output=vllm_env_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --partition=cpu_short
#SBATCH --job-name=vllm_env --account=torch_pr_67_general
#SBATCH --mail-type=END
#SBATCH --mail-user=wx2356@nyu.edu

set -euo pipefail
cd /scratch/wx2356/Hybrid-NL2SVA/verilogFinetune
bash ./setup_vllm_env.sh
