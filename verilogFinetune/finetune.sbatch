#!/bin/bash -x
#SBATCH --output=out_%A_%j.10g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=finetune.sbatch --account=pr_140_tandon_advanced
#SBATCH --mail-type=BEGIN,END 
#SBATCH --mail-user=wx2356@nyu.edu

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/wx2356/env/Finetune;
cd /scratch/wx2356/verilogFinetune;
python /scratch/wx2356/verilogFinetune/save_to_hub.py;
python /scratch/wx2356/verilogFinetune/save_machine_to_hub.py;
python /scratch/wx2356/verilogFinetune/save_prompt_guided_to_hub.py;
conda deactivate;

# srun --accel-bind=v \
#     /bin/bash -c \
#     'conda activate ./difflogic; python ./diffAIG/experiments/main.py  -bs 100 -t  10 --dataset mnist20x20 -ni 200_000 -ef 1_000 -k  8_000 -l 6 --compile_model; conda deactivate;'