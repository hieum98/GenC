#!/bin/bash
#SBATCH --job-name=simcse
#SBATCH --account=uonlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          
#SBATCH --mem=50G
#SBATCH --partition=preempt
#SBATCH --constraint=gpu-40gb|gpu-80gb|h100|v100
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/simcse-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/simcse-%j.err
#SBATCH --array=0-99


######################
### Set enviroment ###
######################
# Activate conda environment
source activate llm
cd /home/hieum/uonlp/LLM_Emb
######################

######################
#### Set network #####
######################

echo "Part: $SLURM_ARRAY_TASK_ID"

python scripts/process_simcse.py --part $SLURM_ARRAY_TASK_ID


