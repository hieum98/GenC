#!/bin/bash

#SBATCH --nodes=16             # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=8    # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:8           # Request N GPUs per machine
#SBATCH --mem=450G    
#SBATCH --cpus-per-task=5
#SBATCH --job-name=genclm
#SBATCH --partition=standard-g
#SBATCH --account=project_462000558
#SBATCH --time=2-00:00:00        
#SBATCH --output=/scratch/project_462000558/peter/hieu/LLM_Emb/genclm-8b-%j.out
#SBATCH --error=/scratch/project_462000558/peter/hieu/LLM_Emb/genclm-8b-%j.err

# Activate conda environment
source /scratch/project_462000558/peter/hieu/.bashrc
conda activate llm_emb
cd /scratch/project_462000558/peter/hieu/LLM_Emb

# Debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python -m genc.main \
    --config_file scripts/configs/fulldata_llamma3.yaml \
    --nodes 16 \
    --devices 8 \
    --mode esft \
    --output_dir output/esft_8b_instruct

