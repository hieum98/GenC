#!/bin/bash

#SBATCH --nodes=12              # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=1    # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:1           # Request N GPUs per machine
#SBATCH --mem=100G    
#SBATCH --constraint=gpu-80gb|h100|gpu-40gb
#SBATCH --cpus-per-task=5
#SBATCH --job-name=genclm
#SBATCH --partition=gpulong
#SBATCH --account=uonlp
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/genclm-7b-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/genclm-7b-%j.err

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
    --config_file scripts/configs/msmarco_llamma3.yaml \
    --nodes 1 \
    --devices 8 \
    --mode edpo \
    --output_dir output/edpo_8b_instruct

