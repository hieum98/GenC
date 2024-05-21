#!/bin/bash

#SBATCH --nodes=4              # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=1    # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:1           # Request N GPUs per machine
#SBATCH --mem=100G    
#SBATCH --constraint=gpu-80gb|h100
#SBATCH --cpus-per-task=5
#SBATCH --job-name=genclm
#SBATCH --partition=gpulong
#SBATCH --account=uonlp
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/genclm-1.5b-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/genclm-1.5b-%j.err

# Activate conda environment
source /home/hieum/.bashrc
conda activate llm
cd /home/hieum/uonlp/LLM_Emb

export HF_HOME=/home/hieum/uonlp/hf_cache

# Run SFT training script
srun python -m genc.main \
    --config_file scripts/configs/sft/full_phi1.5.yaml \
    --nodes 1 \
    --devices 8 \
    --mode sft \
    --output_dir output/sft_1.5b 
