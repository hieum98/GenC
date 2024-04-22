#!/bin/bash

#SBATCH --nodes=8              # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=1    # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:1           # Request N GPUs per machine
#SBATCH --mem=100G    
#SBATCH --constraint=gpu-80gb|h100|gpu-40gb
#SBATCH --cpus-per-task=5
#SBATCH --job-name=genclm
#SBATCH --partition=gpulong
#SBATCH --account=uonlp
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/genclm-7b-simcse-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/genclm-7b-simcse-%j.err

# Activate conda environment
source /home/hieum/.bashrc
conda activate llm
cd /home/hieum/uonlp/LLM_Emb

export HF_HOME=/home/hieum/uonlp/hf_cache

# Debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python -m genc.main \
    --config_file scripts/configs/fulldata.yaml \
    --nodes 8 \
    --devices 1 \
    --mode esft \
    --output_dir output/esft_mistral_7b
    # --checkpoint_path output/simcse_esft_7b/checkpoints/step_200.ckpt

