#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1               # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=5     # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:5            # Request N GPUs per machine
#SBATCH --mem=300G       # Request N GB per GPU
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=10
#SBATCH --job-name=dpoc7b
#SBATCH --partition=preempt
#SBATCH --account=uonlp
#SBATCH --output=/home/hieum/uonlp/LLM_Emb/mistral-7b-dpoc-msmarco-%j.out
#SBATCH --error=/home/hieum/uonlp/LLM_Emb/mistral-7b-dpoc-msmarco-%j.err

# Activate conda environment
source activate llm
cd /home/hieum/uonlp/LLM_Emb

# Debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python -m genc.main scripts/configs/mistral-7b-dpoc_1_node_5gpus.yaml

