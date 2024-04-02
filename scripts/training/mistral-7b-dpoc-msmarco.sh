#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1               # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=6     # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:6            # Request N GPUs per machine
#SBATCH --mem=500G
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --job-name=dpoc-7b-msmarco
#SBATCH --partition=preempt
#SBATCH --account=uonlp

# Activate conda environment
source activate $1
cd /home/hieum/uonlp/LLM_Emb

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python -m genc.main scripts/configs/mistral-7b-dpoc._1_node_6_gpus.yaml

