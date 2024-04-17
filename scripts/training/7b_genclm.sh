#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=3               # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=2     # This needs to match Fabric(devices=...)
#SBATCH --gres=gpu:2            # Request N GPUs per machine
#SBATCH --mem=200G       
#SBATCH --constraint=gpu-80gb,no-mig
#SBATCH --cpus-per-task=10
#SBATCH --job-name=dpoc7b
#SBATCH --partition=gpulong
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
srun python -m genc.main --config_file scripts/configs/msmarco.yaml --nodes 1 --devices 4 --mode esft --output_dir output/esft_msmarco_7b

