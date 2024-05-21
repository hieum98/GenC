# Activate conda environment
source /home/hieum/.bashrc
conda activate llm
cd /home/hieum/uonlp/LLM_Emb

export HF_HOME=/home/hieum/uonlp/hf_cache

# Run SFT training script
python -m genc.main \
    --config_file scripts/configs/sft/full_phi1.5.yaml \
    --nodes 1 \
    --devices 8 \
    --mode sft \
    --output_dir output/sft_1.5b 

python -m genc.main \
    --config_file scripts/configs/sft/full.yaml \
    --nodes 1 \
    --devices 8 \
    --mode sft \
    --output_dir output/sft_7b 

python -m genc.main \
    --config_file scripts/configs/sft/full_llamma3.yaml \
    --nodes 1 \
    --devices 8 \
    --mode sft \
    --output_dir output/sft_8b 
