cd /home/hieum/uonlp/LLM_Emb
source /home/hieum/.bashrc
conda activate llm

export WANDB_PROJECT="em_lm"

# accelerate launch \
#     --config_file /home/hieum/uonlp/LLM_Emb/scripts/configs/dpp_bf16_3gpus.yaml \
python \
    -m hdlm.run \
    --output_dir /home/hieum/uonlp/LLM_Emb/hdlm/outputs/dpp_bf16_3gpus \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --train_data /home/hieum/uonlp/LLM_Emb/hdlm/toy_data \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --max_steps 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --use_gc \
    --gc_mini_batch_size 2 \
    --normalized \
    --temperature 0.02 \
    --dataloader_drop_last \
    --passage_max_len 2048 \
    --mode unified \
    --logging_steps 1 \
    --bf16 \
    --pooling_method mean \
    --loss_gen_type mixed \
    --attn_implementation sdpa \
    --gradient_checkpointing \
    --save_steps 2
