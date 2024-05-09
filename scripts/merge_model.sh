python -m scripts.merge_model \
    --config_file esft_msmarco_8b_instruct/config.yaml \
    --checkpoint_path esft_msmarco_8b_instruct/checkpoints/step_500.ckpt \
    --output_dir ./checkpoint/esft_msmarco_8b_instruct \
    --merge_type emb