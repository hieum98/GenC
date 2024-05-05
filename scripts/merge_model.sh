python -m scripts.merge_model \
    --config_file output/esft_1.5b_instruct/config.yaml \
    --checkpoint_path output/esft_1.5b_instruct/checkpoints/step_1000.ckpt \
    --output_dir ./checkpoint/esft_1.5b_instruct \
    --merge_type emb