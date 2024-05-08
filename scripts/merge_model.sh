python -m scripts.merge_model \
    --config_file output/edpo_1.5b_instruct/config.yaml \
    --checkpoint_path output/edpo_1.5b_instruct/checkpoints/step_450.ckpt \
    --output_dir ./checkpoint/edpo_1.5b_instruct \
    --merge_type emb