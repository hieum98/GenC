python -m scripts.merge_model \
    --config_file output/edpo_8b_instruct_hard/config.yaml \
    --checkpoint_path output/edpo_8b_instruct_hard/checkpoints/step_350.ckpt \
    --output_dir ./checkpoint/edpo_8b_instruct_hard \
    --merge_type emb