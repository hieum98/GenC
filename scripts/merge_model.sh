python -m scripts.merge_model \
    --config_file output/edpo_msmarco_8b_instruct/config.yaml \
    --checkpoint_path output/edpo_msmarco_8b_instruct/edpo/final/model.ckpt \
    --output_dir ./checkpoint/edpo_msmarco_8b_instruct \
    --merge_type emb