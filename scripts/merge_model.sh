python -m scripts.merge_model \
    --config_file output/edpo_msmarco_8b_v2/config.yaml \
    --checkpoint_path output/edpo_msmarco_8b_v2/checkpoints/step_500.ckpt \
    --output_dir output/edpo_msmarco_8b_v2/rerank/edpo_msmarco_8b_v2 \
    --merge_type gen

