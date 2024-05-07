python -m scripts.merge_model \
    --config_file output/esft_msmarco_7b/config.yaml \
    --checkpoint_path output/esft_msmarco_7b/step_500.ckpt \
    --output_dir ./checkpoint/esft_msmarco_7b_instruct \
    --merge_type emb

python -m scripts.merge_model \
    --config_file output/esft_msmarco_7b/config.yaml \
    --checkpoint_path output/esft_msmarco_7b/step_500.ckpt \
    --output_dir ./checkpoint/esft_msmarco_7b_instruct_gen \
    --merge_type gen