python -m scripts.merge_model \
    --config_file output/simcse_esft_7b/config.yaml \
    --checkpoint_path output/simcse_esft_7b/checkpoints/step_300.ckpt \
    --output_dir ./checkpoint/7b-esft_simcse-300 \
    --merge_type emb