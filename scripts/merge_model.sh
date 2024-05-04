python -m scripts.merge_model \
    --config_file output/esft_msmarco_1.5b_instruct/config.yaml \
    --checkpoint_path output/esft_msmarco_1.5b_instruct/checkpoints/step_400.ckpt \
    --output_dir ./checkpoint/esft_msmarco_1.5b_instruct \
    --merge_type emb


python -m scripts.merge_model \
    --config_file output/esft_msmarco_1.5b_instruct/config.yaml \
    --checkpoint_path output/esft_msmarco_1.5b_instruct/checkpoints/step_400.ckpt \
    --output_dir ./checkpoint/esft_msmarco_1.5b_instruct_gen \
    --merge_type gen

python -m scripts.merge_model \
    --config_file output/esft_msmarco_1.5b_instruct/config.yaml \
    --checkpoint_path output/esft_msmarco_1.5b_instruct/checkpoints/step_400.ckpt \
    --output_dir ./checkpoint/esft_msmarco_1.5b_instruct_gen_emb \
    --merge_type both
