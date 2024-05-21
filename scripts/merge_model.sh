export CUDA_VISIBLE_DEVICES=0

python -m scripts.merge_model \
    --config_file output/edpo_msmarco_1.5b_instruct_new/config.yaml \
    --checkpoint_path output/edpo_msmarco_1.5b_instruct_new/checkpoints/step_450.ckpt \
    --output_dir ./checkpoint/edpo_msmarco_1.5b_instruct \
    --merge_type emb

