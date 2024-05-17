export CUDA_VISIBLE_DEVICES=2

python -m scripts.merge_model \
    --config_file output/edpo_msmarco_8b_instruct/config.yaml \
    --checkpoint_path output/edpo_msmarco_8b_instruct/checkpoints/step_150.ckpt \
    --output_dir ./checkpoint/edpo_msmarco_8b_instruct \
    --merge_type emb


python -m scripts.merge_model \
    --config_file output/edpo_msmarco_8b_instruct/config.yaml \
    --checkpoint_path output/edpo_msmarco_8b_instruct/checkpoints/step_150.ckpt \
    --output_dir ./checkpoint/edpo_msmarco_8b_instruct_gen \
    --merge_type gen


python -m scripts.merge_model \
    --config_file output/edpo_msmarco_8b_instruct_joint/config.yaml \
    --checkpoint_path output/edpo_msmarco_8b_instruct_joint/checkpoints/step_150.ckpt \
    --output_dir ./checkpoint/edpo_msmarco_8b_instruct_joint \
    --merge_type emb
