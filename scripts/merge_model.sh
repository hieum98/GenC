export CUDA_VISIBLE_DEVICES=0

python -m scripts.merge_model \
    --config_file output/edpo_msmarco_1.5b_instruct_single_gpu/config.yaml \
    --checkpoint_path output/edpo_msmarco_1.5b_instruct_single_gpu/checkpoints/step_450.ckpt \
    --output_dir ./checkpoint/edpo_msmarco_1.5b_instruct_single_gpu \
    --merge_type emb

python -m scripts.merge_model \
    --config_file output/edpo_msmarco_1.5b_instruct_single_gpu_joint/config.yaml \
    --checkpoint_path output/edpo_msmarco_1.5b_instruct_single_gpu_joint/checkpoints/step_450.ckpt \
    --output_dir ./checkpoint/edpo_msmarco_1.5b_instruct_single_gpu_joint \
    --merge_type emb

