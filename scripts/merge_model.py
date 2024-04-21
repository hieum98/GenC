import argparse
import torch
from transformers import HfArgumentParser

from genc.trainer.load_model import load_model
from genc.args import DataArguments, TrainingArguments, ValidationArgument, ModelArguments


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config_file", type=str, required=True)
arg_parser.add_argument("--checkpoint_path", type=str, required=True)
arg_parser.add_argument("--output_dir", type=str, required=True)
arg_parser.add_argument("--merge_type", type=str, default='emb')
args = arg_parser.parse_args()

config_file = args.config_file
checkpoint_path = args.checkpoint_path

parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments, ValidationArgument))
data_args, model_args, training_args, validation_args = parser.parse_yaml_file(yaml_file=config_file)

model, tokenizer = load_model(
        model_weights_name_or_path=model_args.model_name_or_path,
        use_bidirectional=model_args.use_bidirectional,
        normalized=model_args.normalized,
        pooling_method=model_args.pooling_method,
        loss_gen_type=model_args.loss_gen_type,
        temperature=model_args.temperature,
        quantization=model_args.quantization,
        use_lora=model_args.use_lora,
        emb_adapter_name=model_args.emb_adapter_name,
        gen_adapter_name=model_args.gen_adapter_name,
        lora_target_modules=["all"],
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        inference=False,
        low_memory=training_args.low_memory,
        torch_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        precision=training_args.precision,
        rank=0,
        local_rank=0,
        gradient_checkpointing=training_args.gradient_checkpointing,
        attn_implementation=model_args.attn_implementation,
    )

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
if args.merge_type == 'emb':
    model = model.merge_and_unload(adapter_names=[model_args.emb_adapter_name], progressbar=True)
elif args.merge_type == 'gen':
    model = model.merge_and_unload(adapter_names=[model_args.gen_adapter_name], progressbar=True)
elif args.merge_type == 'both':
    model = model.merge_and_unload(adapter_names=[model_args.emb_adapter_name, model_args.gen_adapter_name], progressbar=True)

save_dir = args.output_dir
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

