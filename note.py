import torch
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.api import CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from transformers import HfArgumentParser
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from genc.trainer.load_model import load_model
from genc.args import DataArguments, ModelArguments, TrainingArguments, ValidationArgument
from genc.trainer.trainer_utils import (
    choose_logger,
    get_default_supported_precision,
    get_wrapping_policy,
)

config_file = "output/checkpoints/7b-esft_msmarco/config.yaml"
checkpoint_path = "output/checkpoints/7b-esft_msmarco/checkpoints/step_50.ckpt"

parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments, ValidationArgument))
data_args, model_args, training_args, validation_args = parser.parse_yaml_file(yaml_file=config_file)

mp_policy = None
if training_args.precision == "bf16-true":
    torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
elif training_args.precision == "32":
    torch_dtype, compute_dtype = torch.float32, torch.float16
elif training_args.precision == "16-mixed":
    compute_dtype, torch_dtype = torch.float16, torch.float32
    mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
elif training_args.precision == "bf16-mixed":
    compute_dtype, torch_dtype = torch.bfloat16, torch.float32
    mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
else:
    raise ValueError("Invalid precision")
strategy = training_args.strategy
if training_args.nodes > 1 or training_args.devices > 1:
    if training_args.strategy == 'fsdp':
        cpu_offload=CPUOffload(offload_params=True) if training_args.use_cpu_offload else None,
        auto_wrap_policy = get_wrapping_policy()
        # Config sharding strategy
        if training_args.sharding_strategy == "full_shard":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif training_args.sharding_strategy == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif training_args.sharding_strategy == "ddp":
            sharding_strategy = ShardingStrategy.NO_SHARD
        elif training_args.sharding_strategy == "hybrid_full_shard":
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        elif training_args.sharding_strategy == "hybrid_shard_grad_op":
            sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            raise ValueError("Invalid sharding strategy")
        strategy = FSDPStrategy(
            cpu_offload=cpu_offload,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy={MistralDecoderLayer},
            sharding_strategy=sharding_strategy,
            state_dict_type="full",
            limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
            sync_module_states=training_args.low_memory,
        )
else:
    strategy = "auto"

fabric = L.Fabric(
    accelerator='gpu',
    strategy=strategy,
    devices=training_args.devices,
    num_nodes=training_args.nodes,
    precision=training_args.precision,
    loggers=None,
)

model, tokenizer = load_model(
        model_weights_name_or_path=model_args.model_name_or_path,
        use_bidirectional=model_args.use_bidirectional,
        normalized=model_args.normalized,
        pooling_method=model_args.pooling_method,
        loss_gen_type=model_args.loss_gen_type,
        temperature=model_args.temperature,
        quantization=model_args.quantization,
        use_lora=model_args.use_lora,
        train_adapter_name=model_args.train_adapter_name,
        lora_weights_name_or_path=model_args.lora_weights_name_or_path,
        lora_target_modules=["all"],
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        inference=False,
        low_memory=training_args.low_memory,
        torch_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        precision=training_args.precision,
        rank=fabric.global_rank,
        local_rank=fabric.local_rank,
        gradient_checkpointing=training_args.gradient_checkpointing,
        attn_implementation=model_args.attn_implementation,
    )

model = fabric.setup_module(model)

stage = fabric.load(checkpoint_path)

print(stage)
