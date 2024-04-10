import functools
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from pprint import pprint
import sys
from weakref import ref
import yaml
from dataclasses import asdict
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import bitsandbytes as bnb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import lightning as L
from lightning import seed_everything
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from transformers import get_cosine_schedule_with_warmup, PreTrainedTokenizerBase, HfArgumentParser, AutoTokenizer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from genc.data.base import DataModule
from genc.data.msmarco import MSMARCODataset, get_dataloader
from genc.trainer.lora_dpo_finetune import fit
from genc.trainer.trainer_utils import (
    choose_logger,
    get_default_supported_precision,
    get_wrapping_policy,
)
from genc.trainer.load_model import load_model
from genc.args import DataArguments, ModelArguments, TrainingArguments, ValidationArgument


def validate_and_correct_args(
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        validation_args: ValidationArgument,
        ):
    # Validate precision
    precision = training_args.precision or get_default_supported_precision(training=True)
    training_args.precision = precision

    if 'bf16' in training_args.precision and not torch.cuda.is_bf16_supported():
        raise ValueError('Current device does not support bfloat16')
    
    # Set no_sync if using cpu_offload and gradient accumulation. Turn off if not using gradient accumulation
    gradient_accumulation_iters = training_args.batch_size(training_args.devices) // training_args.mini_batch_size
    assert gradient_accumulation_iters > 0, "Batch size must be divisible by mini batch size"
    assert training_args.batch_size(training_args.devices) % training_args.mini_batch_size == 0, "Batch size must be divisible by mini batch size"
    if training_args.use_cpu_offload and gradient_accumulation_iters > 1:
        training_args.no_sync = True
    elif training_args.no_sync and gradient_accumulation_iters == 1:
        training_args.no_sync = False

    # Save the corrected args into the yaml file
    config_file = Path(training_args.output_dir) / "config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(asdict(training_args), f)
        yaml.dump(asdict(model_args), f)
        yaml.dump(asdict(data_args), f)
        yaml.dump(asdict(validation_args), f)
        
    return data_args, model_args, training_args, validation_args 

def get_dataloaders(
    fabric: L.Fabric,
    data: DataModule,
    tokenizer: PreTrainedTokenizerBase,
    training_args: TrainingArguments,       
):  
    world_size = training_args.nodes * training_args.devices 
    data.connect(
        world_size=world_size,
        global_rank=fabric.global_rank,
        tokenizer=tokenizer, 
        batch_size=training_args.batch_size(world_size), 
        max_seq_length=training_args.max_seq_length,
        mode=training_args.mode,
        num_negative_samples=training_args.num_negative_samples,
        num_positive_samples=training_args.num_positive_samples,
        prompt_loss_weight=training_args.prompt_loss_weight,
    )
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, 
        val_dataloader, 
        use_distributed_sampler=False,
        move_to_device=True,
        )
    return train_dataloader, val_dataloader

def main(
    fabric: L.Fabric,
    data: DataModule,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
    torch_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):    
    fabric.seed_everything(training_args.seed)
    # Load the model and tokenizer
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
        torch_dtype=torch_dtype,
        compute_dtype=compute_dtype,
        precision=training_args.precision,
        rank=fabric.global_rank,
        local_rank=fabric.local_rank,
        gradient_checkpointing=training_args.gradient_checkpointing,
        attn_implementation=model_args.attn_implementation,
    )
    ref_model, _ = load_model(
        model_weights_name_or_path=model_args.model_name_or_path,
        use_bidirectional=model_args.use_bidirectional,
        normalized=model_args.normalized,
        pooling_method=model_args.pooling_method,
        loss_gen_type=model_args.loss_gen_type,
        temperature=model_args.temperature,
        quantization=True,
        use_lora=False,
        inference=True,
        low_memory=training_args.low_memory,
        torch_dtype=torch_dtype,
        compute_dtype=compute_dtype,
        precision=training_args.precision,
        rank=fabric.global_rank,
        local_rank=fabric.local_rank,
        attn_implementation=model_args.attn_implementation,
    )

    # Load model checkpoint this will load the model state dict into cpu memory 
    if training_args.checkpoint_dir is not None:
        full_state_dict_model_path = Path(training_args.checkpoint_dir) / "model.ckpt"
        if isinstance(fabric.strategy, FSDPStrategy):
            fabric.load_raw(full_state_dict_model_path, model, strict=False)
        else:
            model_checkpoint = lazy_load(full_state_dict_model_path)
            model.load_state_dict(model_checkpoint, strict=False)

    fabric.log_dict({"memory/allocated_after_model_created": torch.cuda.memory_allocated(fabric.local_rank)})
    fabric.log_dict({"memory/reserved_after_model_creation": torch.cuda.memory_reserved(fabric.local_rank)})
    model = fabric.setup_module(model)
    fabric.log_dict({"memory/allocated_after_model_setup": torch.cuda.memory_allocated(fabric.local_rank)})
    fabric.log_dict({"memory/reserved_after_model_setup": torch.cuda.memory_reserved(fabric.local_rank)})

    # Load the data
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, training_args)
    # Synchronize at the start
    fabric.barrier()

    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        training_args.num_train_epochs * steps_per_epoch, 
        (training_args.max_steps or float("inf"))
        )

    # Config optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_args.learning_rate, 
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=lr_max_steps,
        )
    checkpoint_iter_num = 0
    stage = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": checkpoint_iter_num,
    }
    if training_args.checkpoint_dir is not None:
        optim_checkpoint_path = Path(training_args.checkpoint_dir) / "optimizer.ckpt"
        if optim_checkpoint_path.exists():
            fabric.load(optim_checkpoint_path, stage, strict=True)
    fit(
        fabric=fabric,
        model=model,
        ref_model=ref_model,
        stage=stage,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_args=training_args,
        validation_args=validation_args,
    )

    torch.cuda.synchronize()
    save_full_path = Path(training_args.output_dir) / "final" / "model.pt"
    print("Saving full model weights to", save_full_path)
    fabric.save(save_full_path, model)

def setup(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
):
    data_args, model_args, training_args, validation_args = validate_and_correct_args(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        validation_args=validation_args,
    )
    seed_everything(training_args.seed)

    data = MSMARCODataset(
        data_dir=data_args.data_dir,
        train_file=data_args.train_file,
        val_file=data_args.val_file,
        ignore_index=data_args.ignore_index,
        seed=training_args.seed,
        num_workers=data_args.num_workers,
    )

    # Make necessary directories
    out_dir = Path(training_args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

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
                limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
                sync_module_states=training_args.low_memory,
            )
        else:
            strategy = training_args.strategy
    
    logger_dir = Path(training_args.output_dir) / f"logs_{training_args.logger_name}"
    logger_name = f"dpoc-{model_args.model_name_or_path.split('/')[-1]}"
    logger = choose_logger(training_args.logger_name, logger_dir, name=logger_name, log_interval=training_args.log_interval)

    fabric = L.Fabric(
        accelerator='gpu',
        strategy=strategy,
        devices=training_args.devices,
        num_nodes=training_args.nodes,
        precision=training_args.precision,
        loggers=logger,
    )
    fabric.launch(
        main,
        data=data,
        model_args=model_args,
        training_args=training_args,
        validation_args=validation_args,
        torch_dtype=torch_dtype,
        compute_dtype=compute_dtype,
    )


if __name__=='__main__':
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    # os.environ['HF_HOME'] = '/mnt/hieu/hf_cache'
    # os.environ['TRANSFORMERS_CACHE'] = '/mnt/hieu/hf_cache'
    torch.set_float32_matmul_precision("high")

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments, ValidationArgument))
    logging.info(f"Sys args {sys.argv}")
    if len(sys.argv) > 0 and sys.argv[-1].endswith(".yaml"):
        # If we pass only one argument to the script, and it's the path to a yaml file,
        # let's parse it to get our arguments.
        logging.info(f"Loading yaml config {sys.argv[-1]}")
        data_args, model_args, training_args, validation_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        logging.info("No config file passed, using command line arguments.")
        data_args, model_args, training_args, validation_args = parser.parse_args_into_dataclasses()
    
    setup(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        validation_args=validation_args,
    )



