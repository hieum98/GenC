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
from lightning import seed_everything
from transformers import get_cosine_schedule_with_warmup, PreTrainedTokenizerBase, HfArgumentParser, AutoTokenizer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from genc.data.msmarco import get_dataloader
from genc.trainer.lora_finetune import fit
from genc.trainer.trainer_utils import (
    Logger,
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


def main(
    local_rank: int,
    world_size: int,
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
):
    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = training_args.master_addr
    os.environ['MASTER_PORT'] = training_args.master_port
    if 'SLURM_PROCID' in os.environ:
        # assumes same number of GPUs per node.
        rank = int(os.environ['SLURM_PROCID']) * torch.cuda.device_count() + local_rank
    else:
        rank = local_rank
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # Setup logger
    # Flat all arguments into a dictionary
    logger = Logger(
        args={
            **asdict(data_args),
            **asdict(model_args),
            **asdict(training_args),
            **asdict(validation_args),
        }, 
        log_to=training_args.logger_name,
        project_name="genc",
        rank=rank,
    )

    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype. The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses `torch.amp.autocast` to control precision.
    # limited qlora testing shows that fp16 only works with autocast while bf16 trains with both pure and autocast modes.
    # TODO: test how often this holds for mp_fp16
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
    
    # Load the model and tokenizer
    # with fabric.init_module(empty_init=(devices > 1)): # Not use this with transformers because it will cause error due to the weight of pretrained model need to load in
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
        rank=rank,
        local_rank=local_rank,
        _attn_implementation=model_args.attn_implementation,
    )

    # Load model checkpoint this will load the model state dict into cpu memory 
    if training_args.checkpoint_dir is not None:
        full_state_dict_model_path = Path(training_args.checkpoint_dir) / "model.pt"
        model_checkpoint = torch.load(full_state_dict_model_path)
        model.load_state_dict(model_checkpoint)

    logger.log({"memory/allocated_after_model_created": torch.cuda.memory_allocated(local_rank)}, rank)
    logger.log({"memory/reserved_after_model_creation": torch.cuda.memory_reserved(local_rank)}, rank)
    # Wrap model with llama-recipies policy and then wrap with FSDP
    my_auto_wrap_policy = get_wrapping_policy()
    if rank == 0:
        print("Wrapping model w/ FSDP", rank)
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
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        # backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=True) if training_args.use_cpu_offload else None,
        limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=training_args.low_memory,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if (rank!=0 and training_args.low_memory) else None, # TODO note about meta device and why we need this
        mixed_precision=mp_policy,
    )
    if rank == 0:
        print(f"Rank {rank}: Wrapped model: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")
        logger.log({"memory/allocated_after_model_wrap": torch.cuda.memory_allocated(local_rank)}, rank)
        logger.log({"memory/reserved_after_model_wrap": torch.cuda.memory_reserved(local_rank)}, rank)

    # Load the data
    # Load the data
    train_data_file = Path(data_args.data_dir) / data_args.train_file
    train_dataloader = get_dataloader(
        data_files=train_data_file,
        tokenizer=tokenizer,
        is_train=True,
        mode=training_args.mode,
        max_seq_length=training_args.max_seq_length,
        num_negative_samples=training_args.num_negative_samples,
        num_positive_samples=training_args.num_positive_samples,
        prompt_loss_weight=training_args.prompt_loss_weight,
        batch_size=training_args.batch_size(training_args.devices),
        num_workers=os.cpu_count()//2 if os.cpu_count() > 10 else 10,
        seed=training_args.seed,
    )
    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        training_args.num_train_epochs * steps_per_epoch, 
        (training_args.max_steps or float("inf"))
        )
    val_data_file = Path(data_args.data_dir) / data_args.val_file
    val_dataloader = get_dataloader(
        data_files=val_data_file,
        tokenizer=tokenizer,
        is_train=False,
    )
    # Synchronize at the start
    dist.barrier()

    # Apply activation checkpointing
    if training_args.gradient_checkpointing:
        if training_args.reentrant_checkpointing:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT if training_args.reentrant_checkpointing else CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, MistralDecoderLayer)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if training_args.use_activation_cpu_offload:
        model = offload_wrapper(model)

    # Config optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_args.learning_rate, 
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=lr_max_steps,
        )
    checkpoint_iter_num = 0
    if training_args.checkpoint_dir is not None:
        optim_checkpoint_path = Path(training_args.checkpoint_dir) / "optimizer.pt"
        if optim_checkpoint_path.exists():
            checkpoint_state = torch.load(optim_checkpoint_path)
            scheduler_state_dict = checkpoint_state["scheduler"]
            scheduler.load_state_dict(scheduler_state_dict)
            checkpoint_iter_num = checkpoint_state["iter_num"]
            optimizer_state_dict = None
            if rank == 0:
                optimizer_state_dict = checkpoint_state["optimizer"]
            sharded_optimizer_state_dict = FSDP.scatter_full_optim_state_dict(optimizer_state_dict, model)
            optimizer.load_state_dict(sharded_optimizer_state_dict)
    stage = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": checkpoint_iter_num,
    }
    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                print(f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}")
    
    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if training_args.precision in ["16-mixed", "bf16-mixed"]:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if training_args.precision == "fp16_autocast" else None

    init_start_event.record()
    fit(
        local_rank=local_rank,
        rank=rank,
        model=model,
        ref_model=None,
        is_peft_model=model_args.use_lora,
        stage=stage,
        scaler=scaler,
        autocast=autocast,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_args=training_args,
        validation_args=validation_args,
        logger=logger,
        train_adapter_name=model_args.train_adapter_name,
    )
    # Synchronize at the end and record time
    init_end_event.record()
    dist.barrier()
    torch.cuda.synchronize()
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
        if rank==0:
            save_full_path = Path(training_args.output_dir) / "final" / "model.pt"
            print("Saving full model weights to", save_full_path)
            torch.save(cpu_state_dict, save_full_path)

def setup(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
):
    # Set world size
    if training_args.devices == -1:
        world_size = torch.cuda.device_count()
        training_args.devices = world_size
    print(f"World size: {training_args.devices}")
    data_args, model_args, training_args, validation_args = validate_and_correct_args(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        validation_args=validation_args,
    )
    seed_everything(training_args.seed)

    # Make necessary directories
    out_dir = Path(training_args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Run
    mp.spawn(
        main,
        args=(training_args.devices, data_args, model_args, training_args, validation_args),
        nprocs=training_args.devices,
        join=True,
    )


if __name__=='__main__':
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
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



