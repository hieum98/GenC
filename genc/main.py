import logging
import os
from pathlib import Path
from pprint import pprint
import sys
import yaml
from dataclasses import asdict
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import bitsandbytes as bnb
import torch
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from transformers import get_cosine_schedule_with_warmup, PreTrainedTokenizerBase, HfArgumentParser, AutoTokenizer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from genc.data.base import DataModule
from genc.data.msmarco import MSMARCODataset
from genc.trainer.lora_finetune import fit
from genc.trainer.trainer_utils import (
    choose_logger,
    get_default_supported_precision,
    parse_devices, 
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
    if 'bf16' in training_args.precision and  model_args.torch_dtype != 'bfloat16':
        logging.warning("You are using bf16 precision but the model is not bfloat16. We will set the model to bfloat16.")
        model_args.torch_dtype = 'bfloat16'
    if training_args in ['16-true', '16-mixed', '16'] and model_args.torch_dtype != 'float16':
        logging.warning("You are using 16-bit precision but the model is not float16. We will set the model to float16.")
        model_args.torch_dtype = 'float16'
    if training_args.precision == '32' and model_args.torch_dtype != 'float32':
        logging.warning("You are using 32-bit precision but the model is not float32. We will set the model to float32.")
        model_args.torch_dtype = 'float32'

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
    devices: int,        
):
    data.connect(
        tokenizer=tokenizer, 
        batch_size=training_args.batch_size(devices), 
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
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    return train_dataloader, val_dataloader


def main(
    fabric: L.Fabric,
    data: DataModule,
    devices: int,
    seed: int,
    out_dir: Path,
    checkpoint_dir: Path,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
    model_args: ModelArguments,
):
    fabric.seed_everything(seed)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    
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
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        use_lora=model_args.use_lora,
        train_adapter_name=model_args.train_adapter_name,
        lora_weights_name_or_path=model_args.lora_weights_name_or_path,
        lora_target_modules=["all"],
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        torch_dtype=model_args.torch_dtype,
        inference=False,
        fsdp=training_args.strategy == "fsdp",
        attn_implementation=model_args.attn_implementation,
    )
    ref_model, _ = load_model(
        model_weights_name_or_path=model_args.model_name_or_path,
        use_bidirectional=model_args.use_bidirectional,
        normalized=model_args.normalized,
        pooling_method=model_args.pooling_method,
        loss_gen_type=model_args.loss_gen_type,
        temperature=model_args.temperature,
        quantization=model_args.quantization,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        use_lora=False,
        torch_dtype=model_args.torch_dtype,
        inference=False,
        fsdp=training_args.strategy == "fsdp",
        attn_implementation=model_args.attn_implementation,
    )
    model = fabric.setup_module(model)

    # Load the data
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, training_args, devices)
    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        training_args.num_train_epochs * steps_per_epoch, 
        (training_args.max_steps or float("inf"))
        )

    # Config optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        optimizer_cls = bnb.optim.PagedAdamW
    else:
        optimizer_cls = bnb.optim.PagedAdamW
    optimizer = optimizer_cls(
        trainable_params, 
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
    iter_num = 0
    step_count = 0
    epoch = 0
    stage = {
        "iter_num": iter_num,
        "step_count": step_count,
        "epoch": epoch,
        "optimizer": optimizer,
        "scheduler": scheduler,
        }
    
    # Load checkpoint
    # Here we only load checkpoint for the optimizer and scheduler as the model is already loaded
    optim_checkpoint_path = checkpoint_dir / "optim.ckpt"
    if optim_checkpoint_path.exists():
        fabric.load(optim_checkpoint_path, stage)

    train_time = time.perf_counter()
    fit(
        fabric=fabric,
        model=model,
        ref_model=ref_model,
        is_peft_model=model_args.use_lora,
        tokenizer=tokenizer,
        stage=stage,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        devices=devices,
        checkpoint_dir=checkpoint_dir,
        training_args=training_args,
        validation_args=validation_args,
        train_adapter_name=model_args.train_adapter_name,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final model
    if fabric.global_rank == 0:
        model_save_dir = out_dir / "final" / "model"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)
        fabric.print(f"Model saved at {model_save_dir}")


def setup(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
):
    pprint(locals())
    data_args, model_args, training_args, validation_args  = validate_and_correct_args(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        validation_args=validation_args,
    )

    # Load data
    data = MSMARCODataset(
        data_dir=data_args.data_dir,
        train_file=data_args.train_file,
        val_file=data_args.val_file,
        ignore_index=data_args.ignore_index,
        seed=training_args.seed,
        num_workers=data_args.num_workers,
    )
    
    devices = parse_devices(training_args.devices)

    checkpoint_dir = Path(training_args.output_dir)
    out_dir = checkpoint_dir/"trained_model"

    logger_dir = checkpoint_dir / f"logs_{training_args.logger_name}"
    logger_name = f"finetune-{model_args.model_name_or_path.split('/')[-1]}"
    logger = choose_logger(training_args.logger_name, logger_dir, name=logger_name, log_interval=training_args.log_interval)

    precision = training_args.precision
    if model_args.quantization != None:
        logging.warning("Model is quantized when loading, so ignore the precision setting")
        precision = None
        plugins = None
    else:
        if training_args.quantize:
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(training_args.quantize, dtype)
            precision = None
    
    if devices > 1:
        if training_args.strategy == "fsdp":
            strategy = FSDPStrategy(
                auto_wrap_policy={MistralDecoderLayer},
                activation_checkpointing_policy={MistralDecoderLayer},
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
        else:
            strategy = training_args.strategy
    else:
        strategy = "auto"
    
    fabric = L.Fabric(num_nodes=training_args.nodes, devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins,)
    seed = training_args.seed
    fabric.launch(main, data, devices, seed, out_dir, checkpoint_dir, training_args, validation_args, model_args)


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



