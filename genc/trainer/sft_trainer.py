from pathlib import Path
import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import lightning as L
from torchmetrics import RunningMean

from genc.args import ModelArguments, TrainingArguments, ValidationArgument
from genc.model.lora_genc import LoRaGenc
from genc.trainer.trainer_utils import CycleIterator, split_input


def fit(
    fabric: L.Fabric,
    model: LoRaGenc,
    stage: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
):
    optimizer: torch.optim.Optimizer = stage["optimizer"]
    scheduler : torch.optim.lr_scheduler.LambdaLR = stage["scheduler"]
    checkpoint_iter_num = stage["iter_num"]

    model.train()

    train_iterator = CycleIterator(train_dataloader)
    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        training_args.num_train_epochs * steps_per_epoch, 
        (training_args.max_steps or float("inf"))
        )
    iter_num = 0

    gradient_accumulation_iters = training_args.batch_size(fabric.world_size) // training_args.mini_batch_size
    sft_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)

    fabric.print("Training data size:", len(train_dataloader))
    refresh_sampler = False
    fabric.print("Start training with batch size:", training_args.batch_size(fabric.world_size))

    while iter_num < lr_max_steps:
        iter_num += 1
        if iter_num <= checkpoint_iter_num:
            continue
        
        if refresh_sampler:
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(train_iterator.epoch)
            refresh_sampler = False
        if iter_num % steps_per_epoch == 0:
            refresh_sampler = True

        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        if iter_num == 1:
            size_info = {k: v.size() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            fabric.print("First batch data: {}".format(size_info))

        input_ids = batch["choices_input_ids"] # [batch_size, num_pos, max_length]
        attention_mask = batch["choices_attention_mask"]
        labels = batch["choices_labels"]
        loss_weight_mask = batch["choices_loss_weight_mask"]
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weight_mask": loss_weight_mask,
        }
        minibatch_input = split_input(model_inputs, training_args.mini_batch_size)
        accumulated_flags = [True for _ in range(len(minibatch_input)-1)] + [False]
        gradient_accumulation_iters = len(minibatch_input)
        for flag, input_batch in zip(accumulated_flags, minibatch_input):
            with fabric.no_backward_sync(model, enabled=flag):
                output = model(
                    input_ids=input_batch["input_ids"].view(-1, input_batch["input_ids"].size(-1)), # [batch_size*num_pos, max_length]
                    attention_mask=input_batch["attention_mask"].view(-1, input_batch["attention_mask"].size(-1)), # [batch_size*num_pos, max_length]
                    labels=input_batch["labels"].view(-1, input_batch["labels"].size(-1)), # [batch_size*num_pos, max_length]
                    loss_weight_mask=input_batch["loss_weight_mask"].view(-1, input_batch["loss_weight_mask"].size(-1)), # [batch_size*num_pos, max_length]
                    is_gen=True,
                    adapter_name=model_args.gen_adapter_name,
                )
                loss = output['loss']
                fabric.backward(loss/gradient_accumulation_iters)
                sft_loss.update(loss.detach())
        
        if training_args.apply_gradient_clipping and training_args.grad_norm_clip is not None:
            fabric.clip_gradients(model, optimizer, max_norm=training_args.grad_norm_clip)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        if iter_num % training_args.log_interval == 0:
            loss = sft_loss.compute().item()
            t1 = time.perf_counter()

            metrics = {
                "sft_loss": loss,
                "iter": iter_num,
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            
            fabric.log_dict(metrics, step=iter_num)
            fabric.print(
            f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} |"
            f" sft_loss: {metrics['sft_loss']:.3f},"
            # f" val: {val_metric} |"
            f" lr: {metrics['learning_rate']:.2e} |"
            f" iter time: {metrics['iter_time']:.2f} s"
            )
            
        if training_args.save_interval is not None and iter_num % training_args.save_interval == 0:
            checkpoint_path = Path(training_args.output_dir) / "checkpoints" / f"step_{iter_num}.ckpt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            stage = {
                    "iter_num": iter_num,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "model": model,
                }
            def lora_filter(key: str, value: Any) -> bool:
                return "lora_" in key
            fabric.save(checkpoint_path, stage, filter={"model": lora_filter})

