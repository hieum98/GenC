from contextlib import nullcontext
import dataclasses
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import lightning as L
from lightning.fabric.utilities import ThroughputMonitor
from torchmetrics import RunningMean
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftModel

from genc.data.base import DataModule
from genc.trainer.trainer_utils import (
    CycleIterator, 
    get_batch_logps,
    split_input, 
    dpo_loss, 
    kl_loss,
    null_ref_context,
)
from genc.trainer.gradcache import GradCache
from genc.args import DataArguments, ModelArguments, TrainingArguments, ValidationArgument
from genc.utils import compute_metrics


@torch.no_grad()
def validate(
    fabric: L.Fabric, 
    model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    val_dataloader: DataLoader,
    val_args: ValidationArgument,
    ):
    # We only validate on rank 0
    fabric.print("Validation")
    model.eval()
    emb_data = []
    for k, batch in tqdm(enumerate(val_dataloader), desc="Validation", total=min(len(val_dataloader), val_args.max_iters)):
        if k > val_args.max_iters:
            break
        idx = batch["idx"].cpu()
        query_input_ids = batch["query_input_ids"] # [batch_size, max_query_length]
        query_attention_mask = batch["query_attention_mask"]
        query_prompt_length = batch["query_prompt_length"]

        outputs = model(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            prompt_length=query_prompt_length,
            is_emb=True,
            return_dict=True,
        )
        query_embs = outputs['reps']
        query_embs = query_embs.cpu().numpy()
        emb_data.extend([{"idx": idx[i], "embeddings": query_embs[i]} for i in range(query_embs.shape[0])])
        
        pos_input_ids = batch["pos_input_ids"]
        bs, n_pos_per_query, _ = pos_input_ids.size()
        pos_input_ids = pos_input_ids.view(-1, pos_input_ids.size(-1)) # [batch_size * num_pos, max_pos_length]
        pos_attention_mask = batch["pos_attention_mask"]
        pos_attention_mask = pos_attention_mask.view(-1, pos_attention_mask.size(-1))
        pos_prompt_length = batch["pos_prompt_length"]
        pos_prompt_length = pos_prompt_length.flatten()
        pos_embs = model(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            prompt_length=pos_prompt_length,
            is_emb=True,
        )['reps']
        pos_labels = idx.repeat_interleave(n_pos_per_query)
        # Make sure pos_labels and pos_embs are same size
        assert pos_labels.size(0) == pos_embs.size(0), f"{pos_labels.size()} != {pos_embs.size()}"

        pos_embs = pos_embs.cpu().numpy()
        emb_data.extend([{"idx": pos_labels[i].item(), "embeddings": pos_embs[i]} for i in range(pos_embs.shape[0])])
    
    labels = np.array([d["idx"] for d in emb_data])
    emb_data = np.array([d["embeddings"] for d in emb_data])
    metrics = compute_metrics(emb_data, labels)
    fabric.print(metrics)
    return metrics


def fit(
    fabric: L.Fabric,
    model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    stage: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    training_args: TrainingArguments,
    validation_args: ValidationArgument
):  
    val_metric = validate(fabric, model, val_dataloader, dataclasses.replace(validation_args, max_iters=5))
    fabric.print(f"Validation metric: {val_metric}")
    fabric.barrier()

    optimizer: torch.optim.Optimizer = stage["optimizer"]
    scheduler : torch.optim.lr_scheduler.LambdaLR = stage["scheduler"]
    checkpoint_iter_num = stage["iter_num"]

    model.train()
    # initialize gradcache
    gc = None
    if training_args.use_gc:
        fabric.print("Initializing Gradcache")
        scaler = torch.cuda.amp.GradScaler()
        def loss_fn(reps: torch.Tensor, constrastive_labels: torch.Tensor, use_miner: bool = False):
            return model(
                input_reps=reps,
                constrastive_labels=constrastive_labels,
                use_miner=use_miner,
                is_emb=True,
                )['loss_emb']
        gc = GradCache(
            models=[model],
            chunk_sizes=training_args.gc_mini_batch_size,
            loss_fn=loss_fn,
            get_rep_fn=lambda x: x['reps'],
            fp16=training_args.precision=="16-mixed",
            scaler=scaler,
            fabric=fabric,
        )

    train_iterator = CycleIterator(train_dataloader)
    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        training_args.num_train_epochs * steps_per_epoch, 
        (training_args.max_steps or float("inf"))
        )
    iter_num = 0

    gradient_accumulation_iters = training_args.batch_size(fabric.world_size) // training_args.mini_batch_size
    sft_running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    cons_running_loss = RunningMean(window=1, sync_on_compute=False).to(fabric.device)
    throughput = ThroughputMonitor(fabric, window_size=50)

    fabric.print("Training data size:", len(train_dataloader))
    while iter_num < lr_max_steps:
        iter_num += 1
        if iter_num < checkpoint_iter_num:
            continue
        iter_t0 = time.perf_counter()

        # Log memory usage
        if iter_num==0 and fabric.device==0:
            reserved_before_forward = torch.cuda.memory_reserved(fabric.device)
            fabric.log_dict({"memory/allocated_before_forward": torch.cuda.memory_allocated(fabric.device)})
            fabric.log_dict({"memory/reserved_before_forward": reserved_before_forward})

        batch = next(train_iterator)

        query_input_ids = batch["query_input_ids"] # [batch_size, max_length]
        query_attention_mask = batch["query_attention_mask"]
        query_labels = batch["query_labels"] # [batch_size]
        query_prompt_length = batch["query_prompt_length"] # [batch_size]

        pos_input_ids = batch["pos_input_ids"] # [batch_size, num_pos, max_length]
        pos_attention_mask = batch["pos_attention_mask"]
        pos_labels = batch["pos_labels"] # [batch_size, num_pos]
        pos_prompt_length = batch["pos_prompt_length"] # [batch_size, num_pos]

        neg_input_ids = batch["neg_input_ids"] # [batch_size, num_neg, max_length]
        neg_attention_mask = batch["neg_attention_mask"]
        neg_labels = batch["neg_labels"] # [batch_size, num_neg]
        neg_prompt_length = batch["neg_prompt_length"] # [batch_size, num_neg]

        choices_input_ids = batch["choices_input_ids"] # [batch_size, num_pos, max_length]
        choices_attention_mask = batch["choices_attention_mask"]
        choices_labels = batch["choices_labels"]
        choices_loss_weight_mask = batch["choices_loss_weight_mask"]

        # Forward-backward pass for contrastive loss
        bs = query_input_ids.size(0)
        num_pos = pos_input_ids.size(1)
        num_neg = neg_input_ids.size(1)
        
        passages_input_ids = torch.cat([query_input_ids, pos_input_ids.view(bs * num_pos, -1), neg_input_ids.view(bs * num_neg, -1)], dim=0)
        passages_attention_mask = torch.cat([query_attention_mask, pos_attention_mask.view(bs * num_pos, -1), neg_attention_mask.view(bs * num_neg, -1)], dim=0)
        passages_prompt_length = torch.cat([query_prompt_length, pos_prompt_length.flatten(), neg_prompt_length.flatten()], dim=0)
        passage_labels = torch.cat([query_labels, pos_labels.flatten(), neg_labels.flatten()], dim=0)
        model_inputs = {
            "input_ids": passages_input_ids,
            "attention_mask": passages_attention_mask,
            "prompt_length": passages_prompt_length,
        }
        other_kwargs = {
            "is_emb": True,
        }
        if training_args.use_gc:
            no_sync_except_last = torch.distributed.is_initialized()
            inputs = (model_inputs, other_kwargs)
            loss_emb, reps = gc(
                inputs, 
                no_sync_except_last=no_sync_except_last,
                constrastive_labels=passage_labels,
                use_miner=training_args.use_miner,
                )
            loss_emb = loss_emb / fabric.world_size
            loss_emb.detach()
        else:
            model_inputs['constrastive_labels']= passage_labels
            model_inputs['use_miner'] = training_args.use_miner
            model_inputs.update(other_kwargs)
            model_outputs = model(**model_inputs)
            reps = model_outputs['reps']
            loss_emb = model_outputs['loss_emb']
            fabric.backward(loss_emb)

        cons_running_loss.update(loss_emb.detach())

        # Forward-backward pass for SFT
        gen_model_inputs = {
            "input_ids": choices_input_ids.view(-1, choices_input_ids.size(-1)), # [batch_size * num_pos, max_length]
            "attention_mask": choices_attention_mask.view(-1, choices_attention_mask.size(-1)), # [batch_size * num_pos, max_length] 
            "labels": choices_labels.view(-1, choices_labels.size(-1)), # [batch_size * num_pos, max_length]
            "loss_weight_mask": choices_loss_weight_mask.view(-1, choices_loss_weight_mask.size(-1)), # [batch_size * num_pos, max_length]
        }

        chunksize = training_args.mini_batch_size
        gradient_accumulation_iters = bs // chunksize
        assert bs % chunksize == 0, "Batch size must be divisible by chunksize"
        assert gradient_accumulation_iters > 0, "Batch size must be greater than chunksize"

        inner_iter_num = 0
        for gen_input_chunk in split_input(gen_model_inputs, chunksize):
            inner_iter_num += 1
            is_accumulating = ((inner_iter_num % gradient_accumulation_iters) != 0)
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                # SFT loss
                chunk_inputs = {
                    "input_ids": gen_input_chunk["input_ids"],
                    "attention_mask": gen_input_chunk["attention_mask"],
                    "labels": gen_input_chunk["labels"],
                    "loss_weight_mask": gen_input_chunk["loss_weight_mask"],
                    "is_gen": True,
                }
                loss = model(**chunk_inputs)['loss']
                fabric.backward(loss)

            sft_running_loss.update(loss.detach())

        # Log memory usage
        if iter_num==0 and fabric.device==0:
            reserved_after_forward = torch.cuda.memory_reserved(fabric.device)
            fabric.log_dict({"memory/allocated_after_forward": torch.cuda.memory_allocated(fabric.device)})
            fabric.log_dict({"memory/reserved_after_forward": reserved_after_forward})
        
        if training_args.apply_gradient_clipping and training_args.grad_norm_clip is not None:
            fabric.clip_gradients(model, optimizer, max_norm=training_args.grad_norm_clip)
        
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        
        if iter_num % training_args.log_interval == 0:
            _cons_loss = cons_running_loss.compute().item()
            _sft_loss = sft_running_loss.compute().item()
            t1 = time.perf_counter()

            metrics = {
                "cons_loss": _cons_loss,
                "sft_loss": _sft_loss,
                "iter": iter_num,
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            
            fabric.log_dict(metrics, step=iter_num)
            fabric.print(
            f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} |"
            f" loss train: {metrics['cons_loss']:.3f},"
            f" sft loss: {_sft_loss:.3f},"
            # f" val: {val_metric} |"
            f" lr: {metrics['learning_rate']:.2e} |"
            f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
            )

        if iter_num % validation_args.interval == 0:
            t0 = time.perf_counter()
            val_metrics = validate(
                fabric=fabric,
                model=model,  
                val_dataloader=val_dataloader, 
                val_args=validation_args
                )
            t1 = time.perf_counter() - t0
            fabric.log_dict(val_metrics, step=iter_num)
            fabric.print(f"Validation metric: {val_metrics}")
            fabric.barrier()

        if training_args.save_interval is not None and iter_num % training_args.save_interval == 0:
            checkpoint_path = Path(training_args.output_dir) / "checkpoints" / f"step_{iter_num}.ckpt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            stage = {
                    "iter_num": iter_num,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "model": model,
                }
            fabric.save(checkpoint_path, stage)

    



        

