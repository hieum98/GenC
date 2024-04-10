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


def emb_forward_backward(
    fabric: L.Fabric,
    gc: GradCache,
    model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    query_labels: torch.Tensor,
    query_prompt_length: torch.Tensor,
    pos_input_ids: torch.Tensor,
    pos_attention_mask: torch.Tensor,
    pos_labels: torch.Tensor,
    pos_prompt_length: torch.Tensor,
    neg_input_ids: torch.Tensor,
    neg_attention_mask: torch.Tensor,
    neg_labels: torch.Tensor,
    neg_prompt_length: torch.Tensor,
    training_args: TrainingArguments,
    ):
    bs = query_input_ids.size(0)
    num_pos = pos_input_ids.size(1)
    num_neg = neg_input_ids.size(1)
    # Embedding forward-backward pass
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
    
    return reps, loss_emb


def online_hard_example_mining(
    reps: torch.Tensor,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    query_prompt_length: torch.Tensor,
    pos_input_ids: torch.Tensor,
    pos_attention_mask: torch.Tensor,
    pos_prompt_length: torch.Tensor,
    neg_input_ids: torch.Tensor,
    neg_attention_mask: torch.Tensor,
    neg_prompt_length: torch.Tensor,
    choices_input_ids: torch.Tensor,
    choices_attention_mask: torch.Tensor,
    choices_labels: torch.Tensor,
    choices_loss_weight_mask: torch.Tensor,
    rejects_input_ids: torch.Tensor,
    rejects_attention_mask: torch.Tensor,
    rejects_labels: torch.Tensor,
    rejects_loss_weight_mask: torch.Tensor,
    training_args: TrainingArguments,
    ):
    bs = query_input_ids.size(0)
    num_pos = pos_input_ids.size(1)
    num_neg = neg_input_ids.size(1)

    # Get the embeddings
    query_embs = reps.clone().detach()[:bs]
    pos_embs = reps.clone().detach()[bs:bs + bs * num_pos].view(bs, num_pos, -1)
    neg_embs = reps.clone().detach()[bs + bs * num_pos:].view(bs, num_neg, -1)
    # Pairwise cosine similarity
    pos_sim = torch.cosine_similarity(query_embs.unsqueeze(1), pos_embs, dim=-1) # [bs, num_pos]
    neg_sim = torch.cosine_similarity(query_embs.unsqueeze(1), neg_embs, dim=-1) # [bs, num_neg]
    # Get topk similar negatives
    _, topk_neg_sim_idx = torch.topk(neg_sim, k=training_args.topk_neg*2, dim=-1) # [bs, topk_neg]
    # Get top1 dissimilar positives
    _, top1_pos_sim_idx = torch.topk(-pos_sim, k=1, dim=-1) # [bs, 1]
    
    hard_pos_input_ids = []
    hard_pos_attention_mask = []
    hard_pos_prompt_length = []
    hard_neg_input_ids = []
    hard_neg_attention_mask = []
    hard_neg_prompt_length = []
    hard_choices_input_ids = []
    hard_choices_attention_mask = []
    hard_choices_labels = []
    hard_choices_loss_weight_mask = []
    hard_rejects_input_ids = []
    hard_rejects_attention_mask = []
    hard_rejects_labels = []
    hard_rejects_loss_weight_mask = []
    for i in range(bs):
        hard_pos_input_ids.append(pos_input_ids[i, top1_pos_sim_idx[i]]) # [1, max_length]
        hard_pos_attention_mask.append(pos_attention_mask[i, top1_pos_sim_idx[i]])
        hard_pos_prompt_length.append(pos_prompt_length[i, top1_pos_sim_idx[i]])
        hard_neg_input_ids.append(neg_input_ids[i, topk_neg_sim_idx[i]]) # [topk_neg, max_length]
        hard_neg_attention_mask.append(neg_attention_mask[i, topk_neg_sim_idx[i]])
        hard_neg_prompt_length.append(neg_prompt_length[i, topk_neg_sim_idx[i]])
        hard_choices_input_ids.append(choices_input_ids[i, top1_pos_sim_idx[i]]) # [1, max_length]
        hard_choices_attention_mask.append(choices_attention_mask[i, top1_pos_sim_idx[i]])
        hard_choices_labels.append(choices_labels[i, top1_pos_sim_idx[i]])
        hard_choices_loss_weight_mask.append(choices_loss_weight_mask[i, top1_pos_sim_idx[i]])
        hard_rejects_input_ids.append(rejects_input_ids[i, topk_neg_sim_idx[i]]) # [topk_neg, max_length]
        hard_rejects_attention_mask.append(rejects_attention_mask[i, topk_neg_sim_idx[i]])
        hard_rejects_labels.append(rejects_labels[i, topk_neg_sim_idx[i]])
        hard_rejects_loss_weight_mask.append(rejects_loss_weight_mask[i, topk_neg_sim_idx[i]])
    
    hard_pos_input_ids = torch.stack(hard_pos_input_ids, dim=0) # [bs, 1, max_length]
    hard_pos_attention_mask = torch.stack(hard_pos_attention_mask, dim=0)
    hard_pos_prompt_length = torch.stack(hard_pos_prompt_length, dim=0)
    hard_neg_input_ids = torch.stack(hard_neg_input_ids, dim=0) # [bs, topk_neg, max_length]
    hard_neg_attention_mask = torch.stack(hard_neg_attention_mask, dim=0)
    hard_neg_prompt_length = torch.stack(hard_neg_prompt_length, dim=0)
    emb_model_inputs = {
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "query_prompt_length": query_prompt_length,
        "pos_input_ids": hard_pos_input_ids,
        "pos_attention_mask": hard_pos_attention_mask,
        "pos_prompt_length": hard_pos_prompt_length,
        "neg_input_ids": hard_neg_input_ids,
        "neg_attention_mask": hard_neg_attention_mask,
        "neg_prompt_length": hard_neg_prompt_length,
    }

    hard_choices_input_ids = torch.stack(hard_choices_input_ids, dim=0) # [bs, 1, max_length]
    hard_choices_attention_mask = torch.stack(hard_choices_attention_mask, dim=0)
    hard_choices_labels = torch.stack(hard_choices_labels, dim=0)
    hard_choices_loss_weight_mask = torch.stack(hard_choices_loss_weight_mask, dim=0)
    hard_rejects_input_ids = torch.stack(hard_rejects_input_ids, dim=0) # [bs, topk_neg, max_length]
    hard_rejects_attention_mask = torch.stack(hard_rejects_attention_mask, dim=0)
    hard_rejects_labels = torch.stack(hard_rejects_labels, dim=0)
    hard_rejects_loss_weight_mask = torch.stack(hard_rejects_loss_weight_mask, dim=0)
    gen_model_inputs = {
        "choices_input_ids": hard_choices_input_ids,
        "choices_attention_mask": hard_choices_attention_mask,
        "choices_labels": hard_choices_labels,
        "choices_loss_weight_mask": hard_choices_loss_weight_mask,
        "rejects_input_ids": hard_rejects_input_ids,
        "rejects_attention_mask": hard_rejects_attention_mask,
        "rejects_labels": hard_rejects_labels,
        "rejects_loss_weight_mask": hard_rejects_loss_weight_mask,
    }
    return emb_model_inputs, gen_model_inputs


def compute_kl_loss(
    model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    emb_input_chunk: Dict[str, torch.Tensor],
    gen_input_chunk: Dict[str, torch.Tensor],
    chunksize: int,
    ):
    # KL loss
    # Compute the embeddings for the query, positive and negative samples
    emb_input_ids = torch.cat([
        emb_input_chunk["query_input_ids"], # [chunksize, max_length]
        emb_input_chunk["pos_input_ids"].view(-1, emb_input_chunk["pos_input_ids"].size(-1)), # [chunksize * 1, max_length]
        emb_input_chunk["neg_input_ids"].view(-1, emb_input_chunk["neg_input_ids"].size(-1)), # [chunksize * topk_neg, max_length]
    ], dim=0) # [chunksize * (1 + 1 + topk_neg), max_length]
    emb_attention_mask = torch.cat([
        emb_input_chunk["query_attention_mask"], 
        emb_input_chunk["pos_attention_mask"].view(-1, emb_input_chunk["pos_attention_mask"].size(-1)),
        emb_input_chunk["neg_attention_mask"].view(-1, emb_input_chunk["neg_attention_mask"].size(-1)),
    ], dim=0)
    emb_prompt_length = torch.cat([
        emb_input_chunk["query_prompt_length"], # [chunksize]
        emb_input_chunk["pos_prompt_length"].flatten(), # [chunksize * 1]
        emb_input_chunk["neg_prompt_length"].flatten(), # [chunksize * topk_neg]
    ], dim=0) # [chunksize * (1 + 1 + topk_neg)]
    emb_inputs = {
        "input_ids": emb_input_ids,
        "attention_mask": emb_attention_mask,
        "prompt_length": emb_prompt_length,
        "is_emb": True,
    }
    emb_reps = model(**emb_inputs)['reps'] # [chunksize * (1 + 1 + topk_neg), d]
    # Compute pair logits
    pair_input_ids = torch.cat([
        gen_input_chunk["choices_input_ids"].view(-1, gen_input_chunk["choices_input_ids"].size(-1)), # [chunksize, max_length]
        gen_input_chunk["rejects_input_ids"].view(-1, gen_input_chunk["rejects_input_ids"].size(-1)), # [chunksize * topk_neg, max_length]
    ], dim=0)
    pair_attention_mask = torch.cat([
        gen_input_chunk["choices_attention_mask"].view(-1, gen_input_chunk["choices_attention_mask"].size(-1)),
        gen_input_chunk["rejects_attention_mask"].view(-1, gen_input_chunk["rejects_attention_mask"].size(-1)),
    ], dim=0)
    pair_labels = torch.cat([
        gen_input_chunk["choices_labels"].view(-1, gen_input_chunk["choices_labels"].size(-1)),
        gen_input_chunk["rejects_labels"].view(-1, gen_input_chunk["rejects_labels"].size(-1)),
    ], dim=0)
    pair_loss_weight_mask = torch.cat([
        gen_input_chunk["choices_loss_weight_mask"].view(-1, gen_input_chunk["choices_loss_weight_mask"].size(-1)),
        gen_input_chunk["rejects_loss_weight_mask"].view(-1, gen_input_chunk["rejects_loss_weight_mask"].size(-1)),
    ], dim=0)
    pair_inputs = {
        "input_ids": pair_input_ids,
        "attention_mask": pair_attention_mask,
        "is_gen": True,
    }
    # TODO: change to rank score i.e, dpo_gen_logp/ref_gen_logp
    with torch.no_grad():
        pair_logits = model(**pair_inputs)['logits'] # [chunksize * (1 + topk_neg), max_length, vocab_size]
    # KL loss
    loss_kl = kl_loss(emb_reps, pair_logits, pair_labels, pair_loss_weight_mask, bs=chunksize)
    return loss_kl


def compute_dpo_loss(
    model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    ref_model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    gen_input_chunk: Dict[str, torch.Tensor],
    chunksize: int,
    training_args: TrainingArguments,
    ):
    # Get only the first choice and reject
    concat_input_ids = torch.cat([
        gen_input_chunk["choices_input_ids"][:, 0, :], # [chunksize, max_length]
        gen_input_chunk["rejects_input_ids"][:, 0, :], # [chunksize, max_length]
    ], dim=0)
    concat_attention_mask = torch.cat([
        gen_input_chunk["choices_attention_mask"][:, 0, :],
        gen_input_chunk["rejects_attention_mask"][:, 0, :],
    ], dim=0)
    concat_labels = torch.cat([
        gen_input_chunk["choices_labels"][:, 0, :],
        gen_input_chunk["rejects_labels"][:, 0, :],
    ], dim=0)
    concat_loss_weight_mask = torch.cat([
        gen_input_chunk["choices_loss_weight_mask"][:, 0, :],
        gen_input_chunk["rejects_loss_weight_mask"][:, 0, :],
    ], dim=0)
    concat_inputs = {
        "input_ids": concat_input_ids,
        "attention_mask": concat_attention_mask,
        "is_gen": True,
    }
    # The policy model forward pass
    all_policy_logits = model(**concat_inputs)['logits'] # [2 * chunksize, max_length, vocab_size]
    all_policy_logps = get_batch_logps(
        all_policy_logits, 
        concat_labels,
        average_log_prob=training_args.dpo_loss_type == "ipo",
        label_pad_token_id=-100,
        loss_weight_mask=concat_loss_weight_mask,
    )
    policy_choice_logps = all_policy_logps[:chunksize]
    policy_reject_logps = all_policy_logps[chunksize:]
    #The reference model forward pass
    with torch.no_grad():
        ref_model.eval()
        all_ref_logits = ref_model(**concat_inputs)['logits']
        all_ref_logps = get_batch_logps(
            all_ref_logits,
            concat_labels,
            average_log_prob=training_args.dpo_loss_type == "ipo",
            label_pad_token_id=-100,
            loss_weight_mask=concat_loss_weight_mask,
        )
        ref_choice_logps = all_ref_logps[:chunksize]
        ref_reject_logps = all_ref_logps[chunksize:]
    dpo_losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_choice_logps,
        policy_reject_logps,
        ref_choice_logps,
        ref_reject_logps,
        loss_type=training_args.dpo_loss_type,
        label_smoothing=training_args.label_smoothing_factor,
        beta=training_args.dpo_beta,
    )
    dpo_losses = dpo_losses.mean()


def fit(
    fabric: L.Fabric,
    model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    ref_model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
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
    loss_fn = model.cons_loss_fn # The loss function for the model which is required for Gradcache
    if training_args.use_gc:
        fabric.print("Initializing Gradcache")
        assert loss_fn is not None, "You must provide a loss function when using Gradcache"
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
    dpo_running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    kl_running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
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

        rejects_input_ids = batch["rejects_input_ids"] # [batch_size, num_neg, max_length]
        rejects_attention_mask = batch["rejects_attention_mask"]
        rejects_labels = batch["rejects_labels"]
        rejects_loss_weight_mask = batch["rejects_loss_weight_mask"]

        bs = query_input_ids.size(0)

        loss_emb, reps = emb_forward_backward(
            fabric=fabric,
            gc=gc,
            model=model,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            query_labels=query_labels,
            query_prompt_length=query_prompt_length,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            pos_labels=pos_labels,
            pos_prompt_length=pos_prompt_length,
            neg_input_ids=neg_input_ids,
            neg_attention_mask=neg_attention_mask,
            neg_labels=neg_labels,
            neg_prompt_length=neg_prompt_length,
            training_args=training_args,
        )

        cons_running_loss.update(loss_emb.detach())

        # Forward-backward pass for DPO and KL
        emb_model_inputs, gen_model_inputs = online_hard_example_mining(
            reps=reps,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            query_prompt_length=query_prompt_length,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            pos_prompt_length=pos_prompt_length,
            neg_input_ids=neg_input_ids,
            neg_attention_mask=neg_attention_mask,
            neg_prompt_length=neg_prompt_length,
            choices_input_ids=choices_input_ids,
            choices_attention_mask=choices_attention_mask,
            choices_labels=choices_labels,
            choices_loss_weight_mask=choices_loss_weight_mask,
            rejects_input_ids=rejects_input_ids,
            rejects_attention_mask=rejects_attention_mask,
            rejects_labels=rejects_labels,
            rejects_loss_weight_mask=rejects_loss_weight_mask,
            training_args=training_args,
        )

        chunksize = training_args.mini_batch_size
        gradient_accumulation_iters = bs // chunksize
        assert bs % chunksize == 0, "Batch size must be divisible by chunksize"
        assert gradient_accumulation_iters > 0, "Batch size must be greater than chunksize"

        inner_iter_num = 0
        for emb_input_chunk, gen_input_chunk in zip(split_input(emb_model_inputs, chunksize), split_input(gen_model_inputs, chunksize)):
            inner_iter_num += 1
            is_accumulating = ((inner_iter_num % gradient_accumulation_iters) != 0)
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss_kl = compute_kl_loss(
                    model=model,
                    emb_input_chunk=emb_input_chunk,
                    gen_input_chunk=gen_input_chunk,
                    chunksize=chunksize,
                )
                # DPO loss
                dpo_losses = compute_dpo_loss(
                    model=model,
                    ref_model=ref_model,
                    gen_input_chunk=gen_input_chunk,
                    chunksize=chunksize,
                    training_args=training_args,
                )

                # Scale loss for gradient accumulation
                loss = loss_kl + dpo_losses
                loss = loss / gradient_accumulation_iters
                fabric.backward(loss)

            kl_running_loss.update(loss_kl.detach())
            dpo_running_loss.update(dpo_losses.detach())

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
            _kl_loss = kl_running_loss.compute().item()
            _dpo_loss = dpo_running_loss.compute().item()
            t1 = time.perf_counter()

            metrics = {
                "cons_loss": _cons_loss,
                "kl_loss": _kl_loss,
                "dpo_loss": _dpo_loss,
                "iter": iter_num,
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            
            fabric.log_dict(metrics, step=iter_num)
            fabric.print(
            f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} |"
            f" loss train: {metrics['cons_loss']:.3f},"
            f" kl loss: {_kl_loss:.3f},"
            f" dpo loss: {_dpo_loss:.3f},"
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
            torch.cuda.synchronize()
            optim_checkpoint_path = Path(training_args.output_dir) / "checkpoints" / f"step_{iter_num}" / "optimizer.ckpt"
            stage = {
                    "iter_num": iter_num,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                }
            fabric.save(optim_checkpoint_path, stage)
            fabric.print(f"Checkpoint saved at {optim_checkpoint_path}")
            save_full_path = Path(training_args.output_dir) / "checkpoints" / f"step_{iter_num}" / "model.ckpt"
            fabric.print("Saving full model weights to", save_full_path)
            fabric.save(save_full_path, model)

    



        

