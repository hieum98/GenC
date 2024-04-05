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
    local_rank: int,
    model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    val_dataloader: DataLoader,
    val_args: ValidationArgument,
    ):
    # We only validate on rank 0
    if local_rank != 0:
        return
    print("Validation")
    model.eval()
    emb_data = []
    for k, batch in tqdm(enumerate(val_dataloader), desc="Validation", total=len(val_dataloader)):
        if k > val_args.max_iters:
            break
        idx = batch["idx"].cpu()
        query_input_ids = batch["query_input_ids"].to(local_rank) # [batch_size, max_query_length]
        query_attention_mask = batch["query_attention_mask"].to(local_rank)
        query_prompt_length = batch["query_prompt_length"].to(local_rank)

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
        
        pos_input_ids = batch["pos_input_ids"].to(local_rank)
        bs, n_pos_per_query, _ = pos_input_ids.size()
        pos_input_ids = pos_input_ids.view(-1, pos_input_ids.size(-1)) # [batch_size * num_pos, max_pos_length]
        pos_attention_mask = batch["pos_attention_mask"].to(local_rank)
        pos_attention_mask = pos_attention_mask.view(-1, pos_attention_mask.size(-1))
        pos_prompt_length = batch["pos_prompt_length"].to(local_rank)
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
    print(metrics)
    return metrics


def fit(
    local_rank: int,
    rank: int,
    model: Union[FSDP, DDP],
    ref_model: Union[torch.nn.Module, PreTrainedModel, PeftModel],
    is_peft_model: bool,
    stage: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    autocast: Optional[torch.cuda.amp.autocast],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
    logger: Any,
    train_adapter_name: Optional[str] = "default",
):  
    # val_metric = validate(local_rank, model, val_dataloader, validation_args)
    # torch.distributed.barrier()

    optimizer: torch.optim.Optimizer = stage["optimizer"]
    scheduler : torch.optim.lr_scheduler.LambdaLR = stage["scheduler"]
    checkpoint_iter_num = stage["iter_num"]

    model.train()
    # initialize gradcache
    loss_fn = model.cons_loss_fn # The loss function for the model which is required for Gradcache
    if training_args.use_gc:
        if rank == 0:
            print("Initializing Gradcache")
        assert loss_fn is not None, "You must provide a loss function when using Gradcache"
        gc = GradCache(
            models=[model],
            chunk_sizes=training_args.gc_mini_batch_size,
            loss_fn=loss_fn,
            get_rep_fn=lambda x: x['reps'],
            fp16=training_args.precision=="16-mixed",
            scaler=scaler,
        )
        if rank == 0:
            print("Gradcache initialized")

    train_iterator = CycleIterator(train_dataloader)
    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        training_args.num_train_epochs * steps_per_epoch, 
        (training_args.max_steps or float("inf"))
        )
    iter_num = 0

    dpo_running_loss = RunningMean(window=1, sync_on_compute=False).to(local_rank)
    kl_running_loss = RunningMean(window=1, sync_on_compute=False).to(local_rank)
    cons_running_loss = RunningMean(window=1, sync_on_compute=False).to(local_rank)

    memory_stats = []
    torch.cuda.reset_peak_memory_stats(local_rank)

    while iter_num < lr_max_steps:
        iter_num += 1
        if iter_num < checkpoint_iter_num:
            continue
        iter_t0 = time.perf_counter()

        # Log memory usage
        if iter_num==0 and rank == 0:
            reserved_before_forward = torch.cuda.memory_reserved(local_rank)
            memory_stats.append(f"Rank {rank}: Before forward: {reserved_before_forward/2**30:.2f} GiB")
            logger.log({"memory/allocated_before_forward": torch.cuda.memory_allocated(local_rank)}, rank)
            logger.log({"memory/reserved_before_forward": reserved_before_forward}, rank)

        batch = next(train_iterator)

        query_input_ids = batch["query_input_ids"].to(local_rank) # [batch_size, max_length]
        query_attention_mask = batch["query_attention_mask"].to(local_rank)
        query_labels = batch["query_labels"].to(local_rank) # [batch_size]
        query_prompt_length = batch["query_prompt_length"].to(local_rank) # [batch_size]

        pos_input_ids = batch["pos_input_ids"].to(local_rank) # [batch_size, num_pos, max_length]
        pos_attention_mask = batch["pos_attention_mask"].to(local_rank)
        pos_labels = batch["pos_labels"].to(local_rank) # [batch_size, num_pos]
        pos_prompt_length = batch["pos_prompt_length"].to(local_rank) # [batch_size, num_pos]

        neg_input_ids = batch["neg_input_ids"].to(local_rank) # [batch_size, num_neg, max_length]
        neg_attention_mask = batch["neg_attention_mask"].to(local_rank)
        neg_labels = batch["neg_labels"].to(local_rank) # [batch_size, num_neg]
        neg_prompt_length = batch["neg_prompt_length"].to(local_rank) # [batch_size, num_neg]

        choices_input_ids = batch["choices_input_ids"].to(local_rank) # [batch_size, num_pos, max_length]
        choices_attention_mask = batch["choices_attention_mask"].to(local_rank)
        choices_labels = batch["choices_labels"].to(local_rank)
        choices_loss_weight_mask = batch["choices_loss_weight_mask"].to(local_rank)

        rejects_input_ids = batch["rejects_input_ids"].to(local_rank) # [batch_size, num_neg, max_length]
        rejects_attention_mask = batch["rejects_attention_mask"].to(local_rank)
        rejects_labels = batch["rejects_labels"].to(local_rank)
        rejects_loss_weight_mask = batch["rejects_loss_weight_mask"].to(local_rank)

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
            loss_emb = loss_emb
            loss_emb.detach()
        else:
            model_inputs['constrastive_labels']= passage_labels
            model_inputs['use_miner'] = training_args.use_miner
            model_inputs.update(other_kwargs)
            model_outputs = model(**model_inputs)
            reps = model_outputs['reps']
            loss_emb = model_outputs['loss_emb']
            if scaler is not None:
                scaler.scale(loss_emb).backward()
            else:
                loss_emb.backward()
        cons_running_loss.update(loss_emb.detach())

        # Forward-backward pass for DPO and KL loss
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

        # KL and DPO forward-backward pass
        chunksize = training_args.mini_batch_size
        gradient_accumulation_iters = bs // chunksize
        assert bs % chunksize == 0, "Batch size must be divisible by chunksize"
        assert gradient_accumulation_iters > 0, "Batch size must be greater than chunksize"

        inner_iter_num = 0
        for emb_input_chunk, gen_input_chunk in zip(split_input(emb_model_inputs, chunksize), split_input(gen_model_inputs, chunksize)):
            inner_iter_num += 1
            is_accumulating = iter_num % inner_iter_num != 0
            if training_args.no_sync and not is_accumulating:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                with autocast:
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
                    with torch.no_grad():
                        pair_logits = model(**pair_inputs)['logits'] # [chunksize * (1 + topk_neg), max_length, vocab_size]
                    # KL loss
                    loss_kl = kl_loss(emb_reps, pair_logits, pair_labels, pair_loss_weight_mask, bs=chunksize)

                    # DPO loss
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
                        if ref_model is None:
                            assert is_peft_model, "You must provide a reference model when not using PEFT model"
                            with null_ref_context(model=model, is_peft_model=is_peft_model, train_adapter_name=train_adapter_name):
                                all_ref_logits = model(**concat_inputs)['logits']
                        else:
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

                # Scale loss for gradient accumulation
                loss = loss_kl + dpo_losses
                loss = loss / gradient_accumulation_iters
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                kl_running_loss.update(loss_kl.detach())
                dpo_running_loss.update(dpo_losses.detach())

        # Log memory usage
        if iter_num==0 and rank == 0:
            reserved_after_forward = torch.cuda.memory_reserved(local_rank)
            memory_stats.append(f"Rank {rank}: After forward: {reserved_after_forward/2**30:.2f} GiB")
            logger.log({"memory/allocated_after_forward": torch.cuda.memory_allocated(local_rank)}, rank)
            logger.log({"memory/reserved_after_forward": reserved_after_forward}, rank)
        
        if training_args.apply_gradient_clipping and training_args.grad_norm_clip is not None:
            model.clip_grad_norm_(training_args.grad_norm_clip, norm_type=2.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
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
            
            if rank == 0:
                logger.log(metrics, rank)
                print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} |"
                f" loss train: {metrics['cons_loss']:.3f},"
                f" kl loss: {_kl_loss:.3f},"
                f" dpo loss: {_dpo_loss:.3f},"
                # f" val: {val_metric} |"
                f" lr: {metrics['learning_rate']:.2e} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                )


        # if iter_num % validation_args.interval == 0:
        #     t0 = time.perf_counter()
        #     metrics = validate(
        #         local_rank=local_rank,
        #         model=model,  
        #         val_dataloader=val_dataloader, 
        #         val_args=validation_args
        #         )
        #     val_metric = torch.tensor(metrics['R_at_1'])
        #     t1 = time.perf_counter() - t0
        #     if rank == 0:
        #         logger.log({"val_metric": val_metric.item(), "val_time": t1}, rank)
        #         print(f"Validation metric: {val_metric:.3f}")
        #     torch.distributed.barrier()

        if training_args.save_interval is not None and iter_num % training_args.save_interval == 0:
            torch.distributed.barrier()
            torch.cuda.synchronize()
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            optim_checkpoint_path = Path(training_args.output_dir) / "checkpoints" / f"step_{iter_num}" / "optimizer.pt"
            # pull all sharded optimizer states to rank0 cpu...
            optim_state = FSDP.full_optim_state_dict(model, optimizer)
            stage = {
                    "iter_num": iter_num,
                    "optimizer": optim_state,
                    "scheduler": scheduler.state_dict(),
                }
            if rank == 0:
                optim_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(stage, optim_checkpoint_path)
                print(f"Saved optimizer checkpoint to {optim_checkpoint_path}")

            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = model.state_dict()
                if rank==0:
                    save_full_path = Path(training_args.output_dir) / "checkpoints" / f"step_{iter_num}" / "model.pt"
                    print("Saving full model weights to", save_full_path)
                    torch.save(cpu_state_dict, save_full_path)

    



        

