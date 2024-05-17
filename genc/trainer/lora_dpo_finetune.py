import dataclasses
import logging
from random import choice
import time
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader
import lightning as L
from torchmetrics import RunningMean
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from genc.model.lora_genc import LoRaGenc
from genc.trainer.trainer_utils import (
    CycleIterator, 
    get_batch_logps,
    chunked_cross_entropy,
    split_input, 
    dpo_loss, 
    online_hard_example_mining
)
from genc.trainer.gradcache import GradCache
from genc.args import ModelArguments, TrainingArguments, ValidationArgument
from genc.utils import compute_metrics


@torch.no_grad()
def validate(
    fabric: L.Fabric, 
    model: LoRaGenc,
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


def compute_kl_loss(
    model: LoRaGenc,
    emb_input_chunk: Dict[str, torch.Tensor],
    gen_input_chunk: Dict[str, torch.Tensor],
    chunksize: int,
    emb_adapter_name: str = "emb",
    gen_adapter_name: str = "gen",
    temperature: float = 0.05,
    ):
    # KL loss
    # Compute the embeddings for the query, positive and negative samples
    emb_input_ids = torch.cat([
        emb_input_chunk["query_input_ids"], # [chunksize, max_length]
        emb_input_chunk["pos_input_ids"].view(-1, emb_input_chunk["pos_input_ids"].size(-1)), # [chunksize * 1, max_length]
        emb_input_chunk["neg_input_ids"].view(-1, emb_input_chunk["neg_input_ids"].size(-1)), # [chunksize * topk_neg, max_length]
    ], dim=0) # [chunksize + chunksize + chunksize*topk_neg, max_length]
    emb_attention_mask = torch.cat([
        emb_input_chunk["query_attention_mask"], 
        emb_input_chunk["pos_attention_mask"].view(-1, emb_input_chunk["pos_attention_mask"].size(-1)),
        emb_input_chunk["neg_attention_mask"].view(-1, emb_input_chunk["neg_attention_mask"].size(-1)),
    ], dim=0)
    emb_prompt_length = torch.cat([
        emb_input_chunk["query_prompt_length"], # [chunksize]
        emb_input_chunk["pos_prompt_length"].flatten(), # [chunksize * 1]
        emb_input_chunk["neg_prompt_length"].flatten(), # [chunksize * topk_neg]
    ], dim=0) # [chunksize + chunksize + chunksize*topk_neg]
    emb_inputs = {
        "input_ids": emb_input_ids,
        "attention_mask": emb_attention_mask,
        "prompt_length": emb_prompt_length,
        "is_emb": True,
    }
    model.set_adapter(emb_adapter_name)
    emb_reps = model(**emb_inputs)['reps'] # [chunksize + chunksize + chunksize*topk_neg, d]
    query_reps = emb_reps[:chunksize] # [chunksize, emb_dim]
    pos_reps = emb_reps[chunksize:chunksize*2] # [chunksize, emb_dim]
    neg_reps = emb_reps[chunksize*2:] # [chunksize * topk_neg, emb_dim]
    neg_reps = neg_reps.view(chunksize, -1, neg_reps.size(-1)) # [chunksize, topk_neg, emb_dim]
    passage_reps = torch.cat([pos_reps.unsqueeze(1), neg_reps], dim=1) # [chunksize, 1 + topk_neg, emb_dim]
    dual_score = torch.cosine_similarity(query_reps.unsqueeze(1), passage_reps, dim=-1) # [chunksize, 1 + topk_neg]
    dual_score = torch.log_softmax(dual_score, dim=-1) # [chunksize, 1 + topk_neg]

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
    ], dim=0) # [chunksize + chunksize*topk_neg, max_length]
    pair_inputs = {
        "input_ids": pair_input_ids,
        "attention_mask": pair_attention_mask,
        "is_gen": True,
    }
    model.set_adapter(gen_adapter_name)
    with torch.no_grad():
        gen_logits = model(**pair_inputs)['logits'] # [chunksize + chunksize*topk_neg, max_length, vocab_size]
        gen_loss = []
        for i in range(gen_logits.size(0)):
            l = chunked_cross_entropy(
                logits=gen_logits[i],
                targets=pair_labels[i],
                loss_weight_mask=pair_loss_weight_mask[i],
            )
            gen_loss.append(l) 
        # log likelihood
        gen_score = -1 * torch.stack(gen_loss, dim=0) # [chunksize + chunksize*topk_neg]
        gen_pos_score = gen_score[:chunksize] # [chunksize]
        gen_neg_score = gen_score[chunksize:] # [chunksize * topk_neg]
        gen_neg_score = gen_neg_score.view(chunksize, -1) # [chunksize, topk_neg]
        gen_score = torch.cat([gen_pos_score.unsqueeze(1), gen_neg_score], dim=1) # [chunksize, 1 + topk_neg]
        gen_score = torch.softmax(gen_score, dim=-1) # [chunksize, 1 + topk_neg]
    model.set_adapter(emb_adapter_name)
    
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    kl_loss = kl(dual_score, gen_score)

    return kl_loss


def compute_dpo_loss(
    model: LoRaGenc,
    ref_model: LoRaGenc,
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
    choice_logits = all_policy_logits[:chunksize].clone() # [chunksize, max_length, vocab_size]

    # Compute the DPO loss
    all_policy_logps = get_batch_logps(
        all_policy_logits, 
        concat_labels,
        average_log_prob=training_args.dpo_loss_type == "ipo",
        label_pad_token_id=-100,
        loss_weight_mask=concat_loss_weight_mask,
    )
    policy_choice_logps = all_policy_logps[:chunksize]
    policy_reject_logps = all_policy_logps[chunksize:]

    # nll loss
    choice_labels = concat_labels[:chunksize].clone() # [chunksize, max_length]
    choice_loss_weight_mask = concat_loss_weight_mask[:chunksize].clone()
    nll_loss = chunked_cross_entropy(
        logits=choice_logits,
        targets=choice_labels,
        loss_weight_mask=choice_loss_weight_mask,
    )

    #The reference model forward pass
    with torch.no_grad():
        if ref_model is not None:
            all_ref_logits = ref_model(**concat_inputs)['logits']
        else:
            model.disable_adapter()
            all_ref_logits = model(**concat_inputs)['logits']
            model.enable_adapter()

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
    loss = dpo_losses.mean() + nll_loss
    return loss


def fit(
    fabric: L.Fabric,
    model: LoRaGenc,
    tokenizer: PreTrainedTokenizer,
    ref_model: LoRaGenc,
    stage: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    validation_args: ValidationArgument,
    use_online_hard_example_mining: bool = False,
):  
    model.set_adapter(model_args.emb_adapter_name)
    
    optimizer: torch.optim.Optimizer = stage["optimizer"]
    scheduler : torch.optim.lr_scheduler.LambdaLR = stage["scheduler"]
    checkpoint_iter_num = stage["iter_num"]

    model.train()
    # initialize gradcache
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
    dpo_running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    kl_running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    cons_running_loss = RunningMean(window=1, sync_on_compute=False).to(fabric.device)

    fabric.print("Training data size:", len(train_dataloader))
    refresh_sampler = False
    fabric.print("Start training with batch size:", training_args.batch_size(fabric.world_size))
    while iter_num < lr_max_steps:
        iter_num += 1
        if iter_num < checkpoint_iter_num:
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
            fabric.print("Batch size:", batch["query_input_ids"].size(0))

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
        
        if use_online_hard_example_mining:
            model.set_adapter(model_args.emb_adapter_name)
            with torch.no_grad():
                inputs = (model_inputs, other_kwargs)
                reps = gc.get_reps_only(inputs, chunksize=min(bs, 32))
        else:
            reps = None

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
            topk_neg=training_args.topk_neg,
        )

        chunksize = training_args.mini_batch_size
        gradient_accumulation_iters = bs // chunksize
        # assert bs % chunksize == 0, "Batch size must be divisible by chunksize"
        assert gradient_accumulation_iters > 0, "Batch size must be greater than chunksize"

        inner_iter_num = 0
        for emb_input_chunk, gen_input_chunk in zip(split_input(emb_model_inputs, chunksize), split_input(gen_model_inputs, chunksize)):
            inner_iter_num += 1
            is_accumulating = ((inner_iter_num % gradient_accumulation_iters) != 0)
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                # KL loss
                model.set_adapter(model_args.emb_adapter_name)
                loss_kl = compute_kl_loss(
                    model=model,
                    emb_input_chunk=emb_input_chunk,
                    gen_input_chunk=gen_input_chunk,
                    chunksize=chunksize,
                    emb_adapter_name=model_args.emb_adapter_name,
                    gen_adapter_name=model_args.gen_adapter_name,
                    temperature=model_args.temperature,
                )

                # DPO loss
                model.set_adapter(model_args.gen_adapter_name)
                dpo_losses = compute_dpo_loss(
                    model=model,
                    ref_model=ref_model,
                    gen_input_chunk=gen_input_chunk,
                    chunksize=chunksize,
                    training_args=training_args,
                )
                loss = loss_kl * training_args.kl_loss_weight  + dpo_losses * training_args.gen_loss_weight
                # Scaling with gradient accumulation
                loss = loss / gradient_accumulation_iters
                fabric.backward(loss)

            kl_running_loss.update(loss_kl.detach())
            dpo_running_loss.update(dpo_losses.detach())

        if training_args.apply_gradient_clipping and training_args.grad_norm_clip is not None:
            fabric.clip_gradients(model, optimizer, max_norm=training_args.grad_norm_clip)
        optimizer.step()
        optimizer.zero_grad()
        fabric.barrier()

        # Forward-backward pass for contrastive loss
        model.set_adapter(model_args.emb_adapter_name)
        if training_args.use_gc:
            no_sync_except_last = torch.distributed.is_initialized()
            inputs = (model_inputs, other_kwargs)
            loss_emb = gc(
                inputs, 
                no_sync_except_last=no_sync_except_last,
                constrastive_labels=passage_labels,
                use_miner=training_args.use_miner,
                )
            loss_emb = loss_emb / fabric.world_size
            loss_emb.detach()
        else:
            query_inputs = {
                "input_ids": query_input_ids,
                "attention_mask": query_attention_mask,
                "prompt_length": query_prompt_length,
                "constrastive_labels": query_labels,
            }
            pos_inputs = {
                "input_ids": pos_input_ids,
                "attention_mask": pos_attention_mask,
                "prompt_length": pos_prompt_length,
                "constrastive_labels": pos_labels,
            }
            neg_inputs = {
                "input_ids": neg_input_ids,
                "attention_mask": neg_attention_mask,
                "prompt_length": neg_prompt_length,
                "constrastive_labels": neg_labels,
            }
            chunksize = training_args.mini_batch_size
            gradient_accumulation_iters = bs // chunksize
            assert bs % chunksize == 0, "Batch size must be divisible by chunksize"
            assert gradient_accumulation_iters > 0, "Batch size must be greater than chunksize"

            inner_iter_num = 0
            model.set_adapter(model_args.emb_adapter_name)
            for q_input_chunk, p_input_chunk, n_input_chunk in zip(split_input(query_inputs, chunksize), split_input(pos_inputs, chunksize), split_input(neg_inputs, chunksize)):
                inner_iter_num += 1
                is_accumulating = ((inner_iter_num % gradient_accumulation_iters) != 0)
                with fabric.no_backward_sync(model, enabled=is_accumulating):
                    max_len = q_input_chunk["input_ids"].size(-1)
                    input_chunk = {
                        "input_ids": torch.cat([q_input_chunk["input_ids"], p_input_chunk["input_ids"].view(-1, max_len), n_input_chunk["input_ids"].view(-1, max_len)], dim=0),
                        "attention_mask": torch.cat([q_input_chunk["attention_mask"], p_input_chunk["attention_mask"].view(-1, max_len), n_input_chunk["attention_mask"].view(-1, max_len)], dim=0),
                        "prompt_length": torch.cat([q_input_chunk["prompt_length"], p_input_chunk["prompt_length"].flatten(), n_input_chunk["prompt_length"].flatten()], dim=0),
                        "constrastive_labels": torch.cat([q_input_chunk["constrastive_labels"], p_input_chunk["constrastive_labels"].flatten(), n_input_chunk["constrastive_labels"].flatten()], dim=0),
                        "use_miner": training_args.use_miner,
                        "is_emb": True,
                    }
                    model_outputs = model(**input_chunk)
                    loss_emb = model_outputs['loss_emb']
                    # Scaling with gradient accumulation
                    fabric.backward(loss_emb/gradient_accumulation_iters)
        cons_running_loss.update(loss_emb.detach())

        if training_args.apply_gradient_clipping and training_args.grad_norm_clip is not None:
            fabric.clip_gradients(model, optimizer, max_norm=training_args.grad_norm_clip)
        # Update the model with the accumulated gradients for the contrastive loss
        # for embed_weigh and lm_head only uppdate the params corresponding to the trainable tokens
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        fabric.barrier()
        
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
            f" cons loss: {metrics['cons_loss']:.3f},"
            f" kl loss: {_kl_loss:.3f},"
            f" dpo loss: {_dpo_loss:.3f},"
            # f" val: {val_metric} |"
            f" lr: {metrics['learning_rate']:.2e} |"
            f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
            )

        if iter_num % validation_args.interval == 0:
            t0 = time.perf_counter()
            model.set_adapter(model_args.emb_adapter_name)
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
            def lora_filter(key: str, value: Any) -> bool:
                return "lora_" in key or 'lm_head' in key or 'embed_tokens' in key
            fabric.save(checkpoint_path, stage, filter={"model": lora_filter})

    



        

