from collections import UserDict
from contextlib import nullcontext
import os
from collections.abc import Mapping
from functools import partial
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast
import typing

import torch
from torch.utils.data import DataLoader
import lightning as L
from torchmetrics import RunningMean

from genc.args import ModelArguments, TrainingArguments, ValidationArgument
from genc.model.lora_genc import LoRaGenc
from genc.trainer.trainer_utils import CycleIterator, RandContext, chunked_cross_entropy, dpo_loss, get_batch_logps, split_input


class GradCacheTrainer:
    def __init__(
        self,
        fabric: L.Fabric,
        chunk_size: Optional[int] = 1,
        ) -> None:
        self.fabric = fabric
        self.chunk_size: int = chunk_size
    
    def get_input_tensors(self, model_input) -> List[torch.Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, torch.Tensor):
            return [model_input]
        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])
        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])
        else:
            return []
    
    def forward_no_grad(
        self,
        model: LoRaGenc,
        model_inputs: List[Dict[str, torch.Tensor]],
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ):
        with torch.no_grad():
            rnd_state = RandContext(*self.get_input_tensors(model_inputs))

            query_input_ids = model_inputs["query_input_ids"] # [batch_size, max_length]
            query_attention_mask = model_inputs["query_attention_mask"]
            query_prompt_length = model_inputs["query_prompt_length"] # [batch_size]

            pos_input_ids = model_inputs["pos_input_ids"] # [batch_size, num_pos, max_length]
            pos_attention_mask = model_inputs["pos_attention_mask"]
            pos_prompt_length = model_inputs["pos_prompt_length"] # [batch_size, num_pos]

            neg_input_ids = model_inputs["neg_input_ids"] # [batch_size, num_neg, max_length]
            neg_attention_mask = model_inputs["neg_attention_mask"]
            neg_prompt_length = model_inputs["neg_prompt_length"] # [batch_size, num_neg]

            bs = query_input_ids.size(0)
            num_pos = pos_input_ids.size(1)
            num_neg = neg_input_ids.size(1)

            passages_input_ids = torch.cat([query_input_ids, pos_input_ids.view(bs * num_pos, -1), neg_input_ids.view(bs * num_neg, -1)], dim=0) # [bs + bs * num_pos + bs * num_neg, max_length]
            passages_attention_mask = torch.cat([query_attention_mask, pos_attention_mask.view(bs * num_pos, -1), neg_attention_mask.view(bs * num_neg, -1)], dim=0)
            passages_prompt_length = torch.cat([query_prompt_length, pos_prompt_length.flatten(), neg_prompt_length.flatten()], dim=0)
            retriever_reps = model(
                input_ids=passages_input_ids,
                attention_mask=passages_attention_mask,
                prompt_length=passages_prompt_length,
                is_emb=True,
                use_miner=training_args.use_miner,
                adapter_name=model_args.emb_adapter_name,
            )['reps'] # [bs + bs * num_pos + bs * num_neg, hidden_size]

            query_reps = retriever_reps[:bs] # [bs, hidden_size]
            pos_reps = retriever_reps[bs: bs + bs * num_pos].view(bs, num_pos, -1) # [bs, num_pos, hidden_size]
            neg_reps = retriever_reps[bs + bs * num_pos:].view(bs, num_neg, -1) # [bs, num_neg, hidden_size]
        return query_reps, pos_reps, neg_reps, rnd_state
    
    def compute_cons_loss_from_reps(
        self,
        model: LoRaGenc,
        reps: torch.Tensor, # [batch_size, 1 + num_pos + num_neg, hidden_size]
        query_labels: torch.Tensor, # [batch_size]
        pos_labels: torch.Tensor, # [batch_size, num_pos]
        neg_labels: torch.Tensor, # [batch_size, num_neg]
        model_args: ModelArguments,
        training_args: TrainingArguments,
        ):
        """
        Compute the loss from the hidden states of the model.
        """
        passage_reps = reps.view(-1, reps.size(-1)) # [batch_size * (1 + num_pos + num_neg), hidden_size]
        passage_labels = torch.cat([
            query_labels.unsqueeze(1),
            pos_labels,
            neg_labels,
        ], dim=1).flatten() # [batch_size * (1 + num_pos + num_neg)]
        retriever_output = model(
            input_reps=passage_reps,
            constrastive_labels=passage_labels,
            is_emb=True,
            use_miner=training_args.use_miner,
            adapter_name=model_args.emb_adapter_name,
        )
        cons_loss = retriever_output["loss_emb"]
        return cons_loss
    
    @typing.no_type_check
    def build_cache(
        self, 
        model: LoRaGenc,
        reps: torch.Tensor, # [batch_size, 1 + num_pos + num_neg, hidden_size]
        **loss_kwargs
        )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the cache for the current batch of data.
        Args:
            query_hidden_states: The hidden states of the query passages
            pos_hidden_states: The hidden states of the positive passages
            neg_hidden_states: The hidden states of the negative passages
            choices_hidden_states: The hidden states of the choices passages
            rejects_hidden_states: The hidden states of the rejects passages
            **loss_kwargs: Additional keyword arguments for the loss function
        """
        reps = reps.detach().requires_grad_(True)  
        with nullcontext():
            cons_loss = self.compute_cons_loss_from_reps(
                model=model,
                reps=reps,
                **loss_kwargs,
            )
        self.fabric.backward(cons_loss)
        cache = reps.grad
        return cache, cons_loss.detach()

    def forward_backward(
        self,
        model: LoRaGenc,
        ref_model: LoRaGenc,
        compute_dpo_loss: bool,
        stage: RandContext,
        model_inputs: List[Dict[str, torch.Tensor]],
        reps_gradcache: torch.Tensor, # [batch_size, 1 + num_pos + num_neg, hidden_size]
        gradient_accumulation_iters: int,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ):
        with stage:
            query_input_ids = model_inputs["query_input_ids"] # [batch_size, max_length]
            query_attention_mask = model_inputs["query_attention_mask"]
            query_prompt_length = model_inputs["query_prompt_length"] # [batch_size]

            pos_input_ids = model_inputs["pos_input_ids"] # [batch_size, num_pos, max_length]
            pos_attention_mask = model_inputs["pos_attention_mask"]
            pos_prompt_length = model_inputs["pos_prompt_length"] # [batch_size, num_pos]

            neg_input_ids = model_inputs["neg_input_ids"] # [batch_size, num_neg, max_length]
            neg_attention_mask = model_inputs["neg_attention_mask"]
            neg_prompt_length = model_inputs["neg_prompt_length"] # [batch_size, num_neg]

            choices_input_ids = model_inputs["choices_input_ids"] # [batch_size, num_pos, max_length]
            choices_attention_mask = model_inputs["choices_attention_mask"]
            choices_labels = model_inputs["choices_labels"]  # [batch_size, num_pos, max_length]
            choices_loss_weight_mask = model_inputs["choices_loss_weight_mask"]

            rejects_input_ids = model_inputs["rejects_input_ids"] # [batch_size, num_neg, max_length]
            rejects_attention_mask = model_inputs["rejects_attention_mask"]
            rejects_labels = model_inputs["rejects_labels"] # [batch_size, num_neg, max_length]
            rejects_loss_weight_mask = model_inputs["rejects_loss_weight_mask"]

            bs = query_input_ids.size(0)
            num_pos = pos_input_ids.size(1)
            num_neg = neg_input_ids.size(1)

            passages_input_ids = torch.cat([
                query_input_ids.unsqueeze(1), # [bs, 1, max_length]
                pos_input_ids,
                neg_input_ids, 
            ], dim=1).view(-1, query_input_ids.size(-1)) # [bs * (1 + num_pos + num_neg), max_length]
            passages_attention_mask = torch.cat([
                query_attention_mask.unsqueeze(1), # [bs, 1, max_length]
                pos_attention_mask,
                neg_attention_mask,
            ], dim=1).view(-1, query_attention_mask.size(-1))
            passages_prompt_length = torch.cat([
                query_prompt_length.unsqueeze(1), # [bs, 1]
                pos_prompt_length,
                neg_prompt_length,
            ], dim=1).flatten() # [bs * (1 + num_pos + num_neg)]

            concat_input_ids = torch.cat([choices_input_ids, rejects_input_ids], dim=1).view(-1, choices_input_ids.size(-1)) # [bs * (num_pos + num_neg), max_length]
            concat_attention_mask = torch.cat([choices_attention_mask, rejects_attention_mask], dim=1).view(-1, choices_attention_mask.size(-1))
            concat_labels = torch.cat([choices_labels, rejects_labels], dim=1).view(-1, choices_labels.size(-1)) # [bs * (num_pos + num_neg), max_length]
            concat_loss_weight_mask = torch.cat([choices_loss_weight_mask, rejects_loss_weight_mask], dim=1).view(-1, choices_loss_weight_mask.size(-1))
            
            retriever_reps = model(
                input_ids=passages_input_ids,
                attention_mask=passages_attention_mask,
                prompt_length=passages_prompt_length,
                is_emb=True,
                use_miner=training_args.use_miner,
                adapter_name=model_args.emb_adapter_name,
            )['reps'] # [bs * (1 + num_pos + num_neg), hidden_size]
            _retriever_reps = retriever_reps.view(bs, 1 + num_pos + num_neg, -1) # [bs, 1 + num_pos + num_neg, hidden_size]
            query_reps = _retriever_reps[:, 0] # [bs, hidden_size]
            passages_reps = _retriever_reps[:, 1:] # [bs, num_pos + num_neg, hidden_size]

            reranker_logits = model(
                input_ids=concat_input_ids,
                attention_mask=concat_attention_mask,
                is_gen=True,
                adapter_name=model_args.gen_adapter_name,
            )['logits'] # [bs * (num_pos + num_neg), max_length, vocab_size]

            with torch.no_grad():
                if ref_model is not None:
                    ref_logits = ref_model(
                        input_ids=concat_input_ids,
                        attention_mask=concat_attention_mask,
                        is_gen=True,
                    )['logits'] # [bs * (num_pos + num_neg), max_length, vocab_size]
                else: # use the reranker backbone as the reference model (but without adapter)
                    ref_logits = model(
                        input_ids=concat_input_ids,
                        attention_mask=concat_attention_mask,
                        is_gen=True,
                        disable_adapter=True
                    )['logits'] # [bs * (num_pos + num_neg), max_length, vocab_size]
            reranker_logps = get_batch_logps(
                logits=reranker_logits,
                labels=concat_labels,
                loss_weight_mask=concat_loss_weight_mask,
                average_log_prob=training_args.dpo_loss_type == "ipo",
                label_pad_token_id=-100,
            ) # [bs * (num_pos + num_neg)]
            reranker_logps = reranker_logps.view(bs, num_pos + num_neg) # [bs, num_pos + num_neg]
            ref_logps = get_batch_logps(
                logits=ref_logits,
                labels=concat_labels,
                loss_weight_mask=concat_loss_weight_mask,
                average_log_prob=training_args.dpo_loss_type == "ipo",
                label_pad_token_id=-100,
            ) # [bs * (num_pos + num_neg)]
            ref_logps = ref_logps.view(bs, num_pos + num_neg) # [bs, num_pos + num_neg]
            
            # Reranker loss
            if (training_args.mode == 'edpo') and compute_dpo_loss: 
                policy_choice_logps = reranker_logps[:, 0] # [bs]
                policy_reject_logps = reranker_logps[:, num_pos] # [bs]
                ref_choice_logps = ref_logps[:, 0]
                ref_reject_logps = ref_logps[:, num_pos]
                reranker_loss, _, _ = dpo_loss(
                    policy_choice_logps,
                    policy_reject_logps,
                    ref_choice_logps,
                    ref_reject_logps,
                    loss_type=training_args.dpo_loss_type,
                    label_smoothing=training_args.label_smoothing_factor,
                    beta=training_args.dpo_beta,
                )
            elif training_args.mode == 'esft':
                reranker_logits = reranker_logits.view(bs, num_pos + num_neg, -1, reranker_logits.size(-1)) # [bs, num_pos + num_neg, max_length, vocab_size]
                choices_logits = reranker_logits[:, :num_pos] # [bs, num_pos, max_length, vocab_size]
                reranker_loss = chunked_cross_entropy(
                    logits=choices_logits,
                    targets=choices_labels,
                    loss_weight_mask=choices_loss_weight_mask
                )
            else:
                reranker_loss = torch.tensor(0.0, device=reranker_logits.device)
            
            #KL loss
            dual_scores = torch.cosine_similarity(query_reps.unsqueeze(1), passages_reps, dim=-1) # [bs, num_pos + num_neg]
            dual_logps = torch.log_softmax(dual_scores, dim=-1) # [bs, num_pos + num_neg]
            reranker_scores = training_args.dpo_beta * (reranker_logps - ref_logps) # [bs, num_pos + num_neg]
            reranker_probs = torch.softmax(reranker_scores, dim=-1) # [bs, num_pos + num_neg]
            kl_loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            kl_loss = kl_loss_func(dual_logps, reranker_probs)

            loss = training_args.kl_loss_weight * kl_loss + training_args.gen_loss_weight * reranker_loss
            loss = loss / gradient_accumulation_iters
            con_loss_surrogate = torch.dot(retriever_reps.flatten(), reps_gradcache.flatten())

            # Backward pass
            self.fabric.backward(loss)
            self.fabric.backward(con_loss_surrogate)

            return reranker_loss.detach(), kl_loss.detach()

    def training_step(
        self,
        model: LoRaGenc,
        ref_model: LoRaGenc,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        batch: Any,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of data. (including forward and backward pass)
        Args:
            model: The model to train
            ref_model: The reference model for the DPO loss
            model_args: The model arguments
            training_args: The training arguments
            batch: The batch of data
        Returns:
            The loss on the batch of data
        """
        splitted_inputs = split_input(batch, self.chunk_size)
        rnd_states = []
        query_reps = []
        pos_reps = []
        neg_reps = []
        for chunk in splitted_inputs:            
            query_rep, pos_rep, neg_rep, rnd_state = self.forward_no_grad(
                model=model,
                model_inputs=chunk,
                model_args=model_args,
                training_args=training_args,
            )
            query_reps.append(query_rep)
            pos_reps.append(pos_rep)
            neg_reps.append(neg_rep)
            rnd_states.append(rnd_state)
        query_reps = torch.cat(query_reps, dim=0) # [batch_size, hidden_size]
        pos_reps = torch.cat(pos_reps, dim=0) # [batch_size, num_pos, hidden_size]
        neg_reps = torch.cat(neg_reps, dim=0) # [batch_size, hidden_size]
        reps = torch.cat([
            query_reps.unsqueeze(1),
            pos_reps,
            neg_reps,
        ], dim=1) # [batch_size, 1 + num_pos + num_neg, hidden_size]

        cons_loss_kwargs = {
            "query_labels": batch["query_labels"], # [batch_size]
            "pos_labels": batch["pos_labels"], # [batch_size, num_pos]
            "neg_labels": batch["neg_labels"], # [batch_size, num_neg]
            "model_args": model_args,
            "training_args": training_args,
        }
        cache, cons_loss = self.build_cache(
            model=model,
            reps=reps,
            **cons_loss_kwargs,
        ) # [batch_size, 1 + num_pos + num_neg, hidden_size]
        cache = cache.split(self.chunk_size)
        accumulated_flags = [True for _ in range(len(splitted_inputs)-1)] + [False]
        gradient_accumulation_iters = len(splitted_inputs)
        all_reranker_loss = []
        all_kl_loss = []
        for i, (flag, chunk, model_cache, state) in enumerate(zip(accumulated_flags, splitted_inputs, cache, rnd_states)):
            compute_dpo_loss = True if training_args.num_dpo_step_per_batch < i else False
            with self.fabric.no_backward_sync(model, enabled=flag):
                reranker_loss, kl_loss = self.forward_backward(
                    model=model,
                    ref_model=ref_model,
                    compute_dpo_loss=compute_dpo_loss,
                    stage=state,
                    model_inputs=chunk,
                    reps_gradcache=model_cache, # [batch_size, 1 + num_pos + num_neg, hidden_size]
                    gradient_accumulation_iters=gradient_accumulation_iters,
                    model_args=model_args,
                    training_args=training_args,
                )
                all_reranker_loss.append(reranker_loss)
                all_kl_loss.append(kl_loss)
        reranker_loss = torch.stack(all_reranker_loss).mean()
        kl_loss = torch.stack(all_kl_loss).mean()
        return cons_loss, reranker_loss, kl_loss


def fit(
    fabric: L.Fabric,
    model: LoRaGenc,
    ref_model: LoRaGenc,
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
    # initialize gradcache
    fabric.print("Initializing Gradcache")
    gc = GradCacheTrainer(
        fabric=fabric,
        chunk_size=training_args.gc_mini_batch_size,
    )
    train_iterator = CycleIterator(train_dataloader)
    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        training_args.num_train_epochs * steps_per_epoch, 
        (training_args.max_steps or float("inf"))
        )
    iter_num = 0

    reranker_running_loss = RunningMean(window=1, sync_on_compute=False).to(fabric.device)
    kl_running_loss = RunningMean(window=1, sync_on_compute=False).to(fabric.device)
    cons_running_loss = RunningMean(window=1, sync_on_compute=False).to(fabric.device)

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

        cons_loss, reranker_loss, kl_loss = gc.training_step(
            model=model,
            ref_model=ref_model,
            model_args=model_args,
            training_args=training_args,
            batch=batch,
        )
        cons_running_loss(cons_loss.detach())
        reranker_running_loss(reranker_loss.detach())
        kl_running_loss(kl_loss.detach())

        if training_args.apply_gradient_clipping and training_args.grad_norm_clip is not None:
            fabric.clip_gradients(model, optimizer, max_norm=training_args.grad_norm_clip)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        
        if iter_num % training_args.log_interval == 0:
            _cons_loss = cons_running_loss.compute().item()
            _kl_loss = kl_running_loss.compute().item()
            _reranker_loss = reranker_running_loss.compute().item()
            t1 = time.perf_counter()

            metrics = {
                "cons_loss": _cons_loss,
                "kl_loss": _kl_loss,
                "reranker_loss": _reranker_loss,
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
            f" reranker loss: {_reranker_loss:.3f},"
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

