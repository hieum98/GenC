from collections import UserDict
import functools
from itertools import repeat
from contextlib import contextmanager, nullcontext
from tqdm.auto import tqdm
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from typing_extensions import Self
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
import lightning as L
import wandb
from transformers import PreTrainedModel
from peft import PeftModel
# To add a new model, import the transformer, attention, & MLP layers
# for the wrapping policy and `check_fn` in activation checkpointing
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MISTRAL_ATTENTION_CLASSES, MistralMLP



def parse_devices(devices: Union[str, int]) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


class Logger:
    def __init__(self, args, log_to="stdout", project_name="fsdp_qlora", entity=None, group=None, name=None, rank=0):
        # self.log_every_n_steps = log_every_n_steps TODO: add this back as an option
        self.log_to = log_to
        if self.log_to == "wandb" and rank==0:
            wandb.init(project=project_name, entity=entity, group=group, name=name, config=args)

    def log(self, d:Dict, rank:int):
        if rank != 0: return
        if self.log_to == "tqdm":
            for k,v in d.items():
                tqdm.write(f'{k}: {v}')
        elif self.log_to == "wandb":
            wandb.log(d)
        elif self.log_to == "stdout":
            for k,v in d.items():
                print(f'{k}: {v}')

    def finish(self, rank=0):
        if self.log_to == "wandb" and rank==0: wandb.finish()


def get_default_supported_precision(training: bool) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True, 
    layer_norm_names=["layer_norm"], 
    output_embedding_layer_name="lm_head"
):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        if loaded_in_kbit:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32

            """

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


def find_all_linear_names(model, quantization: Optional[int] = None):
    if quantization is None:
        cls = torch.nn.Linear
    elif quantization == 4:
        from bitsandbytes.nn import Linear4bit

        cls = Linear4bit
    elif quantization == 8:
        from bitsandbytes.nn import Linear8bitLt

        cls = Linear8bitLt
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_trainable_parameters(model: PreTrainedModel) -> Tuple[int, int, float]:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (`PreTrainedModel`):
            The model to print the number of trainable parameters for.

    Returns:
        `Tuple[int, int, float]`:
            The number of trainable parameters, the total number of parameters and the
            percentage of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param, 100 * trainable_params / all_param


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(custom_policy:bool=False):
    from peft.tuners import PromptEncoder, PromptEmbedding, PrefixEncoder

    if custom_policy:
        def lambda_policy_fn(module):
            # LORA trainable layers.
            return (isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module))
    else:
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, tuple(*LLAMA_ATTENTION_CLASSES.values(), *MISTRAL_ATTENTION_CLASSES.values()))

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaMLP, MistralMLP))

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer),
    )
    policies=[lambda_policy, transformer_wrap_policy]
    if custom_policy:
        policies.extend([self_attn_policy, mlp_policy])
    return functools.partial(_or_policy, policies=policies)


def split_input(model_input, chunk_size: int) -> List:
    if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, torch.Tensor) for x in model_input.values()):
        keys = list(model_input.keys())
        chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
        return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    elif isinstance(model_input, list) and all(isinstance(x, torch.Tensor) for x in model_input):
        chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
        return [list(s) for s in zip(*chunked_x)]

    elif isinstance(model_input, torch.Tensor):
        return list(model_input.split(chunk_size, dim=0))

    elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
        args_chunks = split_input(model_input[0], chunk_size)
        kwargs_chunks = split_input(model_input[1], chunk_size)
        return list(zip(args_chunks, kwargs_chunks))

    else:
        raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

@contextmanager
def null_ref_context(model: Union[torch.nn.Module, PreTrainedModel, PeftModel], is_peft_model: bool, train_adapter_name: str="default"):
    """Context manager for handling null reference model (that is, peft adapter manipulation)."""
    with model.disable_adapter() if is_peft_model else nullcontext():
        yield
        model.set_adapter(train_adapter_name)

def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    loss_weight_mask: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
    labels = labels[:, 1:].clone()
    loss_weight_mask = loss_weight_mask[..., 1:].clone().contiguous() if loss_weight_mask is not None else None
    loss_mask = labels != label_pad_token_id
    loss_weight_mask = loss_weight_mask * loss_mask if loss_weight_mask is not None else loss_mask
    logits = logits[:, :-1, :]
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_weight_mask).sum(-1) / loss_weight_mask.sum(-1) # (batch_size,)
    else:
        return (per_token_logps * loss_weight_mask).sum(-1) # (batch_size,)


def dpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        loss_type: str = "sigmoid",
        reference_free: bool = False,
        label_smoothing: float = 0.0,
        beta: float = 0.1,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            loss_type: The type of DPO loss to compute. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair'].
            reference_free: Whether to ignore the reference model. If True, the reference model log probabilities are set to 0.
            label_smoothing: The label smoothing parameter for the DPO loss. Should be in the range [0, 1].
            beta: The temperature parameter for the DPO loss. Should be a positive scalar.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        elif loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        elif loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * beta)) ** 2
        elif loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            beta
            * (
                policy_chosen_logps - reference_chosen_logps
            ).detach()
        )
        rejected_rewards = (
            beta
            * (
                policy_rejected_logps
                - reference_rejected_logps
            ).detach()
        )
        return losses, chosen_rewards, rejected_rewards


def kl_loss(
        emb_reps: torch.FloatTensor,
        pair_logits: torch.FloatTensor,
        pair_labels: torch.LongTensor,
        pair_loss_weight_mask: torch.FloatTensor,
        bs: int,
        ):
    query_reps = emb_reps[:bs] # [bs, emb_dim]
    passage_reps = emb_reps[bs:].reshape(bs, -1, emb_reps.size(-1)) # [bs, 1 + topk_neg, emb_dim]
    dual_score = torch.cosine_similarity(query_reps.unsqueeze(1), passage_reps, dim=-1)
    dual_score = torch.log_softmax(dual_score, dim=1) # [bs, 1 + topk_neg]
    gen_logps = get_batch_logps(
        logits=pair_logits,
        labels=pair_labels,
        loss_weight_mask=pair_loss_weight_mask,
        average_log_prob=True
    ) # [bs * (1 + topk_neg)]
    gen_logps = gen_logps.view(bs, -1)
    gen_logps = torch.softmax(gen_logps, dim=1) # [bs, 1 + topk_neg]
    # KL loss
    kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    kl_loss = kl(dual_score, gen_logps)
    return kl_loss


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


