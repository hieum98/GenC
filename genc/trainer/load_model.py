import copy
import json
import logging
import time
from typing import List, Optional, Tuple, Union
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig, 
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from accelerate import init_empty_weights
from bitsandbytes.nn import Linear4bit, Params4bit
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from fastcore.parallel import parallel

from genc.model.genc import MistralEmbeddingLM
from genc.trainer.loading_utils import load_and_quantize_parallel, n_loading_workers, replace_linear, setup_quantized_meta_for_peft, setup_quantized_peft_meta_for_training
from genc.special_tokens import base_bos, user_bos, user_eos, embed_bos, embed_eos, assistant_bos, assistant_eos

logger = logging.getLogger(__name__)

def load_model(
        model_weights_name_or_path: str,
        use_bidirectional: bool = False,
        normalized: bool = True,
        pooling_method: str = "mean",
        loss_gen_type: str = "mixed",
        temperature: float = 0.05,
        quantization: bool = False,
        use_lora: bool = False,
        train_adapter_name: Optional[str] = "default",
        lora_weights_name_or_path: Optional[str] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_r: Optional[int] = 8,
        lora_alpha: Optional[int] = 16,
        lora_dropout: Optional[float] = 0.05,
        inference: bool = False,
        low_memory: bool = True,
        torch_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        precision: str = "bf16",
        rank: int = 0,
        local_rank: int = 0,
        **kwargs,) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizer]:
    if lora_weights_name_or_path is not None and not use_lora:
        logger.warning("You provided a path to LoRA weights but use_lora is set to False. We will set use_lora=True.")

    # Load tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_weights_name_or_path,
        padding_side="right", # Has to be right so masking of instruction tokens works correctly
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if "<|padding|>" in tokenizer.get_vocab():
            # StabilityLM specific fix
            tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        else:
            logger.warning("Tokenizer does not have a pad token. We will use the bos token as pad token.")
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.pad_token_id = tokenizer.bos_token_id
    # Add special tokens into tokenizer
    additional_special_tokens = [base_bos, user_bos, user_eos, embed_bos, embed_eos, assistant_bos, assistant_eos]
    for item in additional_special_tokens:
        if item in tokenizer.vocab:
            additional_special_tokens.remove(item)
    if len(additional_special_tokens) > 0:
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    new_vocab_size = len(tokenizer)

    # Load model
    logger.info(f"Loading model from {model_weights_name_or_path}")
    # Load model config
    if use_lora:
        config = AutoConfig.from_pretrained(
            model_weights_name_or_path,
            trust_remote_code=True,
            pretraining_tp=1,  # Fix mat1 and mat2 shapes cannot be multiplied  error with LLaMA-2
            # See https://github.com/huggingface/transformers/pull/24906
        )
    else:
        config = AutoConfig.from_pretrained(
            model_weights_name_or_path,
            trust_remote_code=True,
        )
    # Create the model
    # Specify model args
    model_args = [use_bidirectional, normalized, pooling_method, loss_gen_type, temperature, new_vocab_size]
    if quantization is False:
        if (low_memory and rank==0) or (not low_memory):
            model = MistralEmbeddingLM.from_pretrained(
                model_weights_name_or_path,
                *model_args,
                use_cache=False,
                torch_dtype=torch_dtype,
                pretraining_tp=1,  # Fix mat1 and mat2 shapes cannot be multiplied  error with LLaMA-2
                **kwargs,
            )
            dtype = torch_dtype if precision == "bf16" else None
            model.to(dtype=dtype, device="cpu" if low_memory else rank)
        else:
            config.use_cache = False
            if "_attn_implementation" in kwargs:
                config._attn_implementation = kwargs["_attn_implementation"]
            with init_empty_weights():
                model = MistralEmbeddingLM._from_config(
                    config,
                    torch_dtype=torch_dtype,
                    use_bidirectional=use_bidirectional,
                    normalized=normalized,
                    pooling_method=pooling_method,
                    loss_gen_type=loss_gen_type,
                    temperature=temperature,
                    new_vocab_size=new_vocab_size,
                )
            if precision == "bf16":
                model = model.to(torch_dtype)
    else:
        config.use_cache = False
        if "_attn_implementation" in kwargs:
            config._attn_implementation = kwargs["_attn_implementation"]
        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model: MistralEmbeddingLM = MistralEmbeddingLM._from_config(
                config,
                use_bidirectional=use_bidirectional,
                normalized=normalized,
                pooling_method=pooling_method,
                loss_gen_type=loss_gen_type,
                temperature=temperature,
                new_vocab_size=new_vocab_size,
            )
            model.model = replace_linear(model.model, Linear4bit, compute_dtype=compute_dtype,
                                             quant_type='nf4', quant_storage=torch_dtype)
        model.is_loaded_in_4bit = True
        # Grab the safetensors files that hold the weights
        try:
            idx = hub.cached_file(model_weights_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
            files, _ = hub.get_checkpoint_shard_files(model_weights_name_or_path, idx)
        except OSError:
            try:
                # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                files = []
                files.append(hub.cached_file(model_weights_name_or_path, SAFE_WEIGHTS_NAME))
            except OSError as e:
                # This means the model probably doesn't have a safetensors file
                raise e
        quant_method = "bnb"
        param_count = sum((p.numel() for n,p in model.named_parameters()))
        n_workers = n_loading_workers(quant_method, param_count)
        if rank == 0:
            logger.info(f"Using n_workers: {n_workers} for loading")
        start = time.time()
        for filename in tqdm(files, desc="Loading & Quantizing Model Shards", disable=rank!=0, position=0):
            weights = safetensors.torch.load_file(filename)
            parallel(load_and_quantize_parallel, iter(weights.items()), n_workers=n_workers, threadpool=True,
                     model=model, dtype=torch_dtype, device=local_rank, skip_names=[],
                     to_cpu=(low_memory and rank==0), to_meta=(low_memory and rank!=0),
                     verbose=True, quant_method=quant_method)

        if rank == 0:
            logger.info(f"Loaded model weights in {time.time()-start:.3f} seconds")
        # cleanup any extra memory usage from parallel loading
        torch.cuda.empty_cache()
    if rank == 0:
        print(f"Rank {rank}: Model created: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")

    # Load LoRA weights
    if use_lora:
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        if rank!=0 and low_memory:
            setup_quantized_meta_for_peft(model)
            
        if lora_weights_name_or_path is None:
            logger.info("No LoRA weights provided, we will use the default random LoRA weights.")
            if lora_target_modules == ["all"]:
                logger.warning(
                    "You provided 'all' as target modules, we will use all the model to which LoRA can be applied."
                )
                from genc.trainer.trainer_utils import find_all_linear_names
                lora_target_modules = find_all_linear_names(model, quantization=quantization)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_target_modules,
                inference_mode=inference,
            )
            model: PeftModel = get_peft_model(model, lora_config, adapter_name=train_adapter_name)
        else:
            logger.info(f"Loading pretrained LORA weights from {lora_weights_name_or_path}")
            model: PeftModel = PeftModel.from_pretrained(model, lora_weights_name_or_path, adapter_name=train_adapter_name, is_trainable=True)

        if rank==0:
            model.print_trainable_parameters()
        elif low_memory:
            # And then setup_quantized_peft_meta_for_training sets quant_state.to back to normal
            setup_quantized_peft_meta_for_training(model)
    
    if len(additional_special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
        config.vocab_size += len(additional_special_tokens)
        model.config.vocab_size = len(tokenizer)
    
    if rank==0:
        logger.info({"memory/allocated_after_model_created": torch.cuda.memory_allocated(local_rank)})
        logger.info({"memory/reserved_after_model_creation": torch.cuda.memory_reserved(local_rank)})
    
    return model, tokenizer

