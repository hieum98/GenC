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
from transformers import AutoTokenizer, AutoConfig
from accelerate import init_empty_weights
from peft import LoraConfig, TaskType
from fastcore.parallel import parallel

from genc.model.genc import LlamaEmbeddingLM, MistralEmbeddingLM, PhiEmbeddingLM
from genc.model.lora_genc import LoRaGenc
from genc.special_tokens import base_bos, user_bos, user_eos, embed_bos, embed_eos, assistant_bos, assistant_eos
from genc.trainer.trainer_utils import find_all_linear_names
from genc.trainer.loading_utils import load_and_quantize_parallel, n_loading_workers, replace_linear

def load_model(
        model_weights_name_or_path: str,
        pretrained_type: str,
        use_bidirectional: bool = False,
        normalized: bool = True,
        pooling_method: str = "mean",
        loss_gen_type: str = "mixed",
        temperature: float = 0.05,
        quantization: bool = False,
        use_lora: bool = False,
        emb_adapter_name: Optional[str] = "emb",
        gen_adapter_name: Optional[str] = "gen",
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
        gradient_checkpointing: bool = False,
        **kwargs,) -> Tuple[Union[PreTrainedModel, LoRaGenc], PreTrainedTokenizer]:

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
            print("Tokenizer does not have a pad token. We will use the bos token as pad token.")
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.pad_token_id = tokenizer.bos_token_id
    # Add special tokens into tokenizer
    additional_special_tokens = [base_bos, user_bos, user_eos, embed_bos, embed_eos, assistant_bos, assistant_eos]
    for item in additional_special_tokens:
        if item in tokenizer.vocab:
            additional_special_tokens.remove(item)
    if len(additional_special_tokens) > 0:
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    # Load model
    print(f"Loading model from {model_weights_name_or_path}")
    # Load model config
    if use_lora:
        config = AutoConfig.from_pretrained(
            model_weights_name_or_path,
            trust_remote_code=True,
            use_cache=False,
            pretraining_tp=1,  # Fix mat1 and mat2 shapes cannot be multiplied  error with LLaMA-2
            # See https://github.com/huggingface/transformers/pull/24906
        )
    else:
        config = AutoConfig.from_pretrained(
            model_weights_name_or_path,
            trust_remote_code=True,
            use_cache=False
        )
    # Create the model
    # Specify model args
    model_args = [use_bidirectional, normalized, pooling_method, loss_gen_type, temperature, tokenizer]
    if pretrained_type == 'llama':
        model_class = LlamaEmbeddingLM
    elif pretrained_type == 'mistral':
        model_class = MistralEmbeddingLM
    elif pretrained_type == 'phi':
        model_class = PhiEmbeddingLM
    else:
        raise ValueError(f"Model type not recognized: {pretrained_type}")
    
    if quantization is False:
        model = model_class.from_pretrained(
            model_weights_name_or_path,
            *model_args,
            config=config,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        dtype = torch_dtype if precision == "bf16" else None
        model.to(dtype=dtype, device="cpu" if low_memory else local_rank)
    elif inference:
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype in ["auto", None] else torch_dtype,
            )
        model: PreTrainedModel = model_class.from_pretrained(
                model_weights_name_or_path,
                *model_args,
                config=config,
                # trust_remote_code=True,
                quantization_config=bnb_config,
                **kwargs,
            )
    else:
        from bitsandbytes.nn import Linear4bit, Params4bit
        config.use_cache = False
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs["attn_implementation"]
        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model = model_class._from_config(
                config,
                use_bidirectional=use_bidirectional,
                normalized=normalized,
                pooling_method=pooling_method,
                loss_gen_type=loss_gen_type,
                temperature=temperature,
                tokenizer=tokenizer,
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
            print(f"Using n_workers: {n_workers} for loading")
        start = time.time()
        for filename in tqdm(files, desc="Loading & Quantizing Model Shards", disable=rank!=0, position=0):
            weights = safetensors.torch.load_file(filename)
            parallel(load_and_quantize_parallel, iter(weights.items()), n_workers=n_workers, threadpool=True,
                     model=model, dtype=torch_dtype, device=local_rank, skip_names=[],
                     to_cpu=(low_memory and local_rank==0), to_meta=(low_memory and local_rank!=0),
                     verbose=True, quant_method=quant_method)

        if rank == 0:
            print(f"Loaded model weights in {time.time()-start:.3f} seconds")
        # cleanup any extra memory usage from parallel loading
        torch.cuda.empty_cache()
    print(f"Rank {rank}: Model created: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")

    if len(additional_special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
        config.vocab_size += len(additional_special_tokens)
        model.config.vocab_size = len(tokenizer)

    # Load LoRA weights
    if use_lora:
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        assert emb_adapter_name is not None or gen_adapter_name is not None, "You must provide at least one adapter name"

        if lora_target_modules == ["all"]:
            print("You provided 'all' as target modules, we will use all the model to which LoRA can be applied.")
            lora_target_modules = find_all_linear_names(model, quantization=quantization)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_target_modules,
            modules_to_save = ["lm_head", "embed_tokens"],
            inference_mode=inference,
        )
        if emb_adapter_name is not None:
            model = LoRaGenc(
                model=model,
                lora_config=lora_config,
                adapter_name=emb_adapter_name
            )
        if gen_adapter_name is not None and gen_adapter_name != emb_adapter_name:
            if isinstance(model, LoRaGenc):
                model.add_adapter(adapter_name=gen_adapter_name, peft_config=lora_config)
            else:
                model = LoRaGenc(
                    model=model,
                    lora_config=lora_config,
                    adapter_name=gen_adapter_name
                )
        # Always set the emb adapter as the current adapter
        model.set_adapter(emb_adapter_name)
        
        if rank==0:
            model.print_trainable_parameters()
    
    if gradient_checkpointing:
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if rank==0:
        print({"memory/allocated_after_model_created": torch.cuda.memory_allocated(local_rank)})
        print({"memory/reserved_after_model_creation": torch.cuda.memory_reserved(local_rank)})
    
    return model, tokenizer

