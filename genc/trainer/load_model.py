import json
import logging
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    BitsAndBytesConfig, 
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from genc.model.genc import MistralEmbeddingLM
from genc.trainer.trainer_utils import get_trainable_parameters, get_device_map
from genc.special_tokens import base_bos, user_bos, user_eos, embed_bos, embed_eos, assistant_bos, assistant_eos


def load_model(
        model_weights_name_or_path: str,
        use_bidirectional: bool = False,
        normalized: bool = True,
        pooling_method: str = "mean",
        loss_gen_type: str = "mixed",
        temperature: float = 0.05,
        quantization: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
        use_lora: bool = False,
        train_adapter_name: Optional[str] = "default",
        lora_weights_name_or_path: Optional[str] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_r: Optional[int] = 8,
        lora_alpha: Optional[int] = 16,
        lora_dropout: Optional[float] = 0.05,
        torch_dtype: Optional[str] = "bfloat16",
        inference: bool = False,
        fsdp: bool = False,
        **kwargs,) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizer]:
    # Sanity checks
    if isinstance(quantization, str):
        quantization = int(quantization)
    assert (quantization is None) or (
        quantization in [4, 8]
    ), f"Quantization must be 4 or 8, or None for FP32/FP16 training. You passed: {quantization}"

    assert torch_dtype in ["bfloat16", "float32", "float16"], f"torch_dtype must be 'bfloat16', 'float32', or 'float16'. You passed: {torch_dtype}"

    if lora_weights_name_or_path is not None and not use_lora:
        logging.warning("You provided a path to LoRA weights but use_lora is set to False. We will set use_lora=True.")

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
            logging.warning("Tokenizer does not have a pad token. We will use the bos token as pad token.")
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
    logging.info(f"Loading model from {model_weights_name_or_path}")
    device_map, max_memory = get_device_map()
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
    # Get the quantization config
    torch_dtype = torch_dtype if torch_dtype in ["auto", None] else getattr(torch, torch_dtype)
    if quantization is not None:
        if quantization == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype in ["auto", None] else torch_dtype,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        logging.info(f"Bits and Bytes config: {json.dumps(bnb_config.to_dict(),indent=4,ensure_ascii=False)}")
    else:
        logging.info(f"Loading model with dtype: {torch_dtype}")
        bnb_config = None
    # Specify model args
    model_args = [use_bidirectional, normalized, pooling_method, loss_gen_type, temperature, new_vocab_size]
    model: PreTrainedModel = MistralEmbeddingLM.from_pretrained(
        model_weights_name_or_path,
        *model_args,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        config=config,
        # trust_remote_code=True,
        quantization_config=bnb_config,
        **kwargs,
    )
    # Quantization
    if quantization is not None and not inference:
        logging.info("Quantizing model")
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    else:
        if use_gradient_checkpointing and not inference:
            model.gradient_checkpointing_enable()
    # Load LoRA weights
    if use_lora:
        if not inference:
            model.enable_input_require_grads() # Enable the gradients for the input embeddings
        if lora_weights_name_or_path is None:
            logging.info("No LoRA weights provided, we will use the default random LoRA weights.")
            if lora_target_modules == ["all"]:
                logging.warning(
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
            )
            model: PeftModel = get_peft_model(model, lora_config, adapter_name=train_adapter_name)
        else:
            logging.info(f"Loading pretrained LORA weights from {lora_weights_name_or_path}")
            model: PeftModel = PeftModel.from_pretrained(model, lora_weights_name_or_path, adapter_name=train_adapter_name, is_trainable=True)
        logging.info(f"\nLoRA config:\n{model.peft_config}\n")
        # Coverting LoRA layers to the desired dtype if bf16 is used for faster training
        from peft.tuners.lora import LoraLayer
        from peft.tuners.tuners_utils import BaseTunerLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if torch_dtype == torch.bfloat16:
                    logging.debug(f"Converting LoRA layer {name} to {torch_dtype}")
                    module = module.to(torch.bfloat16)
            elif isinstance(module, BaseTunerLayer):
                if torch_dtype == torch.bfloat16:
                    logging.debug(f"Converting LoRA layer {name} to {torch_dtype}")
                    module = module.to(torch.bfloat16)
    
    if not fsdp:
        for name, module in model.named_modules():
            if "norm" in name or isinstance(module, nn.LayerNorm):
                logging.debug(f"Converting layer {name} to {torch.float32}")
                module = module.to(torch.float32)
            elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
                if hasattr(module, "weight"):
                    if torch_dtype == torch.bfloat16 and module.weight.dtype == torch.float32:
                        logging.debug(f"Converting layer {name} to {torch_dtype}")
                        module = module.to(torch.bfloat16)
    
    if inference:
        if use_lora:
            if quantization is None:
                # If we are not using quantization, we merge the LoRA layers into the model for faster inference.
                # This is not possible if we are using 4/8 bit quantization.
                logging.info("Merging LoRA layers into the model for faster inference.")
                model = model.merge_and_unload()
            else:
                logging.info(
                    "Quantization is enabled, we will not merge LoRA layers into the model. Inference will be slower."
                )
    else:
        trainable_params, total_params, trainable_percentage = get_trainable_parameters(model)
        logging.info(
            f"---> Trainable params: {trainable_params} || all params: {total_params} ||"
            f" trainable%: {round(trainable_percentage,6)}\n"
        )
        
    if len(additional_special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
        config.vocab_size += len(additional_special_tokens)
        model.config.vocab_size = len(tokenizer)
    return model, tokenizer

