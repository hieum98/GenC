from dataclasses import dataclass
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import torch 
import torch.nn as nn
from transformers import (AutoConfig, 
                          BitsAndBytesConfig, 
                          AutoModelForCausalLM,
                          AutoModel,
                          PreTrainedModel,
                          PreTrainedTokenizer,
                          MistralPreTrainedModel,
                          MistralConfig)
from transformers.file_utils import ModelOutput
from pytorch_metric_learning import losses, miners, distances

from .modules import NextTokenLoss
from .model_utils import get_trainable_parameters


@dataclass
class EmLMTrainOutput(ModelOutput):
    reps: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_emb: Optional[torch.Tensor] = None
    loss_gen: Optional[torch.Tensor] = None


def get_device_map(force_auto_device_map: bool=False, max_memory_MB: int = None) -> Tuple[str, Union[int, List[int]]]:
    if force_auto_device_map:
        if os.environ.get("LOCAL_RANK") is not None:
            # raise ValueError(
            #    "Found DDP environment and force_auto_device_map is set to True, this configuration "
            #    "is not supported. If you want to use DPP, set force_auto_device_map to False, so "
            #    "a copy of the model is loaded in each GPU. If you want the split the model across "
            #    "GPUs (force_auto_device_map=True), do not use DDP (launch your script with "
            #    "pyton -m src/run.py config.json). If you are not in a DDP environment but you see "
            #    "this error, you might have manually set the environment variable 'LOCAL_WORLD_SIZE' to a "
            #    "number different than 1, please, remove this environment variable or set it to 1"
            # )
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
            else:
                logging.warning("You are in a DDP environment but no GPU is available, this may cause errors later on")
                n_gpus = 0

            max_memory = {i: max_memory_MB for i in range(n_gpus)}
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_map = {"": local_rank}
            max_memory = {"": max_memory[local_rank]} if max_memory_MB is not None else None

        else:
            logging.warning(
                "Using auto device map, we will split the model across GPUs and CPU to fit the model in memory."
            )
            device_map = "auto"
            max_memory = max_memory_MB
    else:
        max_memory = None
        device_map = None
    logging.info(f"We will load the model using the following device map: {device_map} and max_memory: {max_memory}")

    return device_map, max_memory

class EmLM(MistralPreTrainedModel):
    def __init__(
            self,
            model_weights_name_or_path: str,
            tokenizer: PreTrainedTokenizer,
            quantization: Optional[int] = None,
            use_gradient_checkpointing: bool = False,
            use_lora: bool = False,
            lora_weights_name_or_path: Optional[str] = None,
            lora_target_modules: Optional[List[str]] = None,
            lora_r: Optional[int] = 8,
            lora_alpha: Optional[int] = 16,
            lora_dropout: Optional[float] = 0.05,
            torch_dtype: Optional[str] = "bfloat16",
            inference: bool = False,
            fsdp: bool = False,
            normalized: bool = True,
            pooling_method: str = 'mean', # One of ['cls', 'lasttoken', 'mean', 'weightedmean']
            loss_gen_type: str = "mixed",
            loss_gen_factor: float = None,
            ) -> None:

        # Sanity checks
        if isinstance(quantization, str):
            quantization = int(quantization)
        assert (quantization is None) or (
            quantization in [4, 8]
        ), f"Quantization must be 4 or 8, or None for FP32/FP16 training. You passed: {quantization}"

        assert torch_dtype in ["bfloat16", "float32", "float16"], f"torch_dtype must be 'bfloat16', 'float32', or 'float16'. You passed: {torch_dtype}"

        if lora_weights_name_or_path is not None and not use_lora:
            logging.warning("You provided a path to LoRA weights but use_lora is set to False. We will set use_lora=True.")
            use_lora = True

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

        super().__init__(config)

        # Load the model weights
        #  Get the quantization config
        quant_args = {}
        torch_dtype = torch_dtype if torch_dtype in ["auto", None] else getattr(torch, torch_dtype)

        if quantization is not None:
            quant_args = {"load_in_4bit": True} if quantization == 4 else {"load_in_8bit": True}
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
        
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_weights_name_or_path,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            config=config,
            trust_remote_code=True,
            quantization_config=bnb_config,
            **quant_args,
        )

        if quantization is not None and not inference:
            logging.info("Quantizing model")
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
        else:
            if use_gradient_checkpointing and not inference:
                model.gradient_checkpointing_enable()
        
        # Load lora weights
        if use_lora:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            if not inference:
                model.enable_input_require_grads() # Enable the gradients for the input embeddings
            
            if lora_weights_name_or_path is None:
                logging.info("No LoRA weights provided, we will use the default random LoRA weights.")

                if lora_target_modules == ["all"]:
                    logging.warning(
                        "You provided 'all' as target modules, we will use all the model to which LoRA can be applied."
                    )
                    from .model_utils import find_all_linear_names
                    lora_target_modules = find_all_linear_names(model, quantization=quantization)
                
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=lora_target_modules,
                )
                model = get_peft_model(model, lora_config)
            else:
                logging.info(f"Loading pretrained LORA weights from {lora_weights_name_or_path}")
                model = PeftModel.from_pretrained(model, lora_weights_name_or_path)
            
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

        if self.model.config.vocab_size < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        
        self._no_split_modules = model._no_split_modules
        
        self.model = model
        self.tokenizer = tokenizer
        self.normalized = normalized
        self.pooling_method = pooling_method

        # Embedding loss
        self.emb_loss = losses.SupConLoss(
            temperature=0.05, 
            distance=distances.CosineSimilarity()
        )
        self.miner = miners.MultiSimilarityMiner(epsilon=0.2)

        # Generation loss
        self.gen_add_kwargs = {"return_dict": True}
        self.gen_loss_fn = NextTokenLoss(
                self.model.config.vocab_size, loss_gen_type, loss_gen_factor
            )

        # Required for accelerate DeepSpeed integration
        self.config = self.model.config

    def pooling(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None, recast: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [b, n, d]
            attention_mask: [b, n]
        """
        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        # Recasting performs slightly worse but saves 50% space
        if recast: return embedding.to(hidden_state.dtype)
        return embedding
    
    def encode(self, features: Dict):
        """
        features: {
            "input_ids": (bs x (1 + num_positive + num_negative), max_seq_len),
            "attention_mask": (bs x (1 + num_positive + num_negative), max_seq_len),
            "instruction_lens": (bs x (1 + num_positive + num_negative), ),
            }
        """
        if features is None: return None

        attention_mask = features['attention_mask'].clone() if 'attention_mask' in features else None
        instruction_lens = features['instruction_lens'] if 'instruction_lens' in features else None
        kwargs = {'input_ids': features.get('input_ids'), 
                  'attention_mask': attention_mask,
                  'is_causal': False}
        outs = self.model(**kwargs)[0] # (bs x (1 + num_positive + num_negative), max_seq_len, hidden_size)

        # Mask out the instruction tokens for pooling
        if instruction_lens is not None:
            # Make a new copy of attention mask to prevent in-place problems
            attention_mask = features['attention_mask'].clone()
            # Mask out the instruction tokens for pooling
            for i, l in enumerate(instruction_lens):
                attention_mask[i, :l] = 0
                # Make sure not all zeros - If this happens it is a bug
                assert attention_mask[i].sum() > 0, f"All 0: {attention_mask[i]}, l: {l}"
        
        reps = self.pooling(outs, attention_mask)

        # Normalize the representations
        if self.normalized: 
            in_dtype = reps.dtype
            # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
            return torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)
        return reps.contiguous()

    def forward(self,
                passage: Dict[str, torch.Tensor] = None,
                labels:  torch.Tensor = None, # (bs x (1 + num_positive + num_negative), )
                generative: Dict[str, torch.Tensor] = None,
                ):
        """
        passage: {
            "input_ids": (bs x (1 + num_positive + num_negative), max_seq_len),
            "attention_mask": (bs x (1 + num_positive + num_negative), max_seq_len),
            "instruction_lens": (bs x (1 + num_positive + num_negative), ),
            },
        labels: (bs x (1 + num_positive + num_negative), )
        generative: {
            "input_ids": (bs, max_seq_len),
            "attention_mask": (bs, max_seq_len),
            "labels": (bs, max_seq_len),
            "loss_weight_mask": (bs, max_seq_len),
            }
        """

        if generative is not None:
            gen_logits = self.model(input_ids=generative['input_ids'], 
                                    attention_mask=generative['attention_mask'], 
                                    **self.gen_add_kwargs).logits
            labels = generative.pop('labels')
            loss_weight_mask = generative.pop('loss_weight_mask')
            
            loss_gen = self.gen_loss_fn(labels, gen_logits, loss_weight_mask)
        else:
            loss_gen = None
        
        if passage is not None:
            reps = self.encode(passage) # (bs x (1 + num_positive + num_negative), hidden_size)
            hard_pairs = self.miner(reps, labels)
            loss_emb = self.emb_loss(reps, labels, hard_pairs)
        else:
            loss_emb = None

        loss = sum([x for x in [loss_emb, loss_gen] if x is not None])

        return EmLMTrainOutput(
            reps=reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )
    
    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)



            






        
        






        





