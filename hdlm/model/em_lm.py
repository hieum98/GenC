from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import torch 
from transformers import (AutoConfig, 
                          MistralPreTrainedModel,
                          MistralForCausalLM,)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.file_utils import ModelOutput
from pytorch_metric_learning import losses, miners, distances

from .modules import NextTokenLoss

@dataclass
class EmLMTrainOutput(ModelOutput):
    reps: Optional[torch.Tensor] = None
    gen_outputs: Optional[CausalLMOutputWithPast] = None
    loss: Optional[torch.Tensor] = None
    loss_emb: Optional[torch.Tensor] = None
    loss_gen: Optional[torch.Tensor] = None


class MistralEmbeddingLM(MistralForCausalLM):
    def __init__(
            self,
            config: AutoConfig,
            normalized: bool = True,
            pooling_method: str = 'mean', # One of ['cls', 'lasttoken', 'mean', 'weightedmean']
            loss_gen_type: str = "mixed",
            loss_gen_factor: float = 1.0,
            temperature: float = 0.05,
            ) -> None:

        super().__init__(config)
        # self.model = MistralForCausalLM(config)
        self.normalized = normalized
        self.pooling_method = pooling_method

        # Embedding loss
        self.emb_loss = losses.NTXentLoss(
            temperature=temperature, 
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
            "input_ids": ((bs  + num_positive + num_negative), max_seq_len),
            "attention_mask": ((bs  + num_positive + num_negative), max_seq_len),
            "instruction_lens": ((bs  + num_positive + num_negative), ),
            }
        """
        if features is None: return None

        attention_mask = features['attention_mask'].clone() if 'attention_mask' in features else None
        instruction_lens = features['instruction_lens'] if 'instruction_lens' in features else None
        kwargs = {'input_ids': features.get('input_ids'), 
                  'attention_mask': attention_mask,
                  'is_causal': False}
        outs = self.model(**kwargs)[0] # ((bs  + num_positive + num_negative), max_seq_len, hidden_size)

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
                labels:  torch.Tensor = None, # ((bs  + num_positive + num_negative), )
                generative: Dict[str, torch.Tensor] = None,
                ):
        """
        passage: {
            "input_ids": ((bs + num_positive + num_negative), max_seq_len),
            "attention_mask": ((bs + num_positive + num_negative), max_seq_len),
            "instruction_lens": ((bs + num_positive + num_negative), ),
            "labels": ((bs + num_positive + num_negative), )
            "indices_tuple": Tuple of tensors (a1_idx, p_idx, a2_idx, n_idx). The first 2 tensors are the indices which form all positive pairs. The second 2 tensors are the indices which form all negative pairs
            },
        generative: {
            "input_ids": (bs, max_seq_len),
            "attention_mask": (bs, max_seq_len),
            "labels": (bs, max_seq_len),
            "loss_weight_mask": (bs, max_seq_len),
            }
        """

        if generative is not None:
            gen_outputs = super().forward(input_ids=generative['input_ids'], 
                                    attention_mask=generative['attention_mask'], 
                                    **self.gen_add_kwargs)
            gen_logits = gen_outputs.logits
            labels = generative.pop('labels')
            loss_weight_mask = generative.pop('loss_weight_mask')
            
            loss_gen = self.gen_loss_fn(labels, gen_logits, loss_weight_mask)
        else:
            loss_gen = None
            gen_outputs = None
        
        if passage is not None:
            indicates = passage.pop('indices_tuple', None)
            labels  = passage.pop('labels', None)
            reps = self.encode(passage) # ((bs + num_positive + num_negative), hidden_size)
            if labels is not None:
                hard_pairs = self.miner(reps, labels)
                loss_emb = self.emb_loss(reps, labels, hard_pairs)
            elif indicates is not None:
                loss_emb = self.emb_loss(reps, indices_tuple=indicates)
            else:
                loss_emb = None
        else:
            reps = None
            loss_emb = None

        loss = sum([x for x in [loss_emb, loss_gen] if x is not None])

        return EmLMTrainOutput(
            reps=reps,
            gen_outputs=gen_outputs,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )
    
    def emb_loss_fn(self, reps, labels=None, indicates=None):
        if labels is not None:
                hard_pairs = self.miner(reps, labels)
                loss_emb = self.emb_loss(reps, labels, hard_pairs)
        elif indicates is not None:
            loss_emb = self.emb_loss(reps, indices_tuple=indicates)
        else:
            raise ValueError("Either labels or indicates must be provided")
        return loss_emb
    
    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)



            






        
        






        





