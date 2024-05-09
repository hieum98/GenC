from typing import List, Optional, Union
import torch

from genc.trainer.trainer_utils import chunked_cross_entropy


class NextTokenLoss:
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        
    def __call__(self, 
                 labels, 
                 logits,
                 loss_weight_mask=None,):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_weight_mask = loss_weight_mask[..., 1:].contiguous() if loss_weight_mask is not None else None
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_weight_mask = loss_weight_mask.to(shift_logits.device) if loss_weight_mask is not None else None

        loss = chunked_cross_entropy(
            shift_logits, shift_labels, loss_weight_mask=loss_weight_mask, ignore_index=-100
        )
        return loss
        
