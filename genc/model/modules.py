from typing import List, Optional, Union
import torch


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    loss_weight_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
            if loss_weight_mask is not None:
                loss = loss * loss_weight_mask
            return loss.mean()

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_weight_mask_chunks = [loss_weight_mask_chunk.reshape(-1) for loss_weight_mask_chunk in loss_weight_mask.split(logits[0].size(1), dim=1)] if loss_weight_mask is not None else 1
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none") * loss_weight_mask_chunk
            for logit_chunk, target_chunk, loss_weight_mask_chunk in zip(logit_chunks, target_chunks, loss_weight_mask_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        # See [non_masked_elems div note]
        return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
        if loss_weight_mask is not None:
            loss = loss * loss_weight_mask
        return loss.mean()

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_weight_mask_chunks = loss_weight_mask.split(chunk_size) if loss_weight_mask is not None else 1
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none") * loss_weight_mask_chunk
        for logit_chunk, target_chunk, loss_weight_mask_chunk in zip(logit_chunks, target_chunks, loss_weight_mask_chunks)
    ]
    non_masked_elems = (targets != ignore_index).sum()
    # [non_masked_elems div note]:
    #   max(1, non_masked_elems) would be more ergonomic to avoid a division by zero. However that
    #   results in a python int which is then passed back to torch division. By using the
    #   `x.maximum(torch.ones_like(x))` pattern we avoid a cudaStreamSynchronize.
    return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

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
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss_weight_mask = loss_weight_mask.view(-1) if loss_weight_mask is not None else None
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_weight_mask = loss_weight_mask.to(shift_logits.device) if loss_weight_mask is not None else None

        loss = chunked_cross_entropy(
            shift_logits, shift_labels, loss_weight_mask=loss_weight_mask, ignore_index=-100
        )
        return loss
        
