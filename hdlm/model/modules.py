import torch


class NextTokenLoss:
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        # How to weight loss:
        # a) Each sample gets the same weight (e.g. used in BLOOMZ https://arxiv.org/abs/2211.01786)
        # -> This leads to shorter generations, as short outputs get the same weight as long ones
        # -> The loss curves for this are unstable if there's noisy or very hard short samples
        # b) Each token gets the same weight
        # -> This leads to longer generations, as long outputs get more weight than short ones
        # b.1) Each token gets the same weight globally across batches
        # -> Just sum the loss as is, optionally divide it by a constant. If using Adam, the scale
        # of the loss doesn't matter, so this is only to balance with another loss like in our case.
        # b.2) Each token gets the same weight per batch
        # -> Divide by the number of tokens in the batch
        # Problem: If distributed training, needs all gather of number of tokens on each process        
        # c) Mix of a) and b) which is what you do if you use the loss in transformers as is and 
        # then do grad acc/multi-gpu with bs>1 
        # (https://github.com/huggingface/transformers/issues/24725; https://github.com/pytorch/pytorch/issues/72047)
        self.loss_gen_factor = loss_gen_factor
        self.loss_gen_type = loss_gen_type
        if loss_gen_type == "token": # b.1)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        elif loss_gen_type == "mixed": # c)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        else:
            raise ValueError(f"Invalid loss_gen_type: {loss_gen_type}")
        
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
        # Normalize by number of non-ignored tokens
        if self.loss_gen_type == "token":
            loss = self.cross_entropy(shift_logits, shift_labels)
            if loss_weight_mask is not None:
                loss = torch.sum(loss * loss_weight_mask) / torch.sum(loss_weight_mask)
            else:
                loss = torch.sum(loss) / labels.size(0)
            return loss * self.loss_gen_factor
        elif self.loss_gen_type == "mixed":
            return self.cross_entropy(shift_logits, shift_labels) * self.loss_gen_factor