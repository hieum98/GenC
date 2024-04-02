from dataclasses import dataclass, field
import os
from typing import List, Optional
from transformers import TrainingArguments


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .jsonl files for the task."}
        )
    train_file: str = field(
        metadata={"help": "The input training data file (a jsonl file)."}
        )
    val_file: str = field(
        default="msmarco_dev.jsonl",
        metadata={"help": "The input validation data file (a jsonl file)."}
        )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input test data file (a jsonl file)."},
        )
    ignore_index: int = field(
        default=-100,
        metadata={"help": "The index to ignore in the loss."},
        )
    num_workers: int = field(
        default=os.cpu_count()//2,
        metadata={"help": "Number of workers for dataloader."},
        )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
    use_bidirectional: bool = field(default=False, metadata={"help": "Whether to use bidirectional attention in compute encodings"})
    attn_implementation: str = field(default='sdpa', metadata={"help": "eager/sdpa/flash_attention_2"})
    normalized: bool = field(default=True, metadata={"help": "Whether to normalize the representations"})
    pooling_method: str = field(default='weightedmean', metadata={"help": "Pooling method for passage. One of ['cls', 'lasttoken', 'mean', 'weightedmean']"})
    loss_gen_type: str = field(default="mixed", metadata={"help": "Type of gen loss: mixed/token"})
    temperature: float = field(
        default=0.02, 
        metadata={
            "help": "Similarity will be sim = sim/temperature before using them to compute loss."
            " A higher temperature can reduce the value of similarity between texts in downstream tasks."
            }
            )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use LoRA. If True, the model will be trained with LoRA: https://arxiv.org/abs/2106.09685"
                )
            },
        )
    train_adapter_name: Optional[str] = field(
        default="default",
        metadata={
            "help": (
                "The name of the adapter to train. If not specified, the model will be trained from scratch."
                )
            },
        )
    lora_weights_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If the model has been trained with LoRA, "
                "path or huggingface hub name or local path to the pretrained weights."
                )
            },
        )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": "Lora attention dimension."},
        )
    lora_alpha: Optional[float] = field(
        default=64,
        metadata={"help": "The alpha parameter for Lora scaling."},
        )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "The dropout probability for Lora layers."},
        )
    torch_dtype: Optional[str] = field(
        default='bfloat16',
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this"
                " dtype. If `auto` is passed, the dtype will be automatically derived"
                " from the model's weights. We will override this if we use quantization."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
            },
        )
    quantization: Optional[int] = field(
        default=None,
        metadata={ "help": (
                "Whether to use '4' or '8' bit quantization. Requires bitsandbytes library:"
                " https://github.com/TimDettmers/bitsandbytes. This parameter is only used for training."
                )},
        )


@dataclass
class TrainingArguments():
    """
    TrainingArguments class to include additional arguments for training.
    """
    seed: int = field(
        default=2708,
        metadata={"help": "Random seed."},
        )
    nodes: int = field(
        default=1,
        metadata={"help": "Number of nodes to use for training."},
        )
    devices: int = field(
        default=1,
        metadata={"help": "Number of devices to use for training."},
        )
    logger_name: str = field(
        default="wandb",
        metadata={"help": "The name of the logger to use for logging. Should be one of ['wandb', 'tensorboard', 'csv']"},
        )
    precision: str = field(
        default="bf16-true",
        metadata={"help": "Precision to use for training. Should be one of ['bf16-true', '16-true', '32-true']"},
        )
    quantize: str = field(
        default=None,
        metadata={"help": "Quantization to use for training. Should be one of ['nf4', 'nf4-dq', 'fp4', 'fp4-dq', 'int8', 'int8-training']"},
        )
    save_interval: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of optimizer steps between saving checkpoints"},
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "Number of optimizer steps between logging training metrics"},
    )
    global_batch_size: int = field(
        default=32,
        metadata={"help": "Global batch size for training"},
    )
    use_gc: Optional[bool] = field(
            default=False,
            metadata={
                "help": (
                    "Whether to use Gradcache. If True, the model will be trained with Gradcache"
                    )
                },
            )
    gc_mini_batch_size: Optional[int] = field(
        default=2,
        metadata={"help": "The mini batch size for Gradcache."}
    )
    use_miner: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use MultiSimilarityMiner. If True, the model will be trained with MultiSimilarityMiner"
            )
        },
    )
    topk_neg: Optional[int] = field(
        default=8,
        metadata={"help": "The number of negative samples to consider for KL divergence loss"},
    )
    dpo_loss_type: Optional[str] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to compute. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']."},
    )
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta value for DPO loss"},
    )
    strategy: Optional[str] = field(
        default="fsdp",
        metadata={"help": "The strategy to use for training. Should be one of ['auto', 'ddp', 'deepspeed', 'fsdp']"},
    )
    prompt_loss_weight: float = field(
        default=0.05,
        metadata={"help": "The weight for prompt loss."},
        )
    num_negative_samples: int = field(
        default=8,
        metadata={"help": "The number of negative samples to consider for constrastive loss."},
        )
    num_positive_samples: int = field(
        default=1,
        metadata={"help": "The number of positive samples to consider for contrastive loss."},
        )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."},
        )
    mode: str = field(
        default='dpoc',
        metadata={"help": "The mode of training. Should be one of ['dpoc', 'emb']"},
        )
    output_dir: str = field(
        default="output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
        )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing."},
        )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "The number of epochs to train."},
        )
    max_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, the total number of training steps to perform. Overrides num_train_epochs."},
        )
    mini_batch_size: int = field(
        default=2,
        metadata={"help": "The mini batch size for training."},
        )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Linear warmup over warmup_steps."},
        )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for the optimizer."},
        )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply."},
        )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for the Adam optimizer."},
        )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for the Adam optimizer."},
        )

    def gradient_accumulation_iters(self, devices: int) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = self.batch_size(devices) // self.gc_mini_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // devices
        assert batch_size > 0
        return batch_size

@dataclass
class ValidationArgument():
    """
    ValidationArgument class to include additional arguments for validation.
    """
    interval: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of optimizer steps between evaluation calls"},
    )
    max_iters: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of iterations to run validation for"},
    )
