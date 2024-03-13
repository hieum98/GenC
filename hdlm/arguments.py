from dataclasses import dataclass, field
import os
from typing import List, Optional
from transformers import TrainingArguments


@dataclass
class DataArguments:
    train_data: str = field(
        default=None,
        metadata={
            "help": """Path to folder or file with training data. 
                    If the path is a folder, for each minibatch all samples will come from one file in the folder. 
                    You can use this to ensure in-batch negatives are very difficult."""
        }
    )
    number_positives: int = field(
        default=1,
        metadata={"help": "Number of positive passages per query"}
    )
    number_negatives: int = field(
        default=15,
        metadata={"help": "Number of negative passages per query"}
    )
    query_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum tokens for the query. Sequences longer than this will be truncated."
        },
    )
    passage_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum tokens for passages (positives & negatives). Sequences longer than this will be truncated."
        },
    )
    generative_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for generative. Sequences longer than this will be truncated."
        },
    )
    max_example_num_per_dataset: int = field(
        default=100_000_000, metadata={"help": "the max number of examples for each dataset"}
    )
    num_samples: Optional[str] = field(
        default=None, metadata={"help": "path to json with number of samples per dataset"}
    )    
    prompt_loss_weight: float = field(
        default=0.005, metadata={"help": "weight for prompt loss"}
    )
    prefixlm: bool = field(default=False, metadata={"help": "PrefixLM for generative"})


    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
    attn_implementation: str = field(default='sdpa', metadata={"help": "eager/sdpa/flash_attention_2"})
    normalized: bool = field(default=True, metadata={"help": "Whether to normalize the representations"})
    pooling_method: str = field(default='weightedmean', metadata={"help": "Pooling method for sentences"})
    use_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use LoRA. If True, the model will be trained with LoRA: https://arxiv.org/abs/2106.09685"
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
    lora_target_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": (
                "The target modules to which LoRA will be applied. If not specified, We"
                " will use the default modules for the model in huggingface PEFT library."
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
        default=0.05,
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


@dataclass
class CustomTrainingArguments(TrainingArguments):
    mode: str = field(
        default='unified', 
        metadata={
            "help": "One of ['unified', 'embedding', 'generative']. For unified,"
            " `train_data` should point to a folder with both embedding and generative data."
        }
    )
    loss_gen_factor: float = field(default=1.0, metadata={"help": "Factor to scale generative loss by"})
    loss_gen_type: str = field(default="mixed", metadata={"help": "Type of gen loss: mixed/token"})
    quantization: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Whether to use '4' or '8' bit quantization. Requires bitsandbytes library:"
                " https://github.com/TimDettmers/bitsandbytes. This parameter is only used for training."
                )
            },
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
        default=1,
        metadata={"help": "The mini batch size for Gradcache."}
    )
    temperature: float = field(
        default=0.02, 
        metadata={
            "help": "Similarity will be sim = sim/temperature before using them to compute loss."
            " A higher temperature can reduce the value of similarity between texts in downstream tasks."
            }
            )
