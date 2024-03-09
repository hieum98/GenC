from dataclasses import dataclass, field
import os
from typing import Optional


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
