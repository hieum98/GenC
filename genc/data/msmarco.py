from calendar import c
from dataclasses import dataclass, field
from json import load
import os
from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset

from genc.data.base import (
    DataModule,
    DPOCDataset,
    DPOCCollator,
    EmbDataset,
    EmbCollator
    )
from genc.data.utils import filter_too_long_example


@dataclass
class MSMARCODataset(DataModule):
    """MS MARCO dataset module."""
    data_dir: Path=Path("./dataset/msmarco")
    train_file: str="msmarco_hard.jsonl"
    val_file: str="msmarco_dev.jsonl"

    seed: int = 42
    num_workers: int = 4
    ignore_index: int = -100
    mode: str = 'dpoc'

    tokenizer: Optional[PreTrainedTokenizerBase] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    num_negative_samples: int = field(default=1, init=False, repr=False)
    num_positive_samples: int = field(default=1, init=False, repr=False)
    prompt_loss_weight: float = field(default=0.02, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[Union[DPOCDataset]] = field(default=None, init=False, repr=False)
    val_dataset: Optional[EmbDataset] = field(default=None, init=False, repr=False)


    def __post_init__(self):
        self.train_file = os.path.join(self.data_dir, self.train_file)
        self.val_file = os.path.join(self.data_dir, self.val_file)

    def connect(
            self, 
            tokenizer: PreTrainedTokenizerBase | None = None, 
            batch_size: int = 1, 
            max_seq_length = 512,
            mode: str = 'dpoc',
            num_negative_samples: int = 1,
            num_positive_samples: int = 1,
            prompt_loss_weight: float=0.02,
            ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_negative_samples = num_negative_samples
        self.num_positive_samples = num_positive_samples
        self.prompt_loss_weight = prompt_loss_weight
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length
        self.mode = mode
    
    def prepare_data(self):
        pass

    def setup(self, stage: str = "") -> None:
        def filter_too_long_instructions(example, tokenizer, max_seq_length):
            # Filter out super long examples to avoid tokenize taking forever
            if not filter_too_long_example(tokenizer, example['query'][0], max_seq_length) \
                or not example['query'][1] \
                or not filter_too_long_example(tokenizer, example['query'][1], max_seq_length):
                return False
            for ex in example['pos'] + example['neg']:
                if not filter_too_long_example(tokenizer, ex[0], max_seq_length) \
                    or not ex[1] \
                    or not filter_too_long_example(tokenizer, ex[1], max_seq_length):
                    return False
            return True
                
        train_ds = load_dataset('json', data_files=self.train_file, split='train')
        train_ds = train_ds.filter(
            lambda ex: filter_too_long_instructions(ex, self.tokenizer, self.max_seq_length),
            num_proc=20,
            load_from_cache_file=True
        )
        val_ds = load_dataset('json', data_files=self.val_file, split='train')

        if self.mode == 'dpoc':
            self.train_dataset = DPOCDataset(
                data=train_ds,
                tokenizer=self.tokenizer,
                num_negative_samples=self.num_negative_samples,
                num_positive_samples=self.num_positive_samples,
                max_seq_length=self.max_seq_length,
                prompt_loss_weight=self.prompt_loss_weight,
            )
        else:
            raise ValueError(f"Mode {self.mode} not supported.")
        
        self.val_dataset = EmbDataset(
            data=val_ds,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
        )
    
    def train_dataloader(self) -> DataLoader:
        if self.mode == 'dpoc':
            collator = DPOCCollator(tokenizer=self.tokenizer, label_pad_token_id=self.ignore_index)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=collator,
        )
    
    def val_dataloader(self) -> DataLoader:
        collator = EmbCollator(tokenizer=self.tokenizer)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
        )

