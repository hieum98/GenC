from dataclasses import dataclass, field
from json import load
import os
from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset

from genc.data.base import (
    DataModule,
    MultipleDPOCDataset,
    MutipleDPOCSampler,
    DPOCCollator,
    EmbDataset,
    EmbCollator
    )
from genc.data.utils import filter_too_long_example


@dataclass
class MEDIDataset(DataModule):
    """
    DataModule for the MEDI dataset.
    """
    data_dir: str="./dataset/MEDI2BGE"
    val_file: str=None

    seed: int = 42
    num_workers: int = 4
    ignore_index: int = -100

    tokenizer: Optional[PreTrainedTokenizerBase] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    num_negative_samples: int = field(default=1, init=False, repr=False)
    num_positive_samples: int = field(default=1, init=False, repr=False)
    prompt_loss_weight: float = field(default=0.02, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[Union[MultipleDPOCDataset]] = field(default=None, init=False, repr=False)
    val_dataset: Optional[EmbDataset] = field(default=None, init=False, repr=False)


    def __post_init__(self):
        # Get all files in the data directory
        self.train_files = [str(f) for f in Path(self.data_dir).glob('*.jsonl')]
        self.train_files.sort()

    def connect(
            self, 
            world_size: int = 1,
            global_rank: int = 0,
            tokenizer: PreTrainedTokenizerBase | None = None, 
            batch_size: int = 1, 
            global_batch_size: int = 1,
            max_seq_length = 512,
            num_negative_samples: int = 1,
            num_positive_samples: int = 1,
            prompt_loss_weight: float=0.02,
            ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        self.num_negative_samples = num_negative_samples
        self.num_positive_samples = num_positive_samples
        self.prompt_loss_weight = prompt_loss_weight
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length
    
    def prepare_data(self):
        def filter_too_long_instructions(example, tokenizer, max_seq_length):
            # Filter out super long examples to avoid tokenize taking forever
            if not filter_too_long_example(example['query'][0], max_seq_length) \
                or not example['query'][1] \
                or not filter_too_long_example(example['query'][1], max_seq_length):
                return False
            for ex in example['pos'] + example['neg']:
                if not filter_too_long_example(ex[0], max_seq_length) \
                    or not ex[1] \
                    or not filter_too_long_example(ex[1], max_seq_length):
                    return False
            return True
        train_ds = []
        for file in self.train_files:
            ds = load_dataset('json', data_files=file, split='train', cache_dir="cache")
            ds = ds.filter(
                lambda ex: filter_too_long_instructions(ex, self.tokenizer, self.max_seq_length),
                num_proc=os.cpu_count()//2 if os.cpu_count() > 10 else 10,
                load_from_cache_file=True,
            )
            train_ds.append(ds)
        
        if self.val_file is not None:
            val_ds = load_dataset('json', data_files=self.val_file, split='train', cache_dir="cache")

    def setup(self, stage: str = "") -> None:
        def filter_too_long_instructions(example, tokenizer, max_seq_length):
            # Filter out super long examples to avoid tokenize taking forever
            if not filter_too_long_example(example['query'][0], max_seq_length) \
                or not example['query'][1] \
                or not filter_too_long_example(example['query'][1], max_seq_length):
                return False
            for ex in example['pos'] + example['neg']:
                if not filter_too_long_example(ex[0], max_seq_length) \
                    or not ex[1] \
                    or not filter_too_long_example(ex[1], max_seq_length):
                    return False
            return True
        train_ds = []
        for file in self.train_files:
            ds = load_dataset('json', data_files=file, split='train', cache_dir="cache")
            ds = ds.filter(
                lambda ex: filter_too_long_instructions(ex, self.tokenizer, self.max_seq_length),
                num_proc=os.cpu_count()//2 if os.cpu_count() > 10 else 10,
                load_from_cache_file=True,
            )
            train_ds.append(ds)
        
        if self.val_file is not None:
            val_ds = load_dataset('json', data_files=self.val_file, split='train', cache_dir="cache")

        self.train_dataset = MultipleDPOCDataset(
            data=train_ds,
            tokenizer=self.tokenizer,
            num_negative_samples=self.num_negative_samples,
            num_positive_samples=self.num_positive_samples,
            max_seq_length=self.max_seq_length,
            prompt_loss_weight=self.prompt_loss_weight,
        )
        
        if self.val_file is not None:
            self.val_dataset = EmbDataset(
                data=val_ds,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
            )
    
    def train_dataloader(self) -> DataLoader:
        collator = DPOCCollator(tokenizer=self.tokenizer, label_pad_token_id=self.ignore_index)
        sampler = MutipleDPOCSampler(
            dataset=self.train_dataset,
            global_batch_size=self.global_batch_size,
            shuffle=True,
            num_replicas=self.world_size,
            rank=self.global_rank,
            seed=self.seed, 
            drop_last=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collator,
        )
    
    def val_dataloader(self) -> DataLoader:
        if hasattr(self, 'val_dataset'):
            collator = EmbCollator(tokenizer=self.tokenizer)
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collator,
            )
        else:
            raise ValueError("Validation dataset not found.")

