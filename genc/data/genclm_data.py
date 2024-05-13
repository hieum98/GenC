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
from genc.data.utils import filter_too_long_instructions, quick_filter_too_long_instructions
from genc.special_tokens import SPECILA_TOKENS


@dataclass
class GenCLMDataset(DataModule):
    """
    DataModule for the MEDI dataset.
    """
    data_dir: str="dataset/GenCLM"
    val_file: str=None
    max_data_samples: int = -1

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
            pretrained_type="",
            ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.pretrained_type = pretrained_type
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        self.num_negative_samples = num_negative_samples
        self.num_positive_samples = num_positive_samples
        self.prompt_loss_weight = prompt_loss_weight
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length
        self.special_tokens = SPECILA_TOKENS[pretrained_type]
    
    def prepare_data(self):
        train_ds = []
        # create cache dir 
        Path(f"cache/{self.pretrained_type}_hard").mkdir(parents=True, exist_ok=True)
        print(f"Creating cache dir or loading from: cache/{self.pretrained_type}_hard")
        for file in self.train_files:
            ds = load_dataset('json', data_files=file, split='train')
            data_name = file.split('/')[-1].split('.')[0]
            # Filter out super long examples to avoid tokenize taking forever and save to cache
            ds = ds.filter(
                lambda ex: filter_too_long_instructions(ex, self.tokenizer, self.max_seq_length, self.special_tokens) if data_name!='msmarco' \
                    else quick_filter_too_long_instructions(ex, self.max_seq_length, self.special_tokens),
                num_proc=50,
                cache_file_name=f"cache/{self.pretrained_type}_hard/{data_name}_filtered.arrow",
            )         
        
        if self.val_file is not None:
            val_ds = load_dataset('json', data_files=self.val_file, split='train')

    def setup(self, stage: str = "") -> None:
        train_ds = []
        for file in self.train_files:
            ds = load_dataset('json', data_files=file, split='train')
            data_name = file.split('/')[-1].split('.')[0]
            ds = ds.filter(
                lambda ex: filter_too_long_instructions(ex, self.tokenizer, self.max_seq_length, self.special_tokens) if data_name!='msmarco' \
                    else quick_filter_too_long_instructions(ex, self.max_seq_length, self.special_tokens),
                num_proc=50,
                cache_file_name=f"cache/{self.pretrained_type}_hard/{data_name}_filtered.arrow",
                load_from_cache_file=True
            )
            if self.max_data_samples > 0:
                num_data_samples = min(len(ds) - 1, self.max_data_samples)
                ds = ds.train_test_split(train_size=num_data_samples)['train']
            train_ds.append(ds)
        
        if self.val_file is not None:
            val_ds = load_dataset('json', data_files=self.val_file, split='train')

        self.train_dataset = MultipleDPOCDataset(
            data=train_ds,
            tokenizer=self.tokenizer,
            num_negative_samples=self.num_negative_samples,
            num_positive_samples=self.num_positive_samples,
            max_seq_length=self.max_seq_length,
            prompt_loss_weight=self.prompt_loss_weight,
            special_tokens=self.special_tokens,
        )
        
        if self.val_file is not None:
            self.val_dataset = EmbDataset(
                data=val_ds,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                special_tokens=self.special_tokens,
            )
    
    def train_dataloader(self) -> DataLoader:
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
        num_workers = min(self.num_workers, max_num_worker_suggest)
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
            num_workers=num_workers,
            collate_fn=collator,
        )
    
    def val_dataloader(self) -> DataLoader:
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
        num_workers = min(self.num_workers, max_num_worker_suggest)
        if hasattr(self, 'val_dataset'):
            collator = EmbCollator(tokenizer=self.tokenizer)
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
            )
        else:
            raise ValueError("Validation dataset not found.")

