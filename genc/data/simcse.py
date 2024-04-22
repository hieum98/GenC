from calendar import c
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
    DPOCDataset,
    DPOCCollator,
    EmbDataset,
    EmbCollator
    )
from genc.data.utils import filter_too_long_example, filter_too_long_instructions

def get_dataloader(
    data_files: Path,
    tokenizer: PreTrainedTokenizerBase,
    is_train: bool=True,
    mode: str='dpoc',
    max_seq_length: int=512,
    num_negative_samples: int=1,
    num_positive_samples: int=1,
    prompt_loss_weight: float=0.02,
    ignore_index: int=-100,
    batch_size: int=1,
    num_workers: int=4,
    seed: int=2708,
) -> DataLoader:
    ds = load_dataset('json', data_files=data_files, split='train')
    if is_train:
        ds = ds.filter(
            lambda ex: filter_too_long_instructions(ex, tokenizer, max_seq_length),
            num_proc=20,
        )
        if mode=='dpoc':
            ds = DPOCDataset(
                data=ds,
                tokenizer=tokenizer,
                num_negative_samples=num_negative_samples,
                num_positive_samples=num_positive_samples,
                max_seq_length=max_seq_length,
                prompt_loss_weight=prompt_loss_weight,
            )
            collator = DPOCCollator(tokenizer=tokenizer, label_pad_token_id=ignore_index)
        else:
            raise ValueError(f"Mode {mode} not supported.")
    else:
        ds = EmbDataset(
            data=ds,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        collator = EmbCollator(tokenizer=tokenizer)

    # For distributed training, use DistributedSampler
    if torch.distributed.is_available() and torch.distributed.is_initialized() and is_train:
        sampler = DistributedSampler(ds, seed=seed, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        sampler=sampler,
    )
    return dataloader
    

@dataclass
class SimCSEDataset(DataModule):
    """SimCSE dataset module."""
    data_dir: Path=Path("dataset/simcse")
    val_file: str="dataset/msmarco/msmarco_test.jsonl"

    seed: int = 42
    num_workers: int = 4
    ignore_index: int = -100

    tokenizer: Optional[PreTrainedTokenizerBase] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    num_negative_samples: int = field(default=1, init=False, repr=False)
    num_positive_samples: int = field(default=1, init=False, repr=False)
    prompt_loss_weight: float = field(default=0.02, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[Union[DPOCDataset]] = field(default=None, init=False, repr=False)
    val_dataset: Optional[EmbDataset] = field(default=None, init=False, repr=False)


    def __post_init__(self):
        self.train_files = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.jsonl')]
        if len(self.train_files) == 1:
            self.train_files = self.train_files[0]
        elif len(self.train_files) == 0:
            raise ValueError(f"No training files found in {self.data_dir}")

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
        self.num_negative_samples = num_negative_samples
        self.num_positive_samples = num_positive_samples
        self.prompt_loss_weight = prompt_loss_weight
        self.max_seq_length = 512 if max_seq_length is None else max_seq_length
    
    def prepare_data(self):
        # to make the cache file            
        train_ds = load_dataset('json', data_files=self.train_files, split='train')
        train_ds = train_ds.filter(
            lambda ex: filter_too_long_instructions(ex, self.tokenizer, self.max_seq_length),
            num_proc=20,
        )
        val_ds = load_dataset('json', data_files=self.val_file, split='train')

    def setup(self, stage: str = "") -> None:
        train_ds = load_dataset('json', data_files=self.train_files, split='train')
        train_ds = train_ds.filter(
            lambda ex: filter_too_long_instructions(ex, self.tokenizer, self.max_seq_length),
            num_proc=20,
        )
        val_ds = load_dataset('json', data_files=self.val_file, split='train')

        self.train_dataset = DPOCDataset(
            data=train_ds,
            tokenizer=self.tokenizer,
            num_negative_samples=self.num_negative_samples,
            num_positive_samples=self.num_positive_samples,
            max_seq_length=self.max_seq_length,
            prompt_loss_weight=self.prompt_loss_weight,
        )
        
        self.val_dataset = EmbDataset(
            data=val_ds,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
        )
    
    def train_dataloader(self) -> DataLoader:
        collator = DPOCCollator(tokenizer=self.tokenizer, label_pad_token_id=self.ignore_index)
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset, 
                seed=self.seed, 
                shuffle=True, 
                num_replicas=self.world_size,
                rank=self.global_rank,)
        else:
            sampler = None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collator,
        )
    
    def val_dataloader(self) -> DataLoader:
        collator = EmbCollator(tokenizer=self.tokenizer)
        return DataLoader(
            self.val_dataset,
            batch_size=8, # Fixed batch size for evaluation
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
        )


