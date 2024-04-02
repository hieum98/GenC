from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from numpy import pad
import numpy as np
import scipy as sp
import torch
from torch import Tensor, neg
from torch.utils.data import Dataset
from lightning import LightningDataModule
from transformers import PreTrainedTokenizerBase, BatchEncoding
import datasets

from genc import special_tokens


class DataModule(LightningDataModule):
    """Base DataModule class for all datasets."""
    @abstractmethod
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
        """All settings that can't be determined at the time of instantiation need to be passed through here
        before any dataloaders can be accessed.
        """

    def setup(self, stage: str="") -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    

class DPOCDataset(Dataset):
    """An in-memory dataset for DPO and Constrastive joint training"""
    def __init__(
            self,
            data: datasets.Dataset,
            tokenizer: PreTrainedTokenizerBase,
            num_negative_samples: int,
            num_positive_samples: int,
            max_seq_length: int=512,
            prompt_loss_weight: float=0.02,
            base_bos: str = special_tokens.base_bos,
            user_bos: str = special_tokens.user_bos,
            user_eos: str = special_tokens.user_eos,
            embed_bos: str = special_tokens.embed_bos,
            embed_eos: str = special_tokens.embed_eos,
            assistant_bos: str = special_tokens.assistant_bos,
            assistant_eos: str = special_tokens.assistant_eos,
            ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.num_positive_samples = num_positive_samples
        self.num_negative_samples = num_negative_samples
        self.max_seq_length = max_seq_length
        self.prompt_loss_weight = prompt_loss_weight
        # This is only apply for 1-turn dialogues
        self.emb_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + embed_bos
        self.emb_example_format = self.emb_prompt_format + "{example}" + embed_eos
        self.gen_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + assistant_bos
        self.gen_example_format = self.gen_prompt_format + "{response}" + assistant_eos

    def __len__(self) -> int:
        return len(self.data)
    
    def tokenize_emb_example(
            self, 
            example: Tuple[str, str],
            ) -> BatchEncoding:
        emb_prompt = self.emb_prompt_format.format(prompt=example[0])
        emb_example = self.emb_example_format.format(prompt=example[0], example=example[1])
        model_inputs = self.tokenizer(
            text=emb_example,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False, # already added
        )
        
        # Find the prompt length
        prompt_ids = self.tokenizer(
            emb_prompt,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
            )["input_ids"]
        if len(prompt_ids) > len(model_inputs["input_ids"]):
            raise ValueError("Prompt is longer than the model input")
        model_inputs["prompt_length"] = len(prompt_ids)
        return model_inputs
    
    def tokenize_gen_example(
            self,
            gen_prompt: str,
            response: str,
            ) -> BatchEncoding:
        prompt = self.gen_prompt_format.format(prompt=gen_prompt)
        example = self.gen_example_format.format(prompt=gen_prompt, response=response)
        model_inputs = self.tokenizer(
            text=example,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False, # already added
        )

        # Add labels to the model inputs
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        # Compute loss weights mask
        prompt_ids = self.tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
            )["input_ids"]
        if len(prompt_ids) > len(model_inputs["input_ids"]):
            raise ValueError("Prompt is longer than the model input")
        prompt_len = len(prompt_ids)
        loss_weight_mask = np.ones(len(model_inputs["labels"]), dtype=np.float32)
        len_prompt = prompt_len
        len_result = len(model_inputs["labels"]) - len_prompt
        prompt_token_weight = len_result * self.prompt_loss_weight  # 'prompt_loss_weight' percent of the total loss
        try:
            prompt_token_weight = prompt_token_weight * (
                len_result / (len_result * (1 - self.prompt_loss_weight))
            )  # Scale so result tokens can have 1.0 weight
            prompt_token_weight = prompt_token_weight / len_prompt  # Divide by the number of prompt tokens
        except ZeroDivisionError:
            logging.warning(
                "Found division by zero in prompt token weight calculation. You might have an empty prompt, empty"
                f" result, or both.Setting prompt token weight to 0.0."
            )
            prompt_token_weight = 0.0
        loss_weight_mask[0: prompt_len] = prompt_token_weight
        model_inputs["loss_weight_mask"] = loss_weight_mask
        return model_inputs
    
    def __getitem__(self, index: int) -> Dict[str, BatchEncoding]:
        example = self.data[index]
        query: Tuple[str, str] = example['query']
        pos_passages: List[Tuple[str, str]] = example['pos']
        neg_passages: List[Tuple[str, str]] = example['neg']
        gen_prompt: str = example['gen_prompt']

        pos_passages = random.sample(pos_passages, self.num_positive_samples) if len(pos_passages) > self.num_positive_samples \
            else random.choices(pos_passages, k=self.num_positive_samples) # sample positive passages
        neg_passages = random.sample(neg_passages, self.num_negative_samples) if len(neg_passages) > self.num_negative_samples \
            else random.choices(neg_passages, k=self.num_negative_samples) # sample negative passages
        
        encoded_query = self.tokenize_emb_example(query)
        encoded_pos = [self.tokenize_emb_example(pos) for pos in pos_passages]
        encoded_neg = [self.tokenize_emb_example(neg) for neg in neg_passages]

        encoded_choices = [self.tokenize_gen_example(f"{gen_prompt}\n{query[1]}", pos[1]) for pos in pos_passages]
        encoded_rejects = [self.tokenize_gen_example(f"{gen_prompt}\n{query[1]}", neg[1]) for neg in neg_passages]

        return {
            "query": encoded_query, # + "promt_length"
            "pos": encoded_pos, # + "promt_length"
            "neg": encoded_neg, # + "promt_length"
            "choices": encoded_choices, # + ["labels", "loss_weight_mask"]
            "rejects": encoded_rejects, # + ["labels", "loss_weight_mask"]
        }


@dataclass
class DPOCCollator:
    """A collator for DPO and Constrastive joint training"""
    tokenizer: PreTrainedTokenizerBase=None
    label_pad_token_id: int=-100

    def pad_gen_example(self, generative: List[BatchEncoding]):
        labels = [item['labels'] for item in generative] if 'labels' in generative[0] else None
        loss_weight_mask = [item['loss_weight_mask'] for item in generative] if 'loss_weight_mask' in generative[0] else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            max_label_length = (
                (max_label_length + 8 - 1)
                // 8
                * 8
            )  # padding to multiple of 8
            padding_side = self.tokenizer.padding_side
            for item in generative:
                remainder = [self.label_pad_token_id] * (max_label_length - len(item["labels"]))
                if isinstance(item["labels"], list):
                    item["labels"] = (
                        item["labels"] + remainder if padding_side == "right" else remainder + item["labels"]
                    )
                elif padding_side == "right":
                    item["labels"] = np.concatenate([item["labels"], remainder]).astype(np.int64)
                else:
                    item["labels"] = np.concatenate([remainder, item["labels"]]).astype(np.int64)

        if loss_weight_mask is not None:
            max_label_length = max(len(l) for l in loss_weight_mask)
            max_label_length = (
                (max_label_length + 8 - 1)
                // 8
                * 8
            ) # padding to multiple of 8
            padding_side = self.tokenizer.padding_side
            for item in generative:
                remainder = [0.0] * (max_label_length - len(item["loss_weight_mask"]))
                if isinstance(item["loss_weight_mask"], list):
                    item["loss_weight_mask"] = (
                        item["loss_weight_mask"] + remainder if padding_side == "right" else remainder + item["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    item["loss_weight_mask"] = np.concatenate([item["loss_weight_mask"], remainder]).astype(np.float32)
                else:
                    item["loss_weight_mask"] = np.concatenate([remainder, item["loss_weight_mask"]]).astype(np.float32)
        
        generative = self.tokenizer.pad(
            generative, 
            pad_to_multiple_of=8,
            return_tensors="pt"
            )
        return generative

    def __call__(self, features) -> Dict[str, Tensor]:
        query = [f['query'] for f in features]
        pos = [f['pos'] for f in features]
        neg = [f['neg'] for f in features]
        choices = [f['choices'] for f in features]
        rejects = [f['rejects'] for f in features]

        bs = len(features)
        n_pos_per_query = len(pos[0])
        n_neg_per_query = len(neg[0])

        # Flatten the list of lists
        pos = [item for sublist in pos for item in sublist]
        neg = [item for sublist in neg for item in sublist]
        choices = [item for sublist in choices for item in sublist]
        rejects = [item for sublist in rejects for item in sublist]

        # Pad the inputs
        passages = query + pos + neg
        if 'label' not in query[0]:
            # create pos_label. For example, if there are 2 pos per query, the pos_label will be [0, 0, 1, 1, 2, 2] with batch size 3
            pos_labels = torch.arange(bs).repeat_interleave(n_pos_per_query)
            labels = torch.arange(len(passages))
            labels[bs: bs + n_pos_per_query*bs] = pos_labels
        else:
            labels = torch.tensor([item.pop('label') for item in passages])
        passages = self.tokenizer.pad(passages, pad_to_multiple_of=8, return_tensors="pt")
        n_choices = len(choices)
        n_rejects = len(rejects)
        pad_gen = self.pad_gen_example(choices + rejects)
        choices = {
            "input_ids": pad_gen["input_ids"][:n_choices],
            "attention_mask": pad_gen["attention_mask"][:n_choices],
            "labels": pad_gen["labels"][:n_choices],
            "loss_weight_mask": pad_gen["loss_weight_mask"][:n_choices]
        }
        rejects = {
            "input_ids": pad_gen["input_ids"][n_choices:],
            "attention_mask": pad_gen["attention_mask"][n_choices:],
            "labels": pad_gen["labels"][n_choices:],
            "loss_weight_mask": pad_gen["loss_weight_mask"][n_choices:]
        }

        # Form the batch
        query_input_ids = passages["input_ids"][:bs]
        query_attention_mask = passages["attention_mask"][:bs]
        query_labels = labels[:bs]
        query_prompt_length = passages["prompt_length"][:bs]

        pos_input_ids = passages["input_ids"][bs: bs + n_pos_per_query*bs].reshape(bs, n_pos_per_query, -1)
        pos_attention_mask = passages["attention_mask"][bs: bs + n_pos_per_query*bs].reshape(bs, n_pos_per_query, -1)
        pos_labels = labels[bs: bs + n_pos_per_query*bs].reshape(bs, n_pos_per_query)
        pos_prompt_length = passages["prompt_length"][bs: bs + n_pos_per_query*bs].reshape(bs, n_pos_per_query)

        neg_input_ids = passages["input_ids"][bs + n_pos_per_query*bs:].reshape(bs, n_neg_per_query, -1)
        neg_attention_mask = passages["attention_mask"][bs + n_pos_per_query*bs:].reshape(bs, n_neg_per_query, -1)
        neg_labels = labels[bs + n_pos_per_query*bs:].reshape(bs, n_neg_per_query)
        neg_prompt_length = passages["prompt_length"][bs + n_pos_per_query*bs:].reshape(bs, n_neg_per_query)

        choices_input_ids = choices["input_ids"].reshape(bs, n_pos_per_query, -1)
        choices_attention_mask = choices["attention_mask"].reshape(bs, n_pos_per_query, -1)
        choices_labels = choices["labels"].reshape(bs, n_pos_per_query, -1)
        choices_loss_weight_mask = choices["loss_weight_mask"].reshape(bs, n_pos_per_query, -1)

        rejects_input_ids = rejects["input_ids"].reshape(bs, n_neg_per_query, -1)
        rejects_attention_mask = rejects["attention_mask"].reshape(bs, n_neg_per_query, -1)
        rejects_labels = rejects["labels"].reshape(bs, n_neg_per_query, -1)
        rejects_loss_weight_mask = rejects["loss_weight_mask"].reshape(bs, n_neg_per_query, -1)

        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "query_labels": query_labels,
            "query_prompt_length": query_prompt_length,
            
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "pos_labels": pos_labels,
            "pos_prompt_length": pos_prompt_length,
            
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
            "neg_labels": neg_labels,
            "neg_prompt_length": neg_prompt_length,
            
            "choices_input_ids": choices_input_ids,
            "choices_attention_mask": choices_attention_mask,
            "choices_labels": choices_labels,
            "choices_loss_weight_mask": choices_loss_weight_mask,
            
            "rejects_input_ids": rejects_input_ids,
            "rejects_attention_mask": rejects_attention_mask,
            "rejects_labels": rejects_labels,
            "rejects_loss_weight_mask": rejects_loss_weight_mask,
        }
            

class EmbDataset(Dataset):
    def __init__(
            self,
            data: datasets.Dataset,
            tokenizer: PreTrainedTokenizerBase,
            max_seq_length: int=512,
            base_bos: str = special_tokens.base_bos,
            user_bos: str = special_tokens.user_bos,
            user_eos: str = special_tokens.user_eos,
            embed_bos: str = special_tokens.embed_bos,
            embed_eos: str = special_tokens.embed_eos,
            ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.emb_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + embed_bos
        self.emb_example_format = self.emb_prompt_format + "{example}" + embed_eos
    
    def __len__(self) -> int:
        return len(self.data)
    
    def tokenize_emb_example(
            self, 
            example: Tuple[str, str],
            ) -> BatchEncoding:
        emb_prompt = self.emb_prompt_format.format(prompt=example[0])
        emb_example = self.emb_example_format.format(prompt=example[0], example=example[1])
        model_inputs = self.tokenizer(
            text=emb_example,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False, # already added
        )
        
        # Find the prompt length
        prompt_ids = self.tokenizer(
            emb_prompt,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
            )["input_ids"]
        if len(prompt_ids) > len(model_inputs["input_ids"]):
            raise ValueError("Prompt is longer than the model input")
        model_inputs["prompt_length"] = len(prompt_ids)
        return model_inputs
    
    def __getitem__(self, index) -> BatchEncoding:
        """
        example: {
            'query': [prompt, example],
            'pos': [[prompt, example], ...],
            }
        """
        example = self.data[index]
        query = example['query']
        pos_passages = example['pos']

        encoded_query = self.tokenize_emb_example(query)
        encoded_pos = [self.tokenize_emb_example(pos) for pos in pos_passages]

        return {
            "query": encoded_query, # + "promt_length"
            "pos": encoded_pos, # + "promt_length"
            "idx": index,
            }


@dataclass
class EmbCollator:
    tokenizer: PreTrainedTokenizerBase=None

    def __call__(self, features) -> BatchEncoding:
        query = [f['query'] for f in features]
        pos = [f['pos'] for f in features]

        bs = len(features)
        n_pos_per_query = min([len(pos[i]) for i in range(bs)])

        # Flatten the list of lists
        pos = [item for sublist in pos for item in sublist[:n_pos_per_query]]

        query = self.tokenizer.pad(query, pad_to_multiple_of=8, return_tensors="pt")
        pos = self.tokenizer.pad(pos, pad_to_multiple_of=8, return_tensors="pt")

        return{
            "idx": torch.tensor([f['idx'] for f in features]),
            "query_input_ids": query["input_ids"],
            "query_attention_mask": query["attention_mask"],
            "query_prompt_length": query["prompt_length"],
            "pos_input_ids": pos["input_ids"].reshape(bs, n_pos_per_query, -1),
            "pos_attention_mask": pos["attention_mask"].reshape(bs, n_pos_per_query, -1),
            "pos_prompt_length": pos["prompt_length"].reshape(bs, n_pos_per_query),
        }

