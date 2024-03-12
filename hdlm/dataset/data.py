from dataclasses import dataclass
import logging
import math
import random
from typing import Iterator, List, Tuple, Union
import numpy as np
import torch
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.utils import PaddingStrategy
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import datasets

from .utils import find_data_idx
from ..arguments import DataArguments


class CustomDatasetForSFT(torch.utils.data.Dataset):
    def __init__(self, 
                dataset: Union[datasets.Dataset, List[datasets.Dataset], Tuple[List[datasets.Dataset], List[datasets.Dataset]], Tuple[datasets.Dataset, datasets.Dataset]],
                args: DataArguments,
                mode: str = 'embedding',
                max_seq_len: int = 2048):
        
        if mode=='unifined':
            assert len(dataset) == 2, "Dataset should be a tuple of two datasets"
            emb_dataset, gen_dataset = dataset
        elif mode=='embedding':
            emb_dataset = dataset
            gen_dataset = None
        elif mode=='generative':
            gen_dataset = dataset
            emb_dataset = None
        else:
            raise ValueError(f"Mode {mode} not supported")
        
        if isinstance(emb_dataset, list):
            self.emb_dataset_size = [len(d) for d in emb_dataset]
            self.total_emb_size = sum(self.emb_dataset_size)
            self.emb_dataset = emb_dataset
        else:
            if emb_dataset is not None:
                self.emb_dataset_size = [len(emb_dataset)]
                self.total_emb_size = self.emb_dataset_size[0]
                self.emb_dataset = [emb_dataset]
            else:
                self.emb_dataset_size = 0
                self.total_emb_size = 0
                self.emb_dataset = None
        
        if isinstance(gen_dataset, list):
            self.gen_dataset_size = [len(d) for d in gen_dataset]
            self.total_gen_size = sum(self.gen_dataset_size)
            self.gen_dataset = gen_dataset
        else:
            if gen_dataset is not None:
                self.gen_dataset_size = [len(gen_dataset)]
                self.total_gen_size = self.gen_dataset_size[0]
                self.gen_dataset = [gen_dataset]
            else:
                self.gen_dataset_size = 0
                self.total_gen_size = 0
                self.gen_dataset = None
        
        assert self.emb_dataset is not None or self.gen_dataset is not None, "At least one of the datasets should not be None"

        self.total_size = max(self.total_emb_size, self.total_gen_size)
        
        self.args = args
        self.mode = mode
        # Too long items will be stuck in communication so cut them on the fly
        self.max_char_len = max_seq_len * 10
        
        self.set_indices()

    def set_indices(self):
        if self.total_emb_size > self.total_gen_size:
            indices_gen = list(range(self.total_gen_size))
            if torch.distributed.is_initialized():
                # world_size and rank are global (i.e. across all nodes and processes)
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                indices_gen = indices_gen[rank::world_size]
            # shuffle indices to make sure that the batch is not always the same
            self.indices_gen = set(random.shuffle(indices_gen))
        elif self.total_emb_size < self.total_gen_size:
            generator = torch.Generator()
            generator.manual_seed(0)
            # Create random indices for each dataset
            indices_emb = [torch.randperm(n, generator=generator).tolist() for n in self.emb_dataset_size]
            # Increase the indices to be indices of the concatenated dataset
            indices_emb = [[i + sum(self.emb_dataset_size[:j]) for i in indices_emb[j]] for j in range(len(self.emb_dataset_size))]
            if torch.distributed.is_initialized():
                # world_size and rank are global (i.e. across all nodes and processes)
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                indices_emb = indices_emb[rank::world_size]
            # shuffle indices to make sure that the batch is not always the same
            self.indices_emb = set(random.shuffle(indices_emb))

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        query, pos_passage, neg_passage, generative = None, None, None, None

        if self.mode in ['unifined', 'embedding'] and self.emb_dataset is not None:
            #TODO: Add label processing here
            if hasattr(self, 'indices_emb'):
                try:
                    idx_emb = self.indices_emb.pop()
                except:
                    idx_emb = random.randint(0, self.total_emb_size - 1)
            else:
                idx_emb = idx
            dataset_idx, idx_emb = find_data_idx(self.emb_dataset_size, idx_emb)
            emb_datapoint = self.emb_dataset[dataset_idx][idx_emb]
            
            query = emb_datapoint['query']
            if isinstance(query, str):
                query = query[:self.max_char_len]
            elif isinstance(query, list):
                query = [x[:self.max_char_len] for x in query]
            
            pos_passage = []
            if len(emb_datapoint['pos']) < self.args.number_positives:
                num = math.ceil(self.args.number_positives / len(emb_datapoint['pos']))
                pos = random.sample(emb_datapoint['pos']*num, self.args.number_positives)
            else:
                pos = random.sample(emb_datapoint['pos'], self.args.number_positives)
            for i, pos in enumerate(pos):
                if isinstance(pos, str):
                    pos = pos[:self.max_char_len]
                elif isinstance(pos, list):
                    pos = [x[:self.max_char_len] for x in pos]
                pos_passage.append(pos)
            
            neg_passage = []
            if len(emb_datapoint['neg']) < self.args.number_negatives:
                num = math.ceil(self.args.number_negatives / len(emb_datapoint['neg']))
                neg = random.sample(emb_datapoint['neg']*num, self.args.number_negatives)
            else:
                neg = random.sample(emb_datapoint['neg'], self.args.number_negatives)
            for i, neg in enumerate(neg):
                if isinstance(neg, str):
                    neg = neg[:self.max_char_len]
                elif isinstance(neg, list):
                    neg = [x[:self.max_char_len] for x in neg]
                neg_passage.append(neg)
        
        if self.mode in ['unifined', 'generative'] and self.gen_dataset is not None:
            if hasattr(self, 'indices_gen'):
                try:
                    idx_emb = self.indices_emb.pop()
                except:
                    idx_emb = random.randint(0, self.total_emb_size - 1)
            else:
                idx_gen = idx
            dataset_idx, idx_gen = find_data_idx(self.gen_dataset_size, idx_gen)
            gen_datapoint = self.gen_dataset[dataset_idx][idx_gen]
            generative = gen_datapoint['text']
        
        return {'query': query, 'pos_passage': pos_passage, 'neg_passage': neg_passage, 'generative': generative}
    

@dataclass
class CustomCollatorForSFT:
    tokenizer: PreTrainedTokenizer

    passage_max_len: int = 128
    generative_max_len: int = 128

    base_bos: str = ""
    turn_sep: str = ""
    user_bos: str = ""
    user_eos: str = ""
    embed_bos: str = ""
    embed_eos: str = ""
    assistant_bos: str = ""
    assistant_eos: str = ""
    prefixlm: bool = False

    prompt_loss_weight: float = 0.05

    padding: Union[bool, str, PaddingStrategy] = True
    label_pad_token_id: int = -100

    def prepare_example(self, example: Union[str, List[str]], is_emb: bool) -> BatchEncoding:
        if isinstance(example, (list, tuple)):
            if is_emb:
                prompt = self.base_bos + self.user_bos + example[0].strip("\t\n :") + self.user_eos + self.embed_bos if example[0].strip("\t\n :") \
                        else self.base_bos + self.embed_bos.lstrip()
                example = self.base_bos + self.user_bos + example[0].strip("\t\n :") + self.user_eos + self.embed_bos + example[1] + self.embed_eos if example[0].strip("\t\n :") \
                        else self.base_bos + self.embed_bos.lstrip() + example[1] + self.embed_eos
                model_inputs = self.tokenizer(
                    text=example,
                    max_length=self.passage_max_len,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                # Make sure the `eos_token_id` is added at the end
                # This bug is reported at https://github.com/huggingface/transformers/issues/22794
                if model_inputs["input_ids"][-1] != self.tokenizer.eos_token_id:
                    model_inputs["input_ids"].append(self.tokenizer.eos_token_id)
                    model_inputs["attention_mask"].append(1)

                # Find the promt length
                prompt = self.tokenizer(
                    text=prompt,
                    max_length=self.passage_max_len,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )["input_ids"]
                # Remove the last token if it is an eos token
                if prompt[-1] == self.tokenizer.eos_token_id:
                    prompt = prompt[:-1]
                
                if len(prompt) > len(model_inputs["input_ids"]):
                    raise ValueError(
                        f"Prompt is longer than the input, something went wrong. Prompt: {prompt}, input:"
                        f" {model_inputs['input_ids']}"
                    )
                model_inputs['instruction_lens'] = torch.tensor(len(prompt))
                #TODO: Add label processing here
            else:
                prompt_lens = [
                    len(self.tokenizer.tokenize(self.user_bos + z + self.user_eos + self.assistant_bos, add_special_tokens=False) if i > 0 
                        else self.tokenizer.tokenize(self.base_bos + self.user_bos + z + self.user_eos + self.assistant_bos, add_special_tokens=False)) if i % 2 == 0
                    else len(self.tokenizer.tokenize(z.strip() + self.assistant_eos, add_special_tokens=False))
                    for i, z in enumerate(example[:-1])
                ]
                example = self.base_bos + self.turn_sep.join([
                    self.user_bos + example[i] + self.user_eos + self.assistant_bos + example[i+1].strip() + self.assistant_eos for i in range(0, len(example), 2)
                ])

                model_inputs = self.tokenizer(
                    text=example,
                    max_length=self.generative_max_len,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                # Make sure the `eos_token_id` is added at the end
                # This bug is reported at https://github.com/huggingface/transformers/issues/22794
                if model_inputs["input_ids"][-1] != self.tokenizer.eos_token_id:
                    model_inputs["input_ids"].append(self.tokenizer.eos_token_id)
                    model_inputs["attention_mask"].append(1)
                model_inputs["labels"] = model_inputs["input_ids"].copy()

                # Find the promt weights
                if sum(prompt_lens) > len(model_inputs["labels"]):
                    raise ValueError(
                        f"Prompt is longer than the input, something went wrong. Prompt: {prompt}, input:"
                        f" {model_inputs['input_ids']}"
                    )
                
                # Create the weight mask
                loss_weight_mask = np.ones(len(model_inputs["labels"]), dtype=np.float32)
                len_prompt = sum(prompt_lens)
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
                        f" result, or both. Example with error: {example}. Setting prompt token weight to 0.0."
                    )
                    prompt_token_weight = 0.0
                
                # Fill the weight mask
                cur_len = 0
                for i, l in enumerate(prompt_lens):
                    if self.prefixlm:
                        loss_weight_mask[cur_len: cur_len+l] = 0.0
                    else:
                        if (i % 2) == 0:
                            loss_weight_mask[cur_len: cur_len+l] = prompt_token_weight
                    cur_len += l
                
                model_inputs['loss_weight_mask'] = loss_weight_mask
        else:
            model_inputs = self.tokenizer(
                    text=example,
                    max_length=self.generative_max_len,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
        
        return model_inputs      

    def __call__(self, features):
        query = [f['query'] for f in features]
        pos_passage = [f['pos_passage'] for f in features]
        neg_passage = [f['neg_passage'] for f in features]
        generative = [f['generative'] for f in features]

        # flatten if list of lists
        if isinstance(pos_passage[0], list):
            n_pos_passages = [len(p) for p in pos_passage] # (bs, )
            pos_passage = sum(pos_passage, [])
        if isinstance(neg_passage[0], list):
            n_neg_passages = [len(p) for p in neg_passage] # (bs, )
            neg_passage = sum(neg_passage, [])

        features = {}
        if query[0] is not None:
            #TODO: Add label processing here
            query = [self.prepare_example(item, is_emb=True) for item in query] # (bs, )
            pos_passage = [self.prepare_example(item, is_emb=True) for item in pos_passage] # (bs x n_pos_passage, )
            neg_passage = [self.prepare_example(item, is_emb=True) for item in neg_passage] # (bs x n_neg_passage, )
            passage = query + pos_passage + neg_passage
            padded_passage = self.tokenizer.pad(
                passage, 
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=8,
                return_tensors="pt"
                ) # {'input_ids': (bs + n_pos_passage + n_neg_passage, max_len), 'attention_mask': (bs + n_pos_passage + n_neg_passage, max_len), 'instruction_lens': (bs + n_pos_passage + n_neg_passage,) 'labels': (bs + n_pos_passage + n_neg_passage,)}

            if 'labels' not in passage[0]:
                bs = len(query)
                p = range(bs, bs + sum(n_pos_passages))
                a1 = [[i] * n_pos_passages[i] for i in range(bs)]
                a1 = sum(a1, [])
                p = torch.tensor(p)
                a1 = torch.tensor(a1)

                n = range(bs + sum(n_pos_passages), bs + sum(n_pos_passages) + sum(n_neg_passages))
                a2 = [[i] * n_neg_passages[i] for i in range(bs)]
                a2 = sum(a2, [])
                n = torch.tensor(n)
                a2 = torch.tensor(a2)
            else:
                labels = padded_passage['labels']
                a1, p, a2, n = lmu.get_all_pairs_indices(labels=labels)
                
            padded_passage['indices_tuple'] = (a1, p, a2, n)
            features['passage'] = padded_passage
        
        if generative[0] is not None:
            generative = [self.prepare_example(item, is_emb=False) for item in generative] # (bs, )
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
                padding=self.padding,
                max_length=self.generative_max_len,
                pad_to_multiple_of=8,
                return_tensors="pt"
                )
            features['generative'] = generative

            return features

            
@dataclass
class CustomRandomSampler(torch.utils.data.sampler.RandomSampler):
    """
    Sampler used when training on multiple datasets to ensure each 
    batch only contains samples from one dataset for the majority of cases.
    """
    total_batch_size: int = 8
    ds_lens: List[int] = None

    def __iter__(self) -> Iterator[int]:
        
        if not hasattr(self, "generator") or self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # We have multiple datasets each with a different number of samples
        # e.g. [100, 150, 50]
        # We would like to sample from them such that as much as possible each batch
        # only has samples from the same dataset.
        # For example if our batch size is 4 then
        # indices might be [0,1,2,3,100,101,102,103,150,151,152,153,50,51,52,53]
        # To do so:
        # 1. Shuffle the indices of each dataset separately
        # 2. Create batches with only samples from one dataset
        # 3. Keep the remaining samples which do not fit into a batch separate
        # 4. Then create mixed batches from the remaining samples
        # 5. Then yield randomly from all the batches
        # Testing:
        # ds_lens = [100, 150, 50]
        # batch_size = 8
        # Create random indices for each dataset
        ds_indices = [torch.randperm(n, generator=generator).tolist() for n in self.ds_lens]
        # Increase the indices to be indices of the concatenated dataset
        ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
        # Create batches with only samples from one dataset
        ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in ds_batches:
            if len(b[-1]) < self.total_batch_size:
                incomplete_indices.append(b.pop())

        if incomplete_indices:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=generator).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(torch.tensor(incomplete_indices), self.total_batch_size))
            if len(mixed_batches[-1]) < self.total_batch_size:
                mixed_batches.pop()
            # Merge all batches to look like [...tensor([259, 273, 284, 289]), tensor([262, 280, 295, 258]), ...]
            ds_batches = sum(ds_batches, []) + mixed_batches
            logging.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches) - len(mixed_batches)} single-dataset batches & {len(mixed_batches)} mixed dataset batches.")
        else:
            ds_batches = sum(ds_batches, [])
            logging.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches)} single-dataset batches.")

        # Randomly permute the order of all batches, then merge them to look like tensor([...259, 273, 284, 289, 262, 280, 295, 258...])
        order = torch.randperm(len(ds_batches), generator=generator).tolist()
        ds_batches = [int(i) for i in torch.cat([ds_batches[i] for i in order]).tolist()]
        # Yield the indices
        yield from ds_batches
    
            
            
            
