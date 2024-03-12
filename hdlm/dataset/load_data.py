import json
import logging
import multiprocessing
import os
import random
import datasets
from transformers import PreTrainedTokenizerBase

from .utils import filter_too_long_instructions
from ..arguments import DataArguments

logger = logging.getLogger(__name__)



def load_data_for_sft(data_args: DataArguments,
                      tokenizer: PreTrainedTokenizerBase,):
    # Load datasets
    data_files = [os.path.join(data_args.train_data, x) for x in os.listdir(data_args.train_data)] if os.path.isdir(data_args.train_data) else [data_args.train_data]
    num_samples = None
    if data_args.num_samples:
        with open(data_args.num_samples, "r") as f:
            num_samples = json.load(f)

    if data_args.generative_max_len is None:
        data_args.generative_max_len = data_args.passage_max_len
    
    ds_name_to_samples = {}
    emb_train_ds = []
    gen_train_ds = []
    ds_embedding_lens = []
    for file in data_files:
        logger.info("Loading dataset %s", file)
        tmp_ds = datasets.load_dataset('json', data_files=file, split='train')
        tmp_ds_len = len(tmp_ds)
        # For testing, can add an origin column:
        # origin_col = [file] * len(tmp_ds)
        # tmp_ds = tmp_ds.add_column("origin", origin_col)
        if tmp_ds_len > data_args.max_example_num_per_dataset:
            tmp_ds = tmp_ds.select(
                random.sample(list(range(tmp_ds_len)), data_args.max_example_num_per_dataset)
            )
        
        # Check if has instructions separated such that they will be masked out later
        # If so filter out samples where the instructions are too long else they will all be 0s
        if "query" in tmp_ds.features:
            if isinstance(tmp_ds[0]['query'], (tuple, list)):
                logger.info(f"Filtering out embedding samples with too long instructions for {file}")
                tmp_ds = filter_too_long_instructions(
                    tokenizer,
                    tmp_ds,
                    data_args.query_max_len,
                    data_args.passage_max_len,
                )
                if num_samples:
                    assert file.split("/")[-1] in num_samples, f'Missing num_samples for {file.split("/")[-1]}'
                    tmp_ds_len = len(tmp_ds)
                    samples = num_samples[file.split("/")[-1]]
                    if tmp_ds_len > samples:                    
                        tmp_ds = tmp_ds.select(random.sample(list(range(tmp_ds_len)), samples))
            ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
            ds_embedding_lens.append(len(tmp_ds))
            emb_train_ds.append(tmp_ds)
            continue
        if "text" in tmp_ds.features:
            if isinstance(tmp_ds[0]['text'], (tuple, list)):
                logger.info(f"Filtering out generative samples with too long instructions for {file}")
                # Use passage_max_len, as this is the seq len limit for the entire generative snippet
                num_proc = max(multiprocessing.cpu_count()-2, 1) if tmp_ds_len > 5000 else 1
                tmp_ds = tmp_ds.filter(
                    lambda ex: len(tokenizer.tokenize(ex["text"][0])) < data_args.generative_max_len - 5,
                    num_proc=num_proc,
                    load_from_cache_file=True,
                )
            ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
            gen_train_ds.append(tmp_ds)
            continue
        logger.info("Skipping dataset %s as its type could not be identified", file)
    
    return emb_train_ds, gen_train_ds, ds_name_to_samples, ds_embedding_lens

    
