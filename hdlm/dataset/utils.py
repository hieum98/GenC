import logging
import multiprocessing
import random
from typing import List


def find_data_idx(dataset_lens: List[int], idx: int):
    """
    Find the index of the dataset that contains the idx-th element and the index of the element in the dataset.
    """
    cumsum = 0
    for i, d in enumerate(dataset_lens):
        if cumsum + d > idx:
            return i, idx - cumsum
        cumsum += d
    logging.warning(f"Index {idx} not found in the datasets {dataset_lens}, using random index instead.")
    data_idx = random.randint(0, len(dataset_lens)-1)
    return data_idx, random.randint(0, dataset_lens[data_idx]-1)

def filter_too_long_instructions(tokenizer, dataset, query_max_len, passage_max_len):
    def filter_fn(example):
        # Filter out super long examples to avoid tokenize taking forever
        if (len(example["query"][0]) > query_max_len * 10) or not(example["query"][1]):
            return False
        if len(tokenizer.tokenize(example["query"][0].strip("\t\n :"))) >= query_max_len - 5:
            return False
        for ex in example["pos"] + example["neg"]:
            if (len(ex[0]) > passage_max_len * 10) or not(ex[1]):
                return False
            if len(tokenizer.tokenize(ex[0].strip("\t\n :"))) >= passage_max_len - 5:
                return False
        return True
    num_proc = max(multiprocessing.cpu_count()-2, 1) if len(dataset) > 5000 else 1
    return dataset.filter(filter_fn, num_proc=num_proc, load_from_cache_file=True)

