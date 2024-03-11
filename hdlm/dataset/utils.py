import logging
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
