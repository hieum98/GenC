from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np
import faiss


def compute_metrics(
        emb_data, 
        labels
        ) -> Dict[str, float]:
    """
    Compute metrics for the embeddings.
    Args:
        emb_data: The embeddings to compute metrics on. [n, d]
        labels: The labels of the embeddings. [n]
    Returns:
        A dictionary containing the metrics.
    """
    # Normalize embeddings
    emb_data /= np.linalg.norm(emb_data, axis=1)[:, None]

    # Build a faiss index
    index = faiss.IndexFlatL2(emb_data.shape[1])
    index.add(emb_data)
    # Perform a search
    D, I = index.search(emb_data, 1000)  # We search the 1000 nearest neighbors

    # Compute metrics
    num_queries = emb_data.shape[0]
    ranks = np.zeros(num_queries, dtype=np.float32)
    reciprocal_ranks = np.zeros(num_queries, dtype=np.float32)
    for i in range(num_queries):
        sorted_topk = labels[I[i]]
        sorted_topk[0] = -1 # Ignore the query itself
        rank = np.where(sorted_topk == labels[i])[0][0] if labels[i] in sorted_topk else 1000
        ranks[i] = rank
        reciprocal_ranks[i] = 1.0 / (rank + 1)

    return_dict = {
        'R_at_1': np.sum(np.less_equal(ranks, 1)) / np.float32(num_queries),
        'R_at_5': np.sum(np.less_equal(ranks, 5)) / np.float32(num_queries),
        'MRR': np.mean(reciprocal_ranks)
    }

    return return_dict
