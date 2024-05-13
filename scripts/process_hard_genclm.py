from copy import deepcopy
from json import load
import os
import random
from typing import Dict, List
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


def pooling(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None] # (batch_size, hidden_size)

def get_embedding(examples: List[str]):
    inputs = tokenizer(examples, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: inputs[key].to(device) for key in inputs}
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs['last_hidden_state']
        attention_mask = inputs['attention_mask']
        embs = pooling(last_hidden_states, attention_mask).detach().cpu().numpy()
    return {'embeddings': [embs[i] for i in range(len(examples))]}

def process_data(example: Dict):
    pos = example['pos']
    neg = example['neg']

    # Retrieve the hard negatives
    emb = example['embeddings']
    emb = np.array(emb)
    _score, retrieved_examples = data_for_indexing.get_nearest_examples_batch('embeddings', emb, k=30)
    for i, item in enumerate(retrieved_examples):
        hard = [x[0] for x in item['pos']] # [(str, str), ...]
        prompt = [x[0] for x in hard]
        hard = [x[1] for x in hard]
        pos_passages = [x[1] for x in pos[i]]
        hard_negatives = [[random.choice(prompt), x] for x in set(hard[20:]) if x not in pos_passages]
        neg[i].extend(hard_negatives)
    return {
        'neg': neg
        }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset

    # if data is already processed, skip
    if os.path.exists(f'/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/hard/{dataset}.jsonl'):
        exit()

    data_path = f"/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/{dataset}.jsonl"
    data = load_dataset('json', data_files=data_path, split='train')

    with torch.set_grad_enabled(False):
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        data = data.map(lambda x: {'text': x['query'][1] + '\n' + x['pos'][0][1]}, num_proc=20)
        data = data.map(lambda x: get_embedding(x['text']), batch_size=32, batched=True)
    
    # index the corpus with faiss
    data_for_indexing = deepcopy(data)
    data_for_indexing.add_faiss_index(column='embeddings')
    
    max_num_worker_suggest = 1
    try:
        max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
    num_workers = min(50, max_num_worker_suggest)

    data = data.map(lambda x: process_data(x), batch_size=256, batched=True)
    data = data.remove_columns(['embeddings', 'text'])
    data.to_json(f'/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/hard/{dataset}.jsonl')



