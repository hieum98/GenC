from json import load
import os
import random
from typing import Dict, List
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
    q_id = [str(x) for x in example['query-id']]
    p_id = [str(x) for x in example['corpus-id']]

    # Find the datapoint that have the same id
    q_example = query_df.loc[q_id].to_dict('list')
    p_example = corpus_df.loc[p_id].to_dict('list')
    q_text = q_example['text']
    p_text = p_example['text']

    # Retrieve the hard negatives
    q_emb = q_example['embeddings']
    q_emb = np.array(q_emb)
    _score, retrieved_examples = indexed_corpus_ds.get_nearest_examples_batch('embeddings', q_emb, k=256)
    
    hard_negatives = []
    for i, item in enumerate(retrieved_examples):
        ids = item['_id']
        texts = item['text']
        hard_negatives.append([x[1] for x in list(zip(ids, texts))[30:] if x[0] != p_id[i]])
    
    query = [[random.sample(query_prompts, k=1)[0], text] for text in q_text]
    pos = [[[random.sample(passage_prompts, k=1)[0], text]] for text in p_text]
    neg = []
    for item in hard_negatives:
        neg.append(
            list(zip(random.choices(passage_prompts, k=len(item)), item))
            )
    return {
        'query': query,
        'pos': pos,
        'neg': neg,
        'gen_prompt': list(random.choices(gen_prompts, k=len(q_text))),
        }


def process_data_eval(example: Dict):
    q_id = [str(x) for x in example['query-id']]
    p_id = [str(x) for x in example['corpus-id']]

    # Find the datapoint that have the same id
    q_example = query_df.loc[q_id].to_dict('list')
    p_example = corpus_df.loc[p_id].to_dict('list')
    q_text = q_example['text']
    p_text = p_example['text']

    query = [[random.sample(query_prompts, k=1)[0], text] for text in q_text]
    pos = [[[random.sample(passage_prompts, k=1)[0], text]] for text in p_text]
    return {
        'query': query,
        'pos': pos,
        }


if __name__ == '__main__':
    # Load the dataset from jsonl file
    corpus_ds = load_dataset('json', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/corpus.jsonl', split='train')
    query_ds = load_dataset('json', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/queries.jsonl', split='train')

    # Load label from tsv file
    train_label = load_dataset('csv', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/qrels/train.tsv', delimiter='\t', split='train')
    dev_label = load_dataset('csv', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/qrels/dev.tsv', delimiter='\t', split='train')
    test_label = load_dataset('csv', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/qrels/test.tsv', delimiter='\t', split='train')

    # load prompts list from file
    query_prompts = load_dataset('json', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/query_prompt.jsonl', split='train')['query_prompt']
    passage_prompts = load_dataset('json', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/passage_prompt.jsonl', split='train')['passage_prompt']
    gen_prompts = load_dataset('json', data_files='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/gen_candidate_prompts.jsonl', split='train')['gen_prompt']

    # filter out the data that has score < 0.9
    train_label = train_label.filter(lambda x: x['score'] >= 0.9)

    with torch.set_grad_enabled(False):
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        device = torch.device("cuda")
        model.to(device)

        # Embedding the query and corpus
        query_ds = query_ds.map(lambda x: get_embedding(x['text']), batched=True, batch_size=64, cache_file_name='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/query_embeddings.cache')
        indexed_corpus_ds = corpus_ds.map(lambda x: get_embedding(x['text']), batched=True, batch_size=64, cache_file_name='/home/hieum/uonlp/LLM_Emb/dataset/msmarco/corpus_embeddings.cache')

    # Index the corpus_ds with faiss
    indexed_corpus_ds.add_faiss_index(column='embeddings')
    query_df = query_ds.to_pandas().set_index('_id')
    corpus_df = corpus_ds.to_pandas().set_index('_id')

    # Process the train data
    train_data = train_label.map(lambda x: process_data(x), batched=True, batch_size=32, num_proc=20)
    train_data.to_json('/home/hieum/uonlp/LLM_Emb/dataset/msmarco/msmarco_hard.jsonl')

    # Process the dev and test data
    dev_data = dev_label.map(lambda x: process_data_eval(x), batched=True, batch_size=32, num_proc=20)
    dev_data.to_json('/home/hieum/uonlp/LLM_Emb/dataset/msmarco/msmarco_dev.jsonl')

    test_data = test_label.map(lambda x: process_data_eval(x), batched=True, batch_size=32, num_proc=20)
    test_data.to_json('/home/hieum/uonlp/LLM_Emb/dataset/msmarco/msmarco_test.jsonl')




