from audioop import add
from json import load
import os
import random
from typing import Dict, List
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)


if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    args = parser.parse_args()
    start = args.part * 10000
    data_file = f"dataset/simcse/simcse_hard_{args.part}.jsonl"
    if os.path.exists(data_file):
        print(f"File {data_file} already exists")
        exit(0)

    gen_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = torch.device("cuda")

    compute_dtype = getattr(torch, 'bfloat16', torch.float16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,)
    # Load base model
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        quantization_config=bnb_config,
        device_map='auto',)
    gen_model.eval()
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"

    def generate_positive_example(example: Dict):
        inputs = example['text']
        prompt = "Generate a positive example that is similar or relevant with the input text. Your output must always be the generated positive example only, do not explain yourself or output anything else. Be creative!\n"
        examples = []
        for text in inputs:
            chat = [{'role': 'user', 'content': f'{prompt}{text}'},]
            example = gen_tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, add_special_tokens=False)
            examples.append(example)
        encoded_examples = gen_tokenizer(examples, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_examples['input_ids']
        results = gen_model.generate(
            input_ids.to(device), 
            attention_mask=encoded_examples['attention_mask'].to(device),
            max_new_tokens=512,)
        results = results[:, input_ids.size(1):]
        pos = []
        txt = gen_tokenizer.batch_decode(results, skip_special_tokens=True)
        for i, t in enumerate(txt):
            pos.append([t])
        
        neg_prompt = "Create a negative example that appears to be relevant or related to the input text, but is actually not truly relevant or connected in a meaningful way. Your output must always be the generated negative example only, do not explain yourself or output anything else. Be creative!\n"
        examples = []
        for text in inputs:
            chat = [{'role': 'user', 'content': f'{neg_prompt}{text}'},]
            example = gen_tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            examples.append(example)
        encoded_examples = gen_tokenizer(examples, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_examples['input_ids']
        results = gen_model.generate(
            input_ids.to(device), 
            attention_mask=encoded_examples['attention_mask'].to(device),
            max_new_tokens=512,)
        results = results[:, input_ids.size(1):]
        neg = []
        txt = gen_tokenizer.batch_decode(results, skip_special_tokens=True)
        for i, t in enumerate(txt):
            neg.append([t])
        
        neg_prompt_2 = "Create a negative example that appears to be relevant or related to the input text, but is actually not truly relevant or connected in a meaningful way. Your output must always be the negative example only, do not explain yourself or output anything else. Be creative!\n"
        examples = []
        for text in inputs:
            chat = [{'role': 'user', 'content': f'{neg_prompt_2}{text}'},]
            example = gen_tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            examples.append(example)
        encoded_examples = gen_tokenizer(examples, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_examples['input_ids']
        results = gen_model.generate(
            input_ids.to(device), 
            attention_mask=encoded_examples['attention_mask'].to(device),
            max_new_tokens=512,)
        results = results[:, input_ids.size(1):]
        txt = gen_tokenizer.batch_decode(results, skip_special_tokens=True)
        for i, t in enumerate(txt):
            neg[i].append(t)

        return {'pos': pos,
                'neg': neg}
    
    data = load_dataset("princeton-nlp/datasets-for-simcse", split=f"train[{start}:{start + 10000}]")

    # Filter out the data that has less than 10 words and more than 2048 words
    data = data.filter(lambda x: len(x['text'].split()) >= 10 and len(x['text'].split()) <= 1024)
    # Apply the function to generate positive and negative examples
    data = data.map(lambda x: generate_positive_example(x), batched=True, batch_size=16)
    # Save the dataset to jsonl file
    data.to_json(f"dataset/simcse/simcse_hard_{args.part}.jsonl")

    


        

