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
        prompt = "Generate a text that is similar or relevant with the input text. Your output must always be the text only, do not explain yourself or output anything else. Be creative!\nInput text:\n"
        examples = []
        for text in inputs:
            chat = [{'role': 'user', 'content': f'{prompt}{text}'},]
            example = gen_tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
            examples.append({
                "input_ids": example,
                "input_length": len(example),
            })
        examples = gen_tokenizer.pad(examples, return_tensors='pt')
        input_ids = examples['input_ids']
        input_len = examples.pop('input_length')
        results = gen_model.generate(
            input_ids.to(device), 
            attention_mask=examples['attention_mask'].to(device),
            max_new_tokens=512,
            num_beams=3,
            do_sample=True)
        pos = []
        for i in range(input_ids.size(0)):
            txt = gen_tokenizer.decode(results[i][input_len[i]:], skip_special_tokens=True)
            pos.append([txt])

        neg_prompt = "Generate a text that seemingly relevant with the input text but actually not. Your output must always be the text only, do not explain yourself or output anything else. Be creative!\nInput text:\n"
        examples = []
        for text in inputs:
            chat = [{'role': 'user', 'content': f'{neg_prompt}{text}'},]
            example = gen_tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
            examples.append({
                "input_ids": example,
                "input_length": len(example),
            })
        examples = gen_tokenizer.pad(examples, return_tensors='pt')
        input_ids = examples['input_ids']
        input_len = examples.pop('input_length')
        results = gen_model.generate(
            input_ids.to(device), 
            attention_mask=examples['attention_mask'].to(device),
            max_new_tokens=512,
            num_beams=3,
            do_sample=True)
        neg = []
        for i in range(input_ids.size(0)):
            txt = gen_tokenizer.decode(results[i][input_len[i]:], skip_special_tokens=True)
            neg.append([txt])

        neg_prompt_2 = "Create a piece of text that appears to be relevant or related to the input text, but is actually not truly relevant or connected in a meaningful way. Your output must always be the text only, do not explain yourself or output anything else. Be creative!\nInput text:\n"
        examples = []
        for text in inputs:
            chat = [{'role': 'user', 'content': f'{neg_prompt_2}{text}'},]
            examples.append({
                "input_ids": example,
                "input_length": len(example),
            })
        examples = gen_tokenizer.pad(examples, return_tensors='pt')
        input_ids = examples['input_ids']
        input_len = examples.pop('input_length')
        results = gen_model.generate(
            input_ids.to(device), 
            attention_mask=examples['attention_mask'].to(device),
            max_new_tokens=512,
            num_beams=3,
            do_sample=True)
        for i in range(input_ids.size(0)):
            txt = gen_tokenizer.decode(results[i][input_len[i]:], skip_special_tokens=True)
            neg[i].append(txt)
        return {'pos': pos,
                'neg': neg}
    
    data = load_dataset("princeton-nlp/datasets-for-simcse", split=f"train[{start}:{start + 10000}]")

    # Filter out the data that has less than 10 words and more than 2048 words
    data = data.filter(lambda x: len(x['text'].split()) >= 10 and len(x['text'].split()) <= 1024)
    # Apply the function to generate positive and negative examples
    data = data.map(lambda x: generate_positive_example(x), batched=True, batch_size=16)
    # Save the dataset to jsonl file
    data.to_json(f"simcse/simcse_hard_{start}.jsonl")

    


        

