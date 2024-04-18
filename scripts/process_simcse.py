import pathlib
import random
from datasets import load_dataset


def process_data(example, emb_prompts, gen_prompts, incontext_data):
    r = random.random()
    if r < 0.05:
        prompt = ""
    elif r >= 0.05 and r < 0.6:
        prompt = random.choice(gen_prompts)
    else:
        prompt = random.choice(gen_prompts)
        idx = random.choice(range(len(incontext_data)))
        incontext = incontext_data[idx]
        incontext_query = incontext['query']
        incontext_pos = incontext['pos'][0]
        example_words = ['\n', 'For example,', 'E.g.,', 'For instance,', 'Fewshot example,', 'To give you a sense,', 'The query could be', 'Examples:', 'Fewshots:', 'Given']
        word = random.choice(example_words)
        prompt = f"{prompt} {word} input: {incontext_query},\n output: {incontext_pos}"
    query = [random.choice(emb_prompts), example['query']]
    pos = [[random.choice(emb_prompts), p] for p in example['pos']]
    neg = [[random.choice(emb_prompts), n] for n in example['neg']]
    return {
        'query': query,
        'pos': pos,
        'neg': neg,
        'gen_prompt': prompt,
        }

if __name__=='__main__':
    data_dir = '/home/hieum/uonlp/LLM_Emb/dataset/simcse'
    gen_prompt = load_dataset('json', data_files=f'{data_dir}/gen_prompts.jsonl', split='train')['gen_prompt']
    emb_prompt = load_dataset('json', data_files=f'{data_dir}/emb_prompts.jsonl', split='train')['emb_prompt']
    data = load_dataset('json', data_files=f'{data_dir}/simcse_hard.jsonl', split='train')

    data = data.train_test_split(test_size=2000)
    incontext_data = data['test']
    data = data['train']

    data = data.map(lambda x: process_data(x, emb_prompt, gen_prompt, incontext_data), num_proc=20)
    data.to_json(f'{data_dir}/simcse_hard_processed.jsonl')

