import pathlib
import random
from datasets import load_dataset

def process_data(example, gen_prompts, incontext_data):
    r = random.random()
    if r < 0.05:
        prompt = ""
    elif r >= 0.05 and r < 0.6:
        prompt = random.choice(gen_prompts)
    else:
        prompt = random.choice(gen_prompts)
        idx = random.choice(range(len(incontext_data)))
        incontext = incontext_data[idx]
        incontext_query = incontext['query'][1]
        incontext_pos = incontext['pos'][0][1]
        example_words = ['\n', 'For example', 'E.g', 'For instance', 'Fewshot example', 'To give you a sense', 'The query could be', 'Examples:', 'Fewshots:', 'Given']
        word = random.choice(example_words)
        prompt = f"{prompt} {word} input: {incontext_query},\n output: {incontext_pos}"
    return {
        'gen_prompt': prompt,
        }

if __name__=='__main__':

    path = "/disk/hieu/IR/data/MEDI2BGE"
    path = pathlib.Path(path)
    files = path.glob('*.jsonl')

    for f in files:
        print(f)
        f = str(f)
        file_name = f.split('/')[-1].split('.')[0]
        gen_prompt = load_dataset('json', data_files=f'/disk/hieu/IR/data/prompt/query/{file_name}.jsonl', split='train')['query_prompt']
        data = load_dataset('json', data_files=f, split='train')

        data = data.train_test_split(test_size=10)
        incontext_data = data['test']
        data = data['train']

        data = data.map(lambda x: process_data(x, gen_prompt, incontext_data), num_proc=20)
        data.to_json(f'/mnt/hieu/MEDI2BGE/{file_name}.jsonl')


