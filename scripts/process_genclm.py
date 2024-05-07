from genericpath import exists
import os
import pathlib
import random
from datasets import load_dataset

def process_data(example, gen_prompts, incontext_data):
    r = random.random()
    if r < 0.05:
        prompt = ""
    elif r >= 0.05 and r < 0.9:
        prompt = random.choice(gen_prompts)
    else:
        prompt = random.choice(gen_prompts)
        idx = random.choice(range(len(incontext_data)))
        incontext = incontext_data[idx]
        try:
            incontext_query = incontext['query'][1]
            incontext_pos = incontext['pos'][0][1]
        except:
            incontext_query = incontext['query']
            incontext_pos = incontext['positive']
        example_words = ['\n', 'For example', 'E.g', 'For instance', 'Fewshot example', 'To give you a sense', 'The query could be', 'Examples:', 'Fewshots:', 'Given']
        word = random.choice(example_words)
        prompt = f"{prompt} {word} input: {incontext_query},\n Output: {incontext_pos}"
    
    try:
        query = [random.choice(q_prompt), example['query'][1]]
        pos = [[random.choice(p_prompt), item[1]] for item in example['pos']]
        neg = [[random.choice(p_prompt), item[1]] for item in example['neg']]
    except:
        query = [random.choice(q_prompt), example['query']]
        pos = [[random.choice(p_prompt), example['positive']]]
        neg = [[random.choice(p_prompt), example['negative']]]

    return {
        'gen_prompt': prompt,
        'query': query,
        'pos': pos,
        'neg': neg
        }

if __name__=='__main__':

    path = "/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2"
    path = pathlib.Path(path)
    files = path.glob('*.jsonl')

    for f in files:
        print(f)
        f = str(f)
        file_name = f.split('/')[-1].split('.')[0]
        if os.path.exists(f'/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/processed/{file_name}.jsonl'):
            continue
        tmp_file_name = file_name
        if 'msmarco' in file_name:
            file_name = 'msmarco'
        gen_prompt = load_dataset('json', data_files=f'/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/prompt/gen/{file_name}.jsonl', split='train')['query_prompt']
        q_prompt = load_dataset('json', data_files=f'/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/prompt/emb/query/{file_name}.jsonl', split='train')['query_prompt']
        p_prompt = load_dataset('json', data_files=f'/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/prompt/emb/candidates/{file_name}.jsonl', split='train')['passage_prompt']
        file_name = tmp_file_name
        data = load_dataset('json', data_files=f, split='train')

        data = data.train_test_split(test_size=10)
        incontext_data = data['test']
        data = data['train']

        data = data.map(lambda x: process_data(x, gen_prompt, incontext_data), num_proc=50)

        # remove the columns except 'gen_prompt', 'query', 'pos', 'neg'
        col_names = data.column_names
        col_names_to_remove = [col for col in col_names if col not in ['gen_prompt', 'query', 'pos', 'neg']]
        data = data.remove_columns(col_names_to_remove)
        data.to_json(f'/home/hieum/uonlp/LLM_Emb/dataset/GenCLM_v2/processed/{file_name}.jsonl')


