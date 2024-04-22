import multiprocessing
from genc.special_tokens import base_bos, user_bos, user_eos, embed_bos, embed_eos, assistant_bos, assistant_eos


def filter_too_long_example(example, max_len):
    # Filter out super long examples to avoid tokenize taking forever
    if len(example.split()) > max_len-10:
        return False
    elif len(example) > max_len*8:
        return False
    return True


def filter_too_long_instructions(example, tokenizer, max_seq_length):
    # Filter out super long examples to avoid tokenize taking forever
    if len(example['query'][1].split()) < 5:
        return False
    
    emb_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + embed_bos
    emb_example_format = emb_prompt_format + "{example}" + embed_eos
    gen_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + assistant_bos
    gen_example_format = gen_prompt_format + "{response}" + assistant_eos

    emb_example = emb_example_format.format(prompt=example['query'][0], example=example['query'][1])

    if not example['query'][1] or not filter_too_long_example(emb_example, max_seq_length):
        return False
    
    for ex in example['pos'] + example['neg']:
        if len(ex[1].split()) < 5:
            return False
        
        _emb_example = emb_example_format.format(prompt=ex[0], example=ex[1])
        _gen_example = gen_example_format.format(prompt=f"{example['gen_prompt']}\n{example['query'][1]}", response=ex[1])

        if not ex[1] \
            or not filter_too_long_example(_emb_example, max_seq_length) \
            or not filter_too_long_example(_gen_example, max_seq_length):
            return False
    
    len_q = len(tokenizer(emb_example)['input_ids'])
    if len_q > max_seq_length-10:
        return False
    
    for ex in example['pos'] + example['neg']:
        emb_prompt = emb_prompt_format.format(prompt=ex[0])
        emb_example = emb_example_format.format(prompt=ex[0], example=ex[1])
        gen_example = gen_example_format.format(prompt=f"{example['gen_prompt']}\n{example['query'][1]}", response=ex[1])

        len_example = len(tokenizer(emb_example)['input_ids'])
        gen_len_example = len(tokenizer(gen_example)['input_ids'])

        if len_example > max_seq_length:
            return False
        if gen_len_example > max_seq_length:
            return False
    return True
