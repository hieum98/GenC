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
    emb_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + embed_bos
    emb_example_format = emb_prompt_format + "{example}" + embed_eos
    gen_prompt_format = base_bos + user_bos + "{prompt}" + user_eos + assistant_bos
    gen_example_format = gen_prompt_format + "{response}" + assistant_eos

    emb_prompt = emb_prompt_format.format(prompt=example['query'][0])
    emb_example = emb_example_format.format(prompt=example['query'][0], example=example['query'][1])
    gen_prompt = gen_prompt_format.format(prompt=f"{example['gen_prompt']}\n{example['query'][1]}")

    if not filter_too_long_example(emb_prompt, max_seq_length/2) \
        or not example['query'][1] \
        or not filter_too_long_example(emb_example, max_seq_length) \
        or not filter_too_long_example(gen_prompt, max_seq_length/1.5):
        return False
    
    for ex in example['pos'] + example['neg']:
        _emb_prompt = emb_prompt_format.format(prompt=ex[0])
        _emb_example = emb_example_format.format(prompt=ex[0], example=ex[1])
        _gen_example = gen_example_format.format(prompt=f"{example['gen_prompt']}\n{example['query'][1]}", response=ex[1])

        if not filter_too_long_example(_emb_prompt, max_seq_length/2) \
            or not ex[1] \
            or not filter_too_long_example(_emb_example, max_seq_length) \
            or not filter_too_long_example(_gen_example, max_seq_length):
            return False
    
    # len_q = len(tokenizer(emb_example)['input_ids'])
    # len_prompt_q = len(tokenizer(emb_prompt)['input_ids'])

    # gen_len_q = len(tokenizer(q_gen_example)['input_ids'])
    # gen_len_prompt_q = len(tokenizer(gen_prompt)['input_ids'])

    # if len_prompt_q > max_seq_length / 2:
    #     return False
    # if len_q > max_seq_length:
    #     return False
    # if len_q < len_prompt_q:
    #     return False
    
    # if gen_len_prompt_q > max_seq_length / 1.5:
    #     return False
    # if gen_len_q > max_seq_length:
    #     return False
    # if gen_len_q < gen_len_prompt_q:
    #     return False
    
    # for ex in example['pos'] + example['neg']:
    #     emb_prompt = emb_prompt_format.format(prompt=ex[0])
    #     emb_example = emb_example_format.format(prompt=ex[0], example=ex[1])
    #     gen_example = gen_example_format.format(prompt=f"{example['gen_prompt']}\n{example['query'][1]}", response=ex[1])

    #     len_prompt = len(tokenizer(emb_prompt)['input_ids'])
    #     len_example = len(tokenizer(emb_example)['input_ids'])
    #     gen_len_example = len(tokenizer(gen_example)['input_ids'])

    #     if len_prompt > max_seq_length / 2:
    #         return False
    #     if len_example > max_seq_length:
    #         return False
    #     if len_example < len_prompt:
    #         return False
        
    #     if gen_len_prompt_q > max_seq_length / 1.5:
    #         return False
    #     if gen_len_example > max_seq_length:
    #         return False
    #     if gen_len_example < gen_len_prompt_q:
    #         return False
    return True
