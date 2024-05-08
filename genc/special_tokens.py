
SPECILA_TOKENS ={
    'llama': {
        'bos': '<|begin_of_text|>',
        'user_bos': '<|start_header_id|>user<|end_header_id|>\n\n',
        'eot': '<|eot_id|>\n',
        'assistant_bos': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'eos': '<|end_of_text|>'
    },
    'mistral':{
        'bos': '<s>',
        'user_bos': '[INST]',
        'eot': '',
        'assistant_bos': '[/INST]',
        'eos': '</s>'
    },
    'phi':{
        'bos': '',
        'user_bos': '<|im_start|>user\n\n',
        'eot': '\n',
        'assistant_bos': '<|im_start|>assistant\n\n',
        'eos': '<|im_end|>'
    }
}