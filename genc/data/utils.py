import multiprocessing


def filter_too_long_example(tokenizer, example, max_len):
    # Filter out super long examples to avoid tokenize taking forever
    if len(example) > max_len * 10:
        return False
    if len(tokenizer.tokenize(example)) >= max_len - 5:
        return False
    return True