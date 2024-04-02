import multiprocessing


def filter_too_long_example(example, max_len):
    # Filter out super long examples to avoid tokenize taking forever
    if len(example) > max_len * 10:
        return False
    return True