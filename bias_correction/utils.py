import random
import numpy as np


def sample_n(arr, n_per_sample):
    result = []
    available = set(arr)

    for n in n_per_sample:
        sample = random.sample(available, n)        
        result.append(sample)
        available -= set(sample)

    return result


def merge_dictionaries(dicts, prefixes=[], separator='_'):
    merged = {}

    for i in range(len(dicts)):
        this_dict = dicts[i]
        other_dicts = dicts[:i] + dicts[i+1:]
        for key in this_dict:
            if any([key in other_dict for other_dict in other_dicts]):
                merged_key = separator.join((prefix[i], key))
            else:
                merged_key = key
            merged[merged_key] = this_dict[key]

    return merged
    