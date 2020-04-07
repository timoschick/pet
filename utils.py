import copy
import json
import random
from typing import Dict

import numpy as np
import torch


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, label, mlm_labels=None, logits=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.mlm_labels = mlm_labels
        self.logits = logits

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def set_seed(args):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def eq_div(N, i):
    """ Equally divide N examples among i buckets. For example, `eq_div(12,3) = [4,4,4]`. """
    return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)


def save_logits(path: str, logits: np.ndarray):
    with open(path, 'w') as fh:
        for example_logits in logits:
            fh.write(' '.join(str(logit) for logit in example_logits) + '\n')
    pass


def save_result(path: str, results: Dict[str, float], key: str = None):
    with open(path, 'w') as fh:
        if key:
            fh.write(str(results[key]) + '\n')
        else:
            fh.write(str(results) + '\n')