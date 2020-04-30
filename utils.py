import copy
import json
import pickle
import random
from typing import Dict, List

import numpy as np
import torch


class LogitsList:
    def __init__(self, score: float, logits: List[List[float]]):
        self.score = score
        self.logits = logits

    def __repr__(self):
        return 'LogitsList(score={}, logits[:2]={})'.format(self.score, self.logits[:2])

    def save(self, path: str) -> None:
        with open(path, 'w') as fh:
            fh.write(str(self.score) + '\n')
            for example_logits in self.logits:
                fh.write(' '.join(str(logit) for logit in example_logits) + '\n')

    @staticmethod
    def load(path: str, with_score: bool = True) -> 'LogitsList':
        score = -1
        logits = []
        with open(path, 'r') as fh:
            for line_idx, line in enumerate(fh.readlines()):
                line = line.rstrip('\n')
                if line_idx == 0 and with_score:
                    score = float(line)
                else:
                    logits.append([float(x) for x in line.split()])
        return LogitsList(score=score, logits=logits)


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

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


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


def softmax(x, temperature=1.0, axis=None):
    """Custom softmax implementation"""
    y = np.atleast_2d(x)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(temperature)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum

    if len(x.shape) == 1:
        p = p.flatten()
    return p
