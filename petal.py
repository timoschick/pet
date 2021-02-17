import argparse
import os
import json
from collections import Counter
from typing import Dict, List

import numpy as np
import random
import torch
from transformers import PreTrainedTokenizer, RobertaTokenizer

from pet.tasks import PROCESSORS, load_examples, TRAIN_SET
from pet.utils import InputExample, eq_div
from pet.wrapper import TransformerModelWrapper, MODEL_CLASSES, WrapperConfig
import log

logger = log.get_logger('root')


def filter_words(tokens: List[str], word_counts=None, max_words: int = -1):
    """
    Given a list of tokens, return a reduced list that contains only tokens from the list that correspond
    to actual words and occur a given number of times.
    :param tokens: the list of tokens to filter
    :param word_counts: a dictionary mapping words to their number of occurrences
    :param max_words: if set to a value >0, only the `max_words` most frequent words according to `word_counts` are kept
    :return: the filtered list of tokens
    """
    tokens = (word for word in tokens if word[0] == 'Ġ' and len([char for char in word[1:] if char.isalpha()]) >= 2)
    if word_counts and max_words > 0:
        tokens = sorted(tokens, key=lambda word: word_counts[word[1:]], reverse=True)[:max_words]
    return tokens


def get_word_to_id_map(tokenizer: PreTrainedTokenizer, word_counts=None, max_words: int = -1):
    """
    Return a mapping from all tokens to their internal ids for a given tokenizer
    :param tokenizer: the tokenizer
    :param word_counts: a dictionary mapping words to their number of occurrences
    :param max_words: if set to a value >0, only the `max_words` most frequent words according to `word_counts` are kept
    :return:
    """
    if not isinstance(tokenizer, RobertaTokenizer):
        raise ValueError("this function currently only supports instances of 'RobertaTokenizer'")

    words = filter_words(tokenizer.encoder.keys(), word_counts, max_words)
    word2id = {word[1:]: tokenizer.convert_tokens_to_ids(word) for word in words}
    logger.info(f"There are {len(word2id)} words left after filtering non-word tokens")
    return word2id


class AutomaticVerbalizerSearch:

    def __init__(self, word2idx: Dict[str, int], labels: List[str], logits_list: List[np.ndarray],
                 expected: Dict[str, np.ndarray]):
        self.word2idx = word2idx
        self.labels = labels
        self.expected = expected

        logits_list = [np.exp(logits) for logits in logits_list]
        self.probs_list = [logits / np.expand_dims(np.sum(logits, axis=1), axis=1) for logits in logits_list]

    def _get_candidates(self, num_candidates: int) -> Dict[str, List[str]]:
        if num_candidates <= 0:
            return {label: self.word2idx.keys() for label in self.labels}

        scores = {label: Counter() for label in self.labels}

        for label in self.labels:
            for probs in self.probs_list:
                for word, idx in self.word2idx.items():
                    score = np.sum(np.log(probs[:, idx]) * self.expected[label])
                    scores[label][word] += score

        return {label: [w for w, _ in scores[label].most_common(num_candidates)] for label in self.labels}

    def _get_top_words(self, candidates: Dict[str, List[str]], normalize: bool = True, words_per_label: int = 10,
                       score_fct: str = 'llr') -> Dict[str, List[str]]:

        scores = {label: Counter() for label in self.labels}

        for label in self.labels:
            for probs in self.probs_list:
                for word in candidates[label]:
                    idx = self.word2idx[word]
                    if score_fct == 'llr':
                        scores[label][word] += self.log_likelihood_ratio(probs[:, idx], self.expected[label], normalize)
                    elif score_fct == 'ce':
                        scores[label][word] += self.cross_entropy(probs[:, idx], self.expected[label], normalize)
                    else:
                        raise ValueError(f"Score function '{score_fct}' not implemented")

        return {label: [w for w, _ in scores[label].most_common(words_per_label)] for label in self.labels}

    @staticmethod
    def log_likelihood_ratio(predictions: np.ndarray, expected: np.ndarray, normalize: bool) -> float:
        scale_factor = sum(1 - expected) / sum(expected) if normalize else 1
        pos_score = scale_factor * (np.sum(np.log(predictions) * expected) - np.sum(np.log(1 - predictions) * expected))
        neg_score = np.sum(np.log(1 - predictions) * (1 - expected)) - np.sum(np.log(predictions) * (1 - expected))
        return pos_score + neg_score

    @staticmethod
    def cross_entropy(predictions: np.ndarray, expected: np.ndarray, normalize: bool) -> float:
        scale_factor = sum(1 - expected) / sum(expected) if normalize else 1
        pos_score = scale_factor * np.sum(np.log(predictions) * expected)
        neg_score = np.sum(np.log(1 - predictions) * (1 - expected))
        return pos_score + neg_score

    def find_verbalizer(self, words_per_label: int = 10, num_candidates: int = 1000, normalize: bool = True,
                        score_fct: str = 'llr'):
        if score_fct == 'random':
            return {label: random.sample(self.word2idx.keys(), words_per_label) for label in self.labels}

        candidates = self._get_candidates(num_candidates=num_candidates)
        return self._get_top_words(candidates=candidates, normalize=normalize, words_per_label=words_per_label,
                                   score_fct=score_fct)


def main():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory. The verbalizers are written to a file 'verbalizer.json' in this directory.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="The model type")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(PROCESSORS.keys()))

    # verbalizer search hyperparameters
    parser.add_argument("--normalize", action='store_true',
                        help="Whether to normalize the loss as proposed in the paper. It is recommended to set this to 'true'.")
    parser.add_argument("--combine_patterns", action='store_true',
                        help="If set to true, a single joint verbalizer is searched for all patterns")
    parser.add_argument("--num_candidates", default=1000, type=int,
                        help="The number of candidate tokens to consider as verbalizers (see Section 4.1 of the paper)")
    parser.add_argument("--words_per_label", default=10, type=int,
                        help="The number of verbalizer tokens to assign to each label")
    parser.add_argument("--score_fct", default='llr', choices=['llr', 'ce', 'random'],
                        help="The function used to score verbalizers. Choices are: the log-likelihood ratio loss proposed in the paper "
                             "('llr'), cross-entropy loss ('ce') and 'random', which assigns random tokens to each label.")

    # other optional parameters
    parser.add_argument("--train_examples", default=50, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--words_file", default=None, type=str,
                        help="Path to a file containing (unlabeled) texts from the task's domain. This text is used to compute "
                             "verbalization candidates by selecting the most frequent words.")
    parser.add_argument("--max_words", default=10000, type=int,
                        help="Only the 10,000 tokens that occur most frequently in the task’s unlabeled data (see --words_file) are "
                             "considered as verbalization candidates")
    parser.add_argument("--additional_input_examples", type=str,
                        help="An optional path to an additional set of input examples (e.g., obtained using iPET)")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for initialization")

    args = parser.parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'config.txt'), 'w', encoding='utf8') as fh:
        json.dump(args.__dict__, fh, indent=2)

    # setup gpu/cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task not found: {}".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()
    args.cache_dir = ""
    args.do_lower_case = False
    args.verbalizer_file = None
    args.wrapper_type = 'mlm'

    # get training data
    train_examples_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
    train_data = load_examples(args.task_name, args.data_dir, set_type=TRAIN_SET, num_examples_per_label=train_examples_per_label)
    if args.additional_input_examples:
        additional_data = InputExample.load_examples(args.additional_input_examples)
        train_data += additional_data
        logger.info(f"Loaded {len(additional_data)} additional examples from {args.additional_input_examples}, total"
                    f"training set size is now {len(train_data)}")

    expected = {label: np.array([1 if x.label == label else 0 for x in train_data]) for label in args.label_list}

    if args.words_file:
        with open(args.words_file, 'r', encoding='utf8') as fh:
            word_counts = Counter(fh.read().split())
    else:
        word_counts = None

    tokenizer_class = MODEL_CLASSES[args.model_type]['tokenizer']
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    word2idx = get_word_to_id_map(tokenizer, word_counts=word_counts, max_words=args.max_words)

    logits = []

    for pattern_id in args.pattern_ids:
        logger.info(f"Processing examples with pattern id {pattern_id}...")
        args.pattern_id = pattern_id

        config = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path, wrapper_type='mlm',
                               task_name=args.task_name, max_seq_length=args.max_seq_length, label_list=args.label_list,
                               pattern_id=args.pattern_id)

        wrapper = TransformerModelWrapper(config)
        wrapper.model.to(device)
        # modify all patterns so that they return a single text segment instead of two segments
        get_parts = wrapper.preprocessor.pvp.get_parts
        wrapper.preprocessor.pvp.get_parts = lambda example: (get_parts(example)[0] + get_parts(example)[1], [])
        wrapper.preprocessor.pvp.convert_mlm_logits_to_cls_logits = lambda mask, x, _=None: x[mask >= 0]

        pattern_logits = wrapper.eval(train_data, device, per_gpu_eval_batch_size=args.per_gpu_eval_batch_size, n_gpu=args.n_gpu)['logits']
        pattern_logits = pattern_logits - np.expand_dims(np.max(pattern_logits, axis=1), axis=1)
        logits.append(pattern_logits)

    logger.info("Starting verbalizer search...")

    if args.combine_patterns:
        avs = AutomaticVerbalizerSearch(word2idx, args.label_list, logits, expected)
        verbalizer = avs.find_verbalizer(
            num_candidates=args.num_candidates,
            words_per_label=args.words_per_label,
            normalize=args.normalize,
            score_fct=args.score_fct
        )
        verbalizers = {pattern_id: verbalizer for pattern_id in args.pattern_ids}

    else:
        verbalizers = {}
        for idx, pattern_id in enumerate(args.pattern_ids):
            avs = AutomaticVerbalizerSearch(word2idx, args.label_list, [logits[idx]], expected)
            verbalizers[pattern_id] = avs.find_verbalizer(
                num_candidates=args.num_candidates,
                words_per_label=args.words_per_label,
                normalize=args.normalize,
                score_fct=args.score_fct
            )

    print(json.dumps(verbalizers, indent=2))
    logger.info("Verbalizer search complete, writing output...")

    with open(os.path.join(args.output_dir, 'verbalizers.json'), 'w', encoding='utf8') as fh:
        json.dump(verbalizers, fh, indent=2)

    logger.info("Done")


if __name__ == "__main__":
    main()
