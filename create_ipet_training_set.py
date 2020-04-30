# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to generate a training set for the next generation in iPET
using the previous generation of models.
"""
import argparse
import ast
import os
import random
from copy import deepcopy
from typing import List
import numpy as np

import log
import utils
from utils import InputExample, LogitsList
from run_training import load_examples, eq_div
from tasks import PROCESSORS

logger = log.get_logger('root')


def generate_train_set(logits_lists: List[LogitsList], labels: List[str], original_data: List[InputExample],
                       num_examples: int, logits_percentage: float, reduction: str = 'mean',
                       n_most_likely: int = -1) -> List[InputExample]:
    """
    Generate a training set for the next generation of iPET models
    :param logits_lists: predictions from the previous generation of models
    :param labels: all task labels
    :param original_data: the original training data corresponding to the logits_lists
    :param num_examples: the number of examples to create
    :param logits_percentage: the percentage of models/logits to choose
    :param reduction: the reduction strategy ('wmean' or 'mean')
    :param n_most_likely: if >0, for each label the n_most_likely examples with the highest logits are chosen
    :return: a list of input examples that serves as training set for the next generation
    """

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    num_logits_lists = round(len(logits_lists) * logits_percentage)
    logits_lists = random.sample(logits_lists, k=num_logits_lists)
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0)
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights)
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)
    logits = utils.softmax(logits, axis=1).tolist()

    for lgs, example in zip(logits, original_data):
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()]
    examples_per_label = eq_div(num_examples, len(labels))

    test_set = []

    for idx, label in enumerate(labels):

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logger.info("There are {} examples for label {}".format(len(examples), label))
            while len(examples) < examples_per_label[idx]:
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [(ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)]
            examples.sort(reverse=True)
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]]
            examples = [deepcopy(ex) for ex in examples]
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples,
            num_examples=examples_per_label[idx])
        test_set.extend(label_examples)

    return test_set


def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int) -> List[InputExample]:
    label_probabilities = [max(example.logits) for example in examples]
    sum_label_probabilities = sum(label_probabilities)
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    return np.random.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logits_dir", type=str, required=True,
                        help="The dir in which the results of all PVPs are stored in separate subdirs. "
                             "Each subdir is expected to have a file 'results.txt' and a file 'logits.txt' in "
                             "it as created by 'run.py'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The dir where the generate training sets are to be saved.")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Whether to overwrite the output dir's content if it already exists")
    parser.add_argument("--reduction", required=True, choices=['mean', 'wmean'],
                        help="The reduction strategy for merging logits. Must be one of 'mean' or 'wmean', "
                             "where the latter is short for 'weighted mean' and the weights for each PVP are "
                             "proportional to its score on the training set before fine-tuning.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")
    parser.add_argument("--lm_train_examples_per_label", required=True, type=int,
                        help="The number of unlabeled examples per label that were annotated using the previous "
                             "generation of models")
    parser.add_argument("--num_examples", required=True, type=int,
                        help="The total number of examples to create")
    parser.add_argument("--logits_percentage", required=True, type=float,
                        help="The percentage of logits (i.e., models) to choose for generating the next training sets")
    parser.add_argument("--seed", default=42, type=int, help="RNG seed")
    parser.add_argument("--n_most_likely", type=int, default=-1)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    processor = PROCESSORS[args.task_name]()
    labels = processor.get_labels()

    subdirs = next(os.walk(args.logits_dir))[1]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    all_train_data = load_examples(args.task_name, args.data_dir, args.lm_train_examples_per_label, evaluate=False)
    for example in all_train_data:
        example.label = None
        example.logits = None

    logits_lists = {}

    for subdir in subdirs:
        results_file = os.path.join(args.logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(args.logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        with open(results_file, 'r') as fh:
            results = ast.literal_eval(fh.read())
            result_train = results['train_set_before_training']

        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
            results_file, result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        logits_lists[subdir] = loglist

    for subdir in subdirs:
        other_logits_lists = [ll for sd, ll in logits_lists.items() if sd != subdir]
        subdir_train_set = generate_train_set(other_logits_lists,
                                              labels=labels,
                                              original_data=all_train_data,
                                              num_examples=args.num_examples,
                                              logits_percentage=args.logits_percentage,
                                              reduction=args.reduction,
                                              n_most_likely=args.n_most_likely)

        InputExample.save_examples(subdir_train_set,
                                   os.path.join(args.output_dir, subdir + '-train.txt'))


if __name__ == "__main__":
    main()
