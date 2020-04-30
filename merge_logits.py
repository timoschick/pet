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
This script can be used to merge logits obtained from different PVPs in order
to train a final model.
"""
import argparse
import ast
import os
from typing import List
import numpy as np

import log
from utils import LogitsList

logger = log.get_logger('root')


def merge_logits_lists(logits_lists: List[LogitsList], reduction: str = 'mean') -> LogitsList:
    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    return LogitsList(score=-1, logits=logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="The dir in which the results of all PVPs are stored in separate subdirs. "
                             "Each subdir is expected to have a file 'results.txt' and a file 'logits.txt' in "
                             "it as created by 'run.py'")
    parser.add_argument("--output_file", type=str, required=True,
                        help="The file where the merged logits are to be saved.")
    parser.add_argument("--overwrite_output_file", action='store_true',
                        help="Whether to overwrite the output file if it already exists")
    parser.add_argument("--reduction", required=True, choices=['mean', 'wmean'],
                        help="The reduction strategy for merging logits. Must be one of 'mean' or 'wmean', "
                             "where the latter is short for 'weighted mean' and the weights for each PVP are "
                             "proportional to its score on the training set before fine-tuning.")
    args = parser.parse_args()

    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        logger.error("Output file already exists")
        exit()

    subdirs = next(os.walk(args.logits_dir))[1]
    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    all_logits_lists = []

    for subdir in subdirs:
        results_file = os.path.join(args.logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(args.logits_dir, subdir, 'logits.txt')
        logits = []

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
        all_logits_lists.append(loglist)

    merged_loglist = merge_logits_lists(all_logits_lists, reduction=args.reduction)
    merged_loglist.save(args.output_file)


if __name__ == "__main__":
    main()
