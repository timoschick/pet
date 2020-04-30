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
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import log
from utils import InputExample

logger = log.get_logger('root')


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1, skip_first=0):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        :param skip_first: If set to a value >0, the first `skip_first` examples for each label are skipped
        """
        self._labels = labels
        self._skip_first = skip_first
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

        for label in self._labels:
            if self._max_examples[label] > 0:
                self._max_examples[label] += skip_first

    def is_full(self):
        """
        Returns `true` iff no more examples can be added to this list
        """
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            if self._examples_per_label[label] > self._skip_first:
                self._examples.append(example)
                return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):

    @abstractmethod
    def get_train_examples(self, data_dir, examples_per_label=-1, skip_first=0) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir, examples_per_label=-1, skip_first=0) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        pass

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", max_examples=examples_per_label,
            skip_first=skip_first)

    def get_dev_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched", max_examples=examples_per_label, skip_first=skip_first)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = LimitedExampleList(self.get_labels(), max_examples, skip_first)

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.add(example)
            if examples.is_full():
                break

        return examples.to_list()


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched", max_examples=examples_per_label, skip_first=skip_first)


class AgnewsProcessor(DataProcessor):
    """Processor for the AG news data set."""

    def get_train_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train", examples_per_label,
                                     skip_first=skip_first)

    def get_dev_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev", examples_per_label,
                                     skip_first=skip_first)

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = LimitedExampleList(self.get_labels(), max_examples, skip_first)

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.add(example)
                if examples.is_full():
                    break

        return examples.to_list()


class YahooAnswersProcessor(DataProcessor):
    """Processor for the Yahoo Answers data set."""

    def get_train_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train", examples_per_label,
                                     skip_first=skip_first)

    def get_dev_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev", examples_per_label,
                                     skip_first=skip_first)

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = LimitedExampleList(self.get_labels(), max_examples, skip_first)

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                guid = "%s-%s" % (set_type, idx)
                text_a = question_title.replace('\\n', ' ').replace('\\', ' ') + ' ' + \
                         question_body.replace('\\n', ' ').replace('\\', ' ')
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.add(example)
                if examples.is_full():
                    break

        return examples.to_list()


class YelpPolarityProcessor(DataProcessor):
    """Processor for the YELP binary classification set."""

    def get_train_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train", examples_per_label,
                                     skip_first=skip_first)

    def get_dev_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev", examples_per_label,
                                     skip_first=skip_first)

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = LimitedExampleList(self.get_labels(), max_examples, skip_first)

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.add(example)
                if examples.is_full():
                    break

        return examples.to_list()


class YelpFullProcessor(YelpPolarityProcessor):
    """Processor for the YELP full classification set."""

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]


class XStanceProcessor(DataProcessor):
    """Processor for the X-Stance data set."""

    def __init__(self, language: str = None):
        if language is not None:
            assert language in ['de', 'fr']
        self.language = language

    def get_train_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train", examples_per_label,
                                     skip_first=skip_first)

    def get_dev_examples(self, data_dir, examples_per_label=-1, skip_first=0):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "dev", examples_per_label,
                                     skip_first=skip_first)

    def get_labels(self):
        """See base class."""
        return ["FAVOR", "AGAINST"]

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = LimitedExampleList(self.get_labels(), max_examples, skip_first)

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = example_json['label']
                id = example_json['id']
                text_a = example_json['question']
                text_b = example_json['comment']
                language = example_json['language']

                if self.language is not None and language != self.language:
                    continue

                example = InputExample(guid=id, text_a=text_a, text_b=text_b, label=label)
                examples.add(example)
                if examples.is_full():
                    break

        return examples.to_list()


PROCESSORS = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "agnews": AgnewsProcessor,
    "yahoo": YahooAnswersProcessor,
    "yelp-polarity": YelpPolarityProcessor,
    "yelp-full": YelpFullProcessor,
    "xstance-de": lambda: XStanceProcessor("de"),
    "xstance-fr": lambda: XStanceProcessor("fr"),
    "xstance": XStanceProcessor,
}


def load_examples(task, data_dir: str, examples_per_label: int, skip_first: int = 0, evaluate=False) \
        -> List[InputExample]:
    """Load examples for a given task."""
    processor = PROCESSORS[task]()

    logger.info("Creating features from dataset file at {}".format(data_dir))
    if evaluate:
        examples = processor.get_dev_examples(data_dir, examples_per_label=examples_per_label, skip_first=skip_first)
    else:
        examples = processor.get_train_examples(data_dir, examples_per_label=examples_per_label, skip_first=skip_first)
    return examples
