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
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""

import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
import torch.nn.functional as F
from transformers import InputExample, PreTrainedTokenizer, GPT2Tokenizer

import log

logger = log.get_logger('root')


def _prepare(word: str, tokenizer: PreTrainedTokenizer) -> str:
    if isinstance(tokenizer, GPT2Tokenizer):
        tokenized_word = tokenizer.tokenize(word, add_prefix_space=True)
        if len(tokenized_word) != 1:
            raise ValueError('"{}" is not a single-token word (tokenized: {})'.format(word, tokenized_word))
        return tokenized_word[0]
    return word


class PVP(ABC):

    def __init__(self, wrapper, pattern_id: int = 0, verbalizer_file: str = None):
        self.wrapper = wrapper
        self.pattern_id = pattern_id

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer = _prepare(verbalizer, self.wrapper.tokenizer)
                verbalizer_id = self.wrapper.tokenizer.convert_tokens_to_ids(verbalizer)
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self):
        return self.wrapper.tokenizer.mask_token

    @property
    def max_num_verbalizers(self):
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        return s, True

    def encode(self, example: InputExample) -> Tuple[List[int], List[int]]:
        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_added_tokens(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @abstractmethod
    def get_parts(self, example: InputExample) -> Tuple[
        List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]:
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        mask_token_id = self.wrapper.tokenizer.mask_token_id
        label_idx = input_ids.index(mask_token_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(l) for l in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class AgnewsPVP(PVP):
    VERBALIZER = {
        "1": ["World"],
        "2": ["Sports"],
        "3": ["Business"],
        "4": ["Tech"]
    }

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'News:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return AgnewsPVP.VERBALIZER[label]


class YahooPVP(PVP):
    VERBALIZER = {
        "1": ["Society"],
        "2": ["Science"],
        "3": ["Health"],
        "4": ["Education"],
        "5": ["Computer"],
        "6": ["Sports"],
        "7": ["Business"],
        "8": ["Entertainment"],
        "9": ["Relationship"],
        "10": ["Politics"],
    }

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'Question:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YahooPVP.VERBALIZER[label]


class MnliPVP(PVP):
    VERBALIZER_A = {
        "contradiction": ["Wrong"],
        "entailment": ["Right"],
        "neutral": ["Maybe"]
    }
    VERBALIZER_B = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> Tuple[
        List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]:

        text_a = self.shortenable(example.text_a.rstrip(string.punctuation))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 1:
            return MnliPVP.VERBALIZER_A[label]
        return MnliPVP.VERBALIZER_B[label]


class YelpPolarityPVP(PVP):
    VERBALIZER = {
        "1": ["bad"],
        "2": ["good"]
    }

    def get_parts(self, example: InputExample) -> Tuple[
        List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]:

        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It was', self.mask, '.', text], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], [text]
        elif self.pattern_id == 3:
            return [text], ['In summary, the restaurant is', self.mask, '.']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YelpPolarityPVP.VERBALIZER[label]


class YelpFullPVP(YelpPolarityPVP):
    VERBALIZER = {
        "1": ["terrible"],
        "2": ["bad"],
        "3": ["okay"],
        "4": ["good"],
        "5": ["great"]
    }

    def verbalize(self, label) -> List[str]:
        return YelpFullPVP.VERBALIZER[label]


class XStancePVP(PVP):
    VERBALIZERS = {
        'en': {"FAVOR": ["Yes"], "AGAINST": ["No"]},
        'de': {"FAVOR": ["Ja"], "AGAINST": ["Nein"]},
        'fr': {"FAVOR": ["Oui"], "AGAINST": ["Non"]}
    }

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2 or self.pattern_id == 4:
            return ['"', text_a, '"'], [self.mask, '. "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3 or self.pattern_id == 5:
            return [text_a], [self.mask, '.', text_b]

    def verbalize(self, label) -> List[str]:
        lang = 'de' if self.pattern_id < 2 else 'en' if self.pattern_id < 4 else 'fr'
        return XStancePVP.VERBALIZERS[lang][label]
