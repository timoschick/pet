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

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from pet.utils import InputFeatures, InputExample, PLMInputFeatures
from pet.pvp import PVP, PVPS


class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:`InputExample` into a :class:`InputFeatures` object so that it can be
    processed by the model being used.
    """

    def __init__(self, wrapper, task_name, pattern_id: int = 0, verbalizer_file: str = None):
        """
        Create a new preprocessor.

        :param wrapper: the wrapper for the language model to use
        :param task_name: the name of the task
        :param pattern_id: the id of the PVP to be used
        :param verbalizer_file: path to a file containing a verbalizer that overrides the default verbalizer
        """
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id, verbalizer_file)  # type: PVP
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class MLMPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""

    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:

        if priming:
            input_ids, token_type_ids = self.pvp.encode(example, priming=True)
            priming_data = example.meta['priming_data']  # type: List[InputExample]

            priming_input_ids = []
            for priming_example in priming_data:
                pe_input_ids, _ = self.pvp.encode(priming_example, priming=True, labeled=True)
                priming_input_ids += pe_input_ids

            input_ids = priming_input_ids + input_ids
            token_type_ids = self.wrapper.tokenizer.create_token_type_ids_from_sequences(input_ids)
            input_ids = self.wrapper.tokenizer.build_inputs_with_special_tokens(input_ids)
        else:
            input_ids, token_type_ids = self.pvp.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100
        logits = example.logits if example.logits else [-1]

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
            if self.wrapper.config.model_type == 'gpt2':
                # shift labels to the left by one
                mlm_labels.append(mlm_labels.pop(0))
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits, idx=example.idx)


class PLMPreprocessor(MLMPreprocessor):
    """Preprocessor for models pretrained using a permuted language modeling objective (e.g., XLNet)."""

    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> PLMInputFeatures:
        input_features = super().get_input_features(example, labelled, priming, **kwargs)
        input_ids = input_features.input_ids

        num_masks = 1  # currently, PLMPreprocessor supports only replacements that require exactly one mask

        perm_mask = np.zeros((len(input_ids), len(input_ids)), dtype=np.float)
        label_idx = input_ids.index(self.pvp.mask_id)
        perm_mask[:, label_idx] = 1  # the masked token is not seen by any other token

        target_mapping = np.zeros((num_masks, len(input_ids)), dtype=np.float)
        target_mapping[0, label_idx] = 1.0

        return PLMInputFeatures(perm_mask=perm_mask, target_mapping=target_mapping, **input_features.__dict__)


class SequenceClassifierPreprocessor(Preprocessor):
    """Preprocessor for a regular sequence classification model."""

    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        inputs = self.wrapper.task_helper.get_sequence_classifier_inputs(example) if self.wrapper.task_helper else None
        if inputs is None:
            inputs = self.wrapper.tokenizer.encode_plus(
                example.text_a if example.text_a else None,
                example.text_b if example.text_b else None,
                add_special_tokens=True,
                max_length=self.wrapper.config.max_seq_length,
            )
        input_ids, token_type_ids = inputs["input_ids"], inputs.get("token_type_ids")

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        if not token_type_ids:
            token_type_ids = [0] * self.wrapper.config.max_seq_length
        else:
            token_type_ids = token_type_ids + ([0] * padding_length)
        mlm_labels = [-1] * len(input_ids)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100
        logits = example.logits if example.logits else [-1]

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits, idx=example.idx)
