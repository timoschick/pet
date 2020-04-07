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
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
import math

import jsonpickle
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, BertForMaskedLM, \
    RobertaForMaskedLM
from transformers import (BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from transformers.data.metrics import simple_accuracy

import log
from preprocessor import SequenceClassifierPreprocessor, MLMPreprocessor
from utils import InputFeatures

logger = log.get_logger('root')

CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: SequenceClassifierPreprocessor,
    MLM_WRAPPER: MLMPreprocessor
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: RobertaForSequenceClassification,
        MLM_WRAPPER: RobertaForMaskedLM
    }
}


def distillation_loss(predictions, targets, temperature):
    p = F.log_softmax(predictions / temperature, dim=1)
    q = F.softmax(targets / temperature, dim=1)
    return F.kl_div(p, q, reduction='sum') * (temperature ** 2) / predictions.shape[0]


class WrapperConfig(object):
    def __init__(self, model_type, wrapper_type, task_name, max_seq_length: int, label_list: List[str],
                 pattern_id: int, verbalizer_file: str):
        self.model_type = model_type
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file


class TransformerModelWrapper:

    def __init__(self, args):
        self.config = WrapperConfig(
            model_type=args.model_type, wrapper_type=args.wrapper_type, task_name=args.task_name,
            max_seq_length=args.max_seq_length, label_list=args.label_list, pattern_id=args.pattern_id,
            verbalizer_file=args.verbalizer_file
        )

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(
            args.model_name_or_path, num_labels=len(args.label_list), finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None)  # type: PreTrainedTokenizer

        self.model = model_class.from_pretrained(args.model_name_or_path, config=model_config,
                                                 cache_dir=args.cache_dir if args.cache_dir else None)

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self, self.config.task_name, self.config.pattern_id,
                                                                    self.config.verbalizer_file)

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file)
        return wrapper

    def save(self, path: str) -> None:
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    def _load_config(self, path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self, task_train_data: List[InputExample], device, per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, per_gpu_helper_batch_size: int = 8, helper_train_data: List[InputExample] = None,
              lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
              max_steps=-1, **_):

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(task_train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        helper_dataloader, helper_iter = None, None

        if lm_training or use_logits:
            assert helper_train_data is not None
            helper_batch_size = per_gpu_helper_batch_size
            helper_dataset = self._generate_dataset(helper_train_data, labelled=False)
            helper_sampler = RandomSampler(helper_dataset)
            helper_dataloader = DataLoader(helper_dataset, sampler=helper_sampler, batch_size=helper_batch_size)
            helper_iter = helper_dataloader.__iter__()

        if use_logits:
            train_dataloader = helper_dataloader

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        t_epoch = len(train_dataloader) // gradient_accumulation_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()

                if lm_training:
                    helper_batch = None
                    while helper_batch is None:
                        try:
                            helper_batch = helper_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting helper batch")
                            helper_iter = helper_dataloader.__iter__()

                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.config.model_type in ['bert', 'xlnet'] else None}

                if self.config.wrapper_type == SEQUENCE_CLASSIFIER_WRAPPER and not use_logits:
                    inputs['labels'] = batch[3]

                inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
                outputs = self.model(**inputs)

                if use_logits:
                    logits_predicted = outputs[0]
                    logits_target = batch[5].to(device)
                    loss = distillation_loss(logits_predicted, logits_target, temperature)
                elif self.config.wrapper_type == SEQUENCE_CLASSIFIER_WRAPPER:
                    loss = outputs[0]
                else:
                    mlm_labels = batch[4].to(device)
                    prediction_scores = self.preprocessor.verbalizer.convert_mlm_logits_to_cls_logits(mlm_labels,
                                                                                                      outputs[0])
                    labels = batch[3].to(device)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

                if lm_training:
                    lm_inputs = {
                        'input_ids': helper_batch[0], 'attention_mask': helper_batch[1],
                        'masked_lm_labels': helper_batch[4],
                        'token_type_ids': helper_batch[2] if self.config.model_type in ['bert', 'xlnet'] else None}

                    lm_inputs['input_ids'], lm_inputs['masked_lm_labels'] = self._mask_tokens(lm_inputs['input_ids'])
                    lm_inputs = {k: v.to(device) if v is not None else None for k, v in lm_inputs.items()}
                    lm_outputs = self.model(**lm_inputs)
                    lm_loss = lm_outputs[0]
                    loss = alpha * loss + (1 - alpha) * lm_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        print(json.dumps({**logs, **{'step': global_step}}))

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return global_step, (tr_loss / global_step if global_step > 0 else -1)

    def eval(self, eval_data: List[InputExample], device, per_gpu_eval_batch_size: int = 8, n_gpu: int = 1,
             output_logits: bool = False, **_):

        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(device) for t in batch)
            labels = batch[3]
            mlm_labels = batch[4]

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.config.model_type in ['bert', 'xlnet'] else None}
                outputs = self.model(**inputs)
                logits = outputs[0]
                if self.config.wrapper_type == MLM_WRAPPER:
                    logits = self.preprocessor.verbalizer.convert_mlm_logits_to_cls_logits(mlm_labels, logits)
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        if output_logits:
            return preds

        preds = np.argmax(preds, axis=1)
        return {"acc": simple_accuracy(preds, out_label_ids)}

    def _generate_dataset(self, data: List[InputExample], labelled: bool = True):
        features = self._convert_examples_to_features(data, labelled)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)
        all_logits = torch.tensor([f.logits for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_mlm_labels,
                                all_logits)
        return dataset

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled)
            features.append(input_features)
        return features

    def _mask_tokens(self, input_ids):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels
