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
import copy
import json
import jsonpickle
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, BertForMaskedLM, \
    RobertaForMaskedLM, XLMRobertaForMaskedLM, XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, \
    XLNetLMHeadModel, BertConfig, BertForSequenceClassification, BertTokenizer, RobertaConfig, \
    RobertaForSequenceClassification, RobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, \
    XLMRobertaTokenizer, AlbertForSequenceClassification, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig, \
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, PegasusConfig, PegasusTokenizer, Adafactor
from transformers import __version__ as transformers_version

import log
from pet import preprocessor
from pet.config import WrapperConfig
from pet.modeling_pegasus import PetPegasusForConditionalGeneration
from pet.pvp import GenerativePVP
from pet.tasks import TASK_HELPERS
from pet.utils import InputExample, InputFeatures, DictDataset, distillation_loss, \
    get_sqrt_schedule_with_warmup, GenerativeInputExample, LabelSmoothingLoss

logger = log.get_logger('root')

CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"
PLM_WRAPPER = "plm"
GENERATIVE_WRAPPER = "generative"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, PLM_WRAPPER, GENERATIVE_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: preprocessor.SequenceClassifierPreprocessor,
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
    PLM_WRAPPER: preprocessor.PLMPreprocessor,
    GENERATIVE_WRAPPER: preprocessor.GenerativePreprocessor,
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
    },
    'xlm-roberta': {
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLMRobertaForSequenceClassification,
        MLM_WRAPPER: XLMRobertaForMaskedLM
    },
    'xlnet': {
        'config': XLNetConfig,
        'tokenizer': XLNetTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLNetForSequenceClassification,
        PLM_WRAPPER: XLNetLMHeadModel
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AlbertForSequenceClassification,
        MLM_WRAPPER: AlbertForMaskedLM
    },
    'gpt2': {
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        MLM_WRAPPER: GPT2LMHeadModel
    },
    'pegasus': {
        'config': PegasusConfig,
        'tokenizer': PegasusTokenizer,
        GENERATIVE_WRAPPER: PetPegasusForConditionalGeneration
    }
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_eval_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_eval_step,
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_train_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: wrapper.sequence_classifier_train_step,
    GENERATIVE_WRAPPER: lambda wrapper: wrapper.generative_train_step,
}

PEGASUS_MASK_SENTENCE_TOKEN_ID = 2


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        """Create a new wrapper from the given config."""
        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(
            config.model_name_or_path, num_labels=len(config.label_list),
            cache_dir=config.cache_dir if config.cache_dir else None)

        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)  # type: PreTrainedTokenizer

        if self.config.model_type == 'pegasus':
            self.tokenizer.add_special_tokens(
                {'mask_token': self.tokenizer.convert_ids_to_tokens(PEGASUS_MASK_SENTENCE_TOKEN_ID)}
            )

        if self.config.model_type == 'gpt2':
            self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token

        self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None)

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self, self.config.task_name, self.config.pattern_ids,
                                                                    self.config.verbalizer_file)
        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_ids, wrapper.config.verbalizer_file)
        wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
            if wrapper.config.task_name in TASK_HELPERS else None
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def train(self, task_train_data: List[InputExample], device, per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, optimizer: str = "adam", adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, per_gpu_unlabeled_batch_size: int = 8, unlabeled_data: List[InputExample] = None,
              lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
              epsilon: float = 0.1, max_steps: int = -1, **_):
        """
        Train the underlying language model.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param optimizer: the optimizer to use (supported optimizers are 'adam' and 'adafactor')
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param unlabeled_data: the unlabeled examples to use
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use the example's logits instead of their labels to compute the loss
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for knowledge distillation
        :param epsilon: the amount of label smoothing to use, between 0 and 1
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)

        if not task_train_data:
            train_dataloader = None
        else:
            train_dataset = self._generate_dataset(task_train_data, pattern_ids=self.config.pattern_ids)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        unlabeled_dataloader, unlabeled_iter = None, None

        if lm_training or use_logits:
            # we need unlabeled data both for auxiliary language modeling and for knowledge distillation
            assert unlabeled_data is not None
            unlabeled_batch_size = per_gpu_unlabeled_batch_size * max(1, n_gpu)
            unlabeled_dataset = self._generate_dataset(unlabeled_data, pattern_ids=self.config.pattern_ids, labelled=False)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset, sampler=unlabeled_sampler,
                                              batch_size=unlabeled_batch_size)
            unlabeled_iter = unlabeled_dataloader.__iter__()

        if use_logits:
            train_dataloader = unlabeled_dataloader

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        if optimizer == 'adafactor':
            optimizer = Adafactor(self.model.parameters(), lr=learning_rate, relative_step=False, scale_parameter=False)
            scheduler = get_sqrt_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif optimizer == 'adam':
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError(f"Unknown optimizer choice: '{optimizer}'")

        # multi-gpu training
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(device)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                unlabeled_batch = None

                batch = {k: t.to(device) for k, t in batch.items()}

                if lm_training:
                    while unlabeled_batch is None:
                        try:
                            unlabeled_batch = unlabeled_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting unlabeled dataset")
                            unlabeled_iter = unlabeled_dataloader.__iter__()

                    lm_input_ids = unlabeled_batch['input_ids']
                    unlabeled_batch['input_ids'], unlabeled_batch['mlm_labels'] = self._mask_tokens(lm_input_ids)
                    unlabeled_batch = {k: t.to(device) for k, t in unlabeled_batch.items()}

                train_step_inputs = {
                    'unlabeled_batch': unlabeled_batch, 'lm_training': lm_training, 'alpha': alpha,
                    'use_logits': use_logits, 'temperature': temperature, 'epsilon': epsilon,
                }
                loss = self.task_helper.train_step(batch, **train_step_inputs) if self.task_helper else None

                if loss is None:
                    loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch, **train_step_inputs)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        if scheduler:
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
             priming: bool = False, decoding_strategy: str = 'default', pattern_id: Optional[int] = None) -> Dict:
        """
        Evaluate the underlying language model.

        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu)
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param priming: whether to use priming
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr' or 'parallel')
        :param pattern_id: the id of the pattern to be used
        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.
        """

        assert pattern_id is not None or len(self.config.pattern_ids) == 1
        pattern_ids = [pattern_id] if pattern_id is not None else [self.config.pattern_ids[0]]

        eval_dataset = self._generate_dataset(eval_data, pattern_ids=pattern_ids, priming=priming)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(device)

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels']
            indices = batch['idx']
            with torch.no_grad():

                # some tasks require special evaluation
                logits = self.task_helper.eval_step(batch,
                                                    decoding_strategy=decoding_strategy) if self.task_helper else None

                if logits is None:
                    logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        return {
            'indices': all_indices,
            'logits': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

    def get_sequence_log_probabilities(self, eval_data: List[GenerativeInputExample], device, pattern_id: int,
                                       per_gpu_eval_batch_size: int = 8, n_gpu: int = 1):
        """
        Compute the log probabilities for a list of input examples.
        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu)
        :param pattern_id: the id of the pattern to use for computing log probabilities
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :return: a list of log probabilities, where the `i`th probability corresponds to the `i`th input
        """
        eval_dataset = self._generate_dataset(eval_data, pattern_ids=[pattern_id])
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.to(device)

        output_log_probabilities = []

        assert all(isinstance(pvp, GenerativePVP) for pvp in self.preprocessor.pvps.values())

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()

            output_prefix_ids = self.preprocessor.pvps[pattern_id].generative_prefix_ids()

            output_prefix_ids = torch.tensor([output_prefix_ids], dtype=torch.long) \
                .repeat(batch['output_ids'].shape[0], 1)

            batch = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': torch.cat([output_prefix_ids, batch['output_ids']], dim=1)
            }

            labels = batch['labels']
            first_token_tensor = torch.tensor(
                [[self.tokenizer.pad_token_id]], device=labels.device
            ).repeat(labels.shape[0], 1)
            batch['decoder_input_ids'] = torch.cat([first_token_tensor, labels], dim=1)[:, :-1]

            batch = {k: t.to(device) for k, t in batch.items()}

            with torch.no_grad():
                logits = model(**batch)[1]
                log_probs = self._compute_log_probs(logits, batch, ignore_first=output_prefix_ids.shape[1])
                output_log_probabilities.extend(log_probs.tolist())

        return output_log_probabilities

    def _compute_log_probs(self, logits, batch, ignore_first: int = 0):
        """
        Compute the log probability of each element in the batch.
        :param logits: the logits for each example in the batch
        :param batch: the batch, with required fields 'input_ids', 'attention_mask' and 'labels'
        :param ignore_first: if set to a value > 0, the first `ignore_first` tokens are masked out for loss computation
        :return: a list of log probabilities, where the `i`th probability corresponds to example `i` in the batch
        """
        if ignore_first > 0:
            labels = batch['labels']
            mask = torch.tensor([0] * ignore_first + [1] * (labels.shape[1] - ignore_first)).to(labels.device)
            batch['labels'] = labels * mask + self.tokenizer.pad_token_id * (1 - mask)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')
        loss = loss_fct(logits.permute([1, 2, 0]), batch['labels'].permute([1, 0]))
        loss = loss.permute([1, 0])

        num_tokens = (batch['labels'] != self.tokenizer.pad_token_id).sum(dim=1)
        assert not any(num_tokens == 0)

        avg_loss = loss.sum(dim=1)
        return -avg_loss

    def generate_new_examples(self, gen_data: List[GenerativeInputExample], device, pattern_id: int, per_gpu_batch_size: int = 8,
                              n_gpu: int = 1):
        outputs = self.generate(gen_data, device, pattern_id=pattern_id, per_gpu_eval_batch_size=per_gpu_batch_size, n_gpu=n_gpu)
        new_examples = []
        for example, prediction in zip(gen_data, outputs['predictions']):
            new_example = copy.deepcopy(example)
            new_example.output_text = prediction
            try:
                new_example.output_text = prediction[:prediction.rindex('.') + 1]
            except ValueError:
                # text does not contain a full stop, continue
                pass
            new_examples.append(new_example)
        return new_examples

    def generate(self, eval_data: List[InputExample], device, pattern_id: int = None, pattern_ids: List[int] = None,
                 per_gpu_eval_batch_size: int = 8, n_gpu: int = 1, joint_decoding: bool = True, flatten: bool = True):

        assert (pattern_id is not None) ^ (pattern_ids is not None), "Exactly one of 'pattern_id' and 'pattern_ids' must be given"
        if pattern_id is not None:
            pattern_ids = [pattern_id]

        if not joint_decoding and flatten:
            raise ValueError("`flatten` must be set to False if no joint decoding is performed.")

        if per_gpu_eval_batch_size % len(pattern_ids) != 0:
            logger.warning(f"Batch size ({per_gpu_eval_batch_size}) must be a multiple of the number of patterns ({len(pattern_ids)})")
            per_gpu_eval_batch_size = max(1, per_gpu_eval_batch_size // len(pattern_ids)) * len(pattern_ids)
            logger.warning(f"Setting batch size to {per_gpu_eval_batch_size}")

        eval_dataset = self._generate_dataset(eval_data, pattern_ids=pattern_ids)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.to(device)

        indices = [example.idx for example in eval_data]
        output_texts = []

        assert all(isinstance(pvp, GenerativePVP) for pvp in self.preprocessor.pvps.values())
        output_prefix_ids = [self.preprocessor.pvps[pattern_id].generative_prefix_ids() for pattern_id in pattern_ids]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = {k: t.to(device) for k, t in batch.items()}

            with torch.no_grad():
                grouped_output_ids = model.generate(**batch, num_beams=1, max_length=self.config.output_max_seq_length,
                                                    output_prefix_ids=output_prefix_ids, joint_decoding=joint_decoding)

                processed_output_ids = []

                for all_output_ids_for_input in grouped_output_ids:
                    assert len(all_output_ids_for_input) == len(output_prefix_ids)
                    all_output_ids_for_input = [output_ids[len(output_prefix_ids[pattern_idx]) + 1:] for pattern_idx, output_ids in
                                                enumerate(all_output_ids_for_input)]

                    if joint_decoding:
                        assert len(set(tuple(output_ids.tolist()) for output_ids in all_output_ids_for_input)) == 1
                        all_output_ids_for_input = all_output_ids_for_input[:1]

                    processed_output_ids.append(all_output_ids_for_input)

                tgt_texts = [self.tokenizer.batch_decode(output_ids, skip_special_tokens=True) for output_ids in processed_output_ids]
                tgt_texts = [[text.replace('<n>', ' ') for text in texts] for texts in tgt_texts]

                if flatten:
                    assert all(len(text_list) == 1 for text_list in tgt_texts)
                    tgt_texts = [texts[0] for texts in tgt_texts]

                output_texts.extend(tgt_texts)

        return {
            'predictions': output_texts,
            'indices': indices
        }

    def _generate_dataset(self, data: List[InputExample], pattern_ids: List[int], labelled: bool = True,
                          priming: bool = False) -> DictDataset:
        features = self._convert_examples_to_features(data, pattern_ids=pattern_ids, labelled=labelled, priming=priming)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'mlm_labels': torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'idx': torch.tensor([f.idx for f in features], dtype=torch.long)
        }
        if self.config.wrapper_type == PLM_WRAPPER:
            feature_dict['perm_mask'] = torch.tensor([f.perm_mask for f in features], dtype=torch.float)
            feature_dict['target_mapping'] = torch.tensor([f.target_mapping for f in features], dtype=torch.float)

        elif self.config.wrapper_type == GENERATIVE_WRAPPER:
            feature_dict['output_ids'] = torch.tensor([f.output_ids for f in features], dtype=torch.long)
            feature_dict['output_loss_mask'] = torch.tensor([f.output_loss_mask for f in features], dtype=torch.long)
            if 'token_ids' in features[0].meta:
                tprob_key = 'token_probabilities'
                feature_dict['token_ids'] = torch.tensor([f.meta['token_ids'] for f in features], dtype=torch.long)
                feature_dict[tprob_key] = torch.tensor([f.meta[tprob_key] for f in features], dtype=torch.float)

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], pattern_ids: List[int], labelled: bool = True,
                                      priming: bool = False) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            for pattern_id in pattern_ids:
                input_features = self.preprocessor.get_input_features(
                    example, pattern_id=pattern_id, labelled=labelled, priming=priming)
                if self.task_helper:
                    self.task_helper.add_special_input_features(example, input_features)
                features.append(input_features)

        logger.info(f"Created {len(features)} input features from {len(examples)} examples...")
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

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.model_type in ['bert', 'xlnet']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs

    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor],
                       unlabeled_batch: Optional[Dict[str, torch.Tensor]] = None, lm_training: bool = False,
                       alpha: float = 0, **_) -> torch.Tensor:
        """Perform a MLM training step."""

        assert len(self.preprocessor.pvps) == 1
        pvp = self.preprocessor.pvps[0]

        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']

        outputs = self.model(**inputs)
        prediction_scores = pvp.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        if lm_training:
            lm_inputs = self.generate_default_inputs(unlabeled_batch)
            lm_inputs['masked_lm_labels'] = unlabeled_batch['mlm_labels']
            lm_loss = self.model(**lm_inputs)[0]
            loss = alpha * loss + (1 - alpha) * lm_loss
        return loss

    def plm_train_step(self, labeled_batch: Dict[str, torch.Tensor], lm_training: bool = False, **_):
        """Perform a PLM training step."""

        assert len(self.preprocessor.pvps) == 1
        pvp = self.preprocessor.pvps[0]

        inputs = self.generate_default_inputs(labeled_batch)
        inputs['perm_mask'], inputs['target_mapping'] = labeled_batch['perm_mask'], labeled_batch['target_mapping']
        labels = labeled_batch['labels']
        outputs = self.model(**inputs)
        prediction_scores = pvp.convert_plm_logits_to_cls_logits(outputs[0])
        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))

        if lm_training:
            raise NotImplementedError("Language model training is currently not implemented for PLMs")

        return loss

    def generative_train_step(self, labeled_batch: Dict[str, torch.Tensor], epsilon: float = 0.1, **_):

        inputs = self.generate_default_inputs(labeled_batch)
        inputs['labels'] = labeled_batch['output_ids']

        # we need to manually provide the decoder_input_ids here due to a bug in the original implementation that
        # shifts all tokens to the right (which is correct for BART, but not for PEGASUS) instead of inserting <pad>.
        labels = inputs['labels']
        first_token_tensor = torch.tensor(
            [[self.tokenizer.pad_token_id]], device=labels.device
        ).repeat(labels.shape[0], 1)

        inputs['decoder_input_ids'] = torch.cat([first_token_tensor, labels], dim=1)[:, :-1]

        lm_logits = self.model(**inputs, use_cache=False)[1]
        cfg = self.model.module.config if hasattr(self.model, 'module') else self.model.config

        labels = labeled_batch['output_ids']
        label_mask = labeled_batch['output_loss_mask']
        labels = labels * label_mask + (1 - label_mask) * self.tokenizer.pad_token_id

        loss_fct = LabelSmoothingLoss(label_smoothing=epsilon, tgt_vocab_size=cfg.vocab_size,
                                      ignore_index=self.tokenizer.pad_token_id).to(labels.device)
        loss = loss_fct(lm_logits.view(-1, cfg.vocab_size), labels.view(-1))
        return loss

    def sequence_classifier_train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False,
                                       temperature: float = 1, **_) -> torch.Tensor:
        """Perform a sequence classifier training step."""

        inputs = self.generate_default_inputs(batch)
        if not use_logits:
            inputs['labels'] = batch['labels']

        outputs = self.model(**inputs)

        if use_logits:
            logits_predicted, logits_target = outputs[0], batch['logits']
            return distillation_loss(logits_predicted, logits_target, temperature)
        else:
            return outputs[0]

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        assert len(self.preprocessor.pvps) == 1
        pvp = self.preprocessor.pvps[0]
        return pvp.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])

    def plm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a PLM evaluation step."""
        inputs = self.generate_default_inputs(batch)
        inputs['perm_mask'], inputs['target_mapping'] = batch['perm_mask'], batch['target_mapping']
        outputs = self.model(**inputs)
        assert len(self.preprocessor.pvps) == 1
        pvp = self.preprocessor.pvps[0]
        return pvp.convert_plm_logits_to_cls_logits(outputs[0])

    def sequence_classifier_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier evaluation step."""
        inputs = self.generate_default_inputs(batch)
        return self.model(**inputs)[0]
