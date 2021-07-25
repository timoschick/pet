import copy
import json
import math
import os
import random
from collections import defaultdict
from typing import List

import torch

import log
import pet
import pet.config
from pet.utils import GenerativeInputExample

logger = log.get_logger('root')


def load_distillation_examples(path: str, original_train_data: List[GenerativeInputExample], samples_per_example: int = 2,
                               cutoff_percentage: float = 0) -> List[GenerativeInputExample]:
    raw_data = GenerativeInputExample.load_examples(path)
    dist_examples = _build_distillation_examples(raw_data, original_train_data=original_train_data, samples_per_example=samples_per_example,
                                                 cutoff_percentage=cutoff_percentage)
    logger.info(f"Created {len(dist_examples)} distillation examples from {len(raw_data)} annotated examples found at {path}")
    return dist_examples


def _build_distillation_examples(raw_data: List[GenerativeInputExample], original_train_data: List[GenerativeInputExample], seed: int = 42,
                                 samples_per_example: int = 2, cutoff_percentage: float = 0) -> List[GenerativeInputExample]:
    rng = random.Random(seed)
    dist_examples = []

    if cutoff_percentage > 0:
        # remove the least probable outputs (i.e., the bottom `cutoff_percentage` percent of outputs)
        outputs_list = []
        for example in raw_data:
            for output, output_log_prob_raw in zip(example.meta['outputs'], example.meta['output_probabilities']):
                outputs_list.append((output_log_prob_raw, output, example))
        outputs_list.sort(key=lambda x: x[0])

        # perform cutoff
        cutoff_idx = int(len(outputs_list) * cutoff_percentage)
        outputs_list = outputs_list[cutoff_idx:]

        valid_outputs = defaultdict(set)
        for score, output, example in outputs_list:
            valid_outputs[example].add(output)

        for example in raw_data:
            new_outputs, new_output_probs = [], []
            for output, output_prob in zip(example.meta['outputs'], example.meta['output_probabilities_normalized']):
                if output in valid_outputs[example]:
                    new_outputs.append(output)
                    new_output_probs.append(output_prob)
            example.meta['outputs'] = new_outputs
            example.meta['output_probabilities_normalized'] = new_output_probs

    for example in raw_data:
        if not example.meta['outputs']:
            continue

        output_texts_with_indices = rng.choices(
            population=list(enumerate(example.meta['outputs'])),
            weights=example.meta['output_probabilities_normalized'],
            k=samples_per_example)

        for output_idx, output_text in output_texts_with_indices:
            copy_example = copy.deepcopy(example)
            copy_example.meta = {}
            copy_example.output_text = output_text
            dist_examples.append(copy_example)

    for _ in range(samples_per_example):
        dist_examples += original_train_data

    return dist_examples


def compute_output_probabilities(examples: List[GenerativeInputExample], output_dir: str, eval_config: pet.config.EvalConfig,
                                 model_config: pet.config.WrapperConfig, use_untrained_model: bool = False, uniform_sampling: bool = False) -> None:
    subdirs = next(os.walk(output_dir))[1]
    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    idx_to_outputs = defaultdict(set)

    # Step 1: Load all outputs and merge them, so that for each index, we obtain a set of all possible outputs.
    valid_subdirs = []

    for subdir in subdirs:
        predictions_files = []
        for filename in os.listdir(os.path.join(output_dir, subdir)):
            if filename.startswith('unlabeled_predictions') and filename.endswith('.jsonl'):
                predictions_files.append(os.path.join(output_dir, subdir, filename))

        if not predictions_files:
            logger.warning(f"Skipping subdirectory '{subdir}' because it contains no file that matches 'unlabeled_predictions*.jsonl'...")
            continue

        for predictions_file in predictions_files:
            with open(predictions_file, 'r', encoding='utf8') as fh:
                for line in fh:
                    content = json.loads(line)
                    idx_to_outputs[content['idx']].add(content['label'])

        valid_subdirs.append(subdir)

    eval_examples = []

    for example in examples:
        if not idx_to_outputs[example.idx]:
            raise ValueError(f"No outputs found for example {example.idx}")

        for output in idx_to_outputs[example.idx]:
            copy_example = copy.deepcopy(example)
            copy_example.output_text = output
            copy_example.meta['output_log_probabilities'] = []
            eval_examples.append(copy_example)

    # Step 2: Load each model and use it to compute a probability for each possible output.
    if use_untrained_model:
        wrapper = pet.init_model(model_config)

    for subdir in valid_subdirs:
        if uniform_sampling:
            for example in eval_examples:
                example.meta['output_log_probabilities'].append(0)
            continue

        if not use_untrained_model:
            try:
                wrapper = pet.TransformerModelWrapper.from_pretrained(os.path.join(output_dir, subdir))
            except FileNotFoundError:
                # model does not exist, we are in a zero shot setting
                pattern_identifier = subdir.split('-')[0]
                assert pattern_identifier[0] == 'p'
                pattern_id = int(pattern_identifier[1:])
                logger.warning(f"No model was found at {os.path.join(output_dir, subdir)}, initializing a new model "
                               f"with pattern id {pattern_id} from scratch.")
                model_config.pattern_ids = [pattern_id]
                wrapper = pet.init_model(model_config)

        device = torch.device(eval_config.device if eval_config.device else "cuda" if torch.cuda.is_available() else "cpu")

        for pattern_id in wrapper.config.pattern_ids:
            output_log_probabilities = wrapper.get_sequence_log_probabilities(
                eval_examples, device=device, pattern_id=pattern_id, per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
                n_gpu=eval_config.n_gpu)

            assert len(eval_examples) == len(output_log_probabilities)
            for example, log_probability in zip(eval_examples, output_log_probabilities):
                example.meta['output_log_probabilities'].append(log_probability)

        wrapper.model = None
        torch.cuda.empty_cache()

    idx_to_examples = defaultdict(list)
    for example in eval_examples:
        output_log_probabilities = example.meta['output_log_probabilities']
        example.meta['output_log_probability'] = sum(output_log_probabilities) / len(output_log_probabilities)
        idx_to_examples[example.idx].append(example)

    # Step 3: Assign the outputs and probabilities to the original examples.
    for example in examples:
        if example.idx not in idx_to_examples:
            raise ValueError(f"No outputs were found for input example {example.idx}")
        example_copies = idx_to_examples[example.idx]
        output_probabilities = [math.exp(ex.meta['output_log_probability']) for ex in example_copies]
        output_probabilities_sum = sum(output_probabilities)

        if output_probabilities_sum == 0:
            output_probabilities_normalized = [1 / len(output_probabilities) for _ in output_probabilities]
        else:
            output_probabilities_normalized = [prob / output_probabilities_sum for prob in output_probabilities]

        example.meta['outputs'] = [ex.output_text for ex in example_copies]
        example.meta['output_probabilities_normalized'] = output_probabilities_normalized
        example.meta['output_probabilities'] = output_probabilities
