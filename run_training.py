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
This script can be used to train and evaluate either a regular supervised model or a PET model on
one of the supported tasks and datasets.
"""

import argparse
import os
import statistics
from collections import defaultdict
import torch

from tasks import PROCESSORS, load_examples
from utils import set_seed, eq_div, save_logits, LogitsList, InputExample
from wrapper import TransformerModelWrapper, WRAPPER_TYPES
import log

logger = log.get_logger('root')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_examples", required=True, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--wrapper_type", required=True, choices=WRAPPER_TYPES,
                        help="The wrapper type - either sequence_classifier (corresponding to"
                             "regular supervised training) or mlm (corresponding to PET training)")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="The model type (currently supported are bert and roberta)")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(PROCESSORS.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Optional parameters
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--lm_train_examples_per_label", default=10000, type=int,
                        help="The total number of training examples for auxiliary language modeling, "
                             "where -1 equals all examples")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET training)")
    parser.add_argument("--repetitions", default=3, type=int,
                        help="The number of times to repeat training and testing with different seeds.")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET training)")
    parser.add_argument("--save_train_logits", action='store_true',
                        help="Whether to save logits on the lm_train_examples in a separate file. This takes some "
                             "additional time but is required for combining PVPs  (only for PET training)")
    parser.add_argument("--additional_data_dir", default=None, type=str,
                        help="Path to a directory containing additional automatically labeled training examples (only "
                             "for iPET)")
    parser.add_argument("--per_gpu_helper_batch_size", default=4, type=int,
                        help="Batch size for the auxiliary task (only for PET training)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary task (only for PET training)")
    parser.add_argument("--temperature", default=1, type=float,
                        help="Temperature used for combining PVPs (only for PET training)")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET training)")
    parser.add_argument("--logits_file", type=str,
                        help="The logits file for combining multiple PVPs, which can be created using the"
                             "merge_logits.py script")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to perform lower casing")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.output_mode = "classification"
    args.label_list = processor.get_labels()
    args.use_logits = args.logits_file is not None

    wrapper = None

    logger.info("Training/evaluation parameters: {}".format(args))
    results = defaultdict(list)

    train_examples_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
    test_examples_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1

    train_data = load_examples(args.task_name, args.data_dir, train_examples_per_label, evaluate=False)
    eval_data = load_examples(args.task_name, args.data_dir, test_examples_per_label, evaluate=True)

    if args.lm_training or args.save_train_logits or args.use_logits:
        all_train_data = load_examples(args.task_name, args.data_dir, args.lm_train_examples_per_label, evaluate=False)
    else:
        all_train_data = None

    if args.use_logits:
        logits = LogitsList.load(args.logits_file).logits
        assert len(logits) == len(all_train_data)
        logger.info("Got {} logits from file {}".format(len(logits), args.logits_file))
        for example, example_logits in zip(all_train_data, logits):
            example.logits = example_logits

    for pattern_id in args.pattern_ids:
        args.pattern_id = pattern_id
        for iteration in range(args.repetitions):

            results_dict = {}

            output_dir = "{}/p{}-i{}".format(args.output_dir, args.pattern_id, iteration)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if args.do_train:
                wrapper = TransformerModelWrapper(args)
                wrapper.model.to(device)

                results_dict['train_set_before_training'] = wrapper.eval(train_data, device, **vars(args))['acc']

                pattern_iter_train_data = []
                pattern_iter_train_data.extend(train_data)

                if args.additional_data_dir:
                    p = os.path.join(args.additional_data_dir, 'p{}-i{}-train.txt'.format(args.pattern_id, iteration))
                    additional_data = InputExample.load_examples(p)
                    for example in additional_data:
                        example.logits = None
                    pattern_iter_train_data.extend(additional_data)
                    logger.info("Loaded {} additional examples from {}, total training size is now {}".format(
                        len(additional_data), p, len(pattern_iter_train_data)
                    ))

                logger.info("Starting training...")

                global_step, tr_loss = wrapper.train(
                    pattern_iter_train_data, device,
                    helper_train_data=all_train_data if args.lm_training or args.use_logits else None,
                    tmp_dir=output_dir, **vars(args))

                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
                logger.info("Training complete")

                results_dict['train_set_after_training'] = wrapper.eval(train_data, device, **vars(args))['acc']

                with open(os.path.join(output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                logger.info("Saving trained model at {}...".format(output_dir))
                wrapper.save(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving complete")

                if args.save_train_logits:
                    logits = wrapper.eval(all_train_data, device, output_logits=True, **vars(args))
                    save_logits(os.path.join(output_dir, 'logits.txt'), logits)

                if not args.do_eval:
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

            # Evaluation
            if args.do_eval:
                logger.info("Starting evaluation...")
                if not wrapper:
                    wrapper = TransformerModelWrapper.from_pretrained(output_dir)
                    wrapper.model.to(device)

                result = wrapper.eval(eval_data, device, **vars(args))
                logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info(result)

                results_dict['test_set_after_training'] = result['acc']
                with open(os.path.join(output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                for key, value in result.items():
                    results['{}-p{}'.format(key, args.pattern_id)].append(value)

                wrapper.model = None
                torch.cuda.empty_cache()

    logger.info("=== OVERALL RESULTS ===")

    with open(os.path.join(args.output_dir, 'result_test.txt'), 'w') as fh:
        for key, values in results.items():
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            result_str = "{}: {} +- {}".format(key, mean, stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

        all_results = [result for pattern_results in results.values() for result in pattern_results]
        all_mean = statistics.mean(all_results)
        all_stdev = statistics.stdev(all_results)
        result_str = "acc-all-p: {} +- {}".format(all_mean, all_stdev)
        logger.info(result_str)
        fh.write(result_str + '\n')


if __name__ == "__main__":
    main()
