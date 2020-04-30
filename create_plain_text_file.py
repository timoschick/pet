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
This script can be used to convert a labeled dataset into a plain text
file that can be used for unsupervised language model pretraining.
"""
import argparse
import log

from tasks import load_examples

logger = log.get_logger('root')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output file where the plain text will be written.")

    args = parser.parse_args()
    train_data = load_examples(args.task_name, args.data_dir, examples_per_label=-1, evaluate=False)

    with open(args.output_file, 'w', encoding='utf8') as fh:
        for input_example in train_data:
            if input_example.text_b:
                fh.write('{} {}\n'.format(input_example.text_a, input_example.text_b))
            else:
                fh.write('{}\n'.format(input_example.text_a))

    logger.info("Done writing plain text for {} examples to {}".format(len(train_data), args.output_file))


if __name__ == '__main__':
    main()
