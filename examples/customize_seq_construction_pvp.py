"""
An example script for generating sequences based on customized PVPs (likely for text augmentation usage).
"""
import string
from typing import List
from pet.modeling import init_model
from pet.wrapper import WrapperConfig
from pet.pvp import PVP, PVPS
from pet.utils import InputExample


class CustomizeRtePVP(PVP):
    # check more pvp impls (:pet.pvp.py) to get inspirations
    VERBALIZER = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }
    TASK_NAME = "custom_rte"

    def get_parts(self, example: InputExample):
        # switch text_a and text_b to get the correct order
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))

        if self.pattern_id == 0:
            return ['"', text_b, '" ?'], [self.mask, ', "', text_a, '"']
        elif self.pattern_id == 1:
            return [text_b, '?'], [self.mask, ',', text_a]
        if self.pattern_id == 2:
            return ['"', text_b, '" ?'], [self.mask, '. "', text_a, '"']
        elif self.pattern_id == 3:
            return [text_b, '?'], [self.mask, '.', text_a]
        elif self.pattern_id == 4:
            return [text_a, ' question: ', self.shortenable(example.text_b), ' True or False? answer:', self.mask], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return ['true'] if label == 'entailment' else ['false']
        return CustomizeRtePVP.VERBALIZER[label]


# register the PVP for this task with its name
PVPS[CustomizeRtePVP.TASK_NAME] = CustomizeRtePVP

# configure pattern id that can be 0,1,2,3,4 in this case
pattern_id = 4
model_config = {'model_type': "bert", 'model_name_or_path': "bert-base-cased", 'wrapper_type': 'mlm',
                'task_name': CustomizeRtePVP.TASK_NAME,
                'max_seq_length': 256, 'label_list': ['entailment', 'not_entailment'], 'pattern_id': pattern_id,
                'verbalizer_file': None, 'cache_dir': ''}

config = WrapperConfig(**model_config)
model_wrapper = init_model(config)

# here take two RTE examples as example
example1 = {'guid': 'train-2324',
            'text_a': 'The incident in Mogadishu, the Somali capital, came as U.S. forces began the  final phase of their promised March 31 pullout.',
            'text_b': 'The capital of Somalia is Mogadishu.', 'label': 'not_entailment', 'logits': None,
            'idx': 2324,
            'meta': {}}
example2 = {'guid': 'train-1340',
            'text_a': "By the time a case of rabies is confirmed, the disease may have taken hold in the area.",
            'text_b': 'A case of rabies was confirmed.',
            'label': 'not_entailment', 'logits': None, 'idx': 1340, 'meta': {}}
examples = []
examples.append(InputExample(**example1))
examples.append(InputExample(**example2))
features = model_wrapper._convert_examples_to_features(examples)
for input_features in features:
    decoed_str = model_wrapper.tokenizer.decode(input_features.input_ids)
    print(decoed_str.replace(model_wrapper.tokenizer.pad_token, ""))

# Output:
# [CLS] The incident in Mogadishu, the Somali capital, came as U. S. forces began the final phase of their promised March 31 pullout. question : The capital of Somalia is Mogadishu. True or False? answer : [MASK] [SEP]
# [CLS] By the time a case of rabies is confirmed, the disease may have taken hold in the area. question : A case of rabies was confirmed. True or False? answer : [MASK] [SEP]
