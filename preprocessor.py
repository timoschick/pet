from abc import ABC, abstractmethod

from utils import InputFeatures, InputExample

from pvp import AgnewsPVP, MnliPVP, YelpPolarityPVP, YelpFullPVP, \
    YahooPVP, PVP, XStancePVP

PVPS = {
    'agnews': AgnewsPVP,
    'mnli': MnliPVP,
    'yelp-polarity': YelpPolarityPVP,
    'yelp-full': YelpFullPVP,
    'yahoo': YahooPVP,
    'xstance': XStancePVP,
    'xstance-de': XStancePVP,
    'xstance-fr': XStancePVP,
}


class Preprocessor(ABC):

    def __init__(self, wrapper, task_name, pattern_id: int = 0, verbalizer_file: str = None):
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id, verbalizer_file)  # type: PVP
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, **kwargs) -> InputFeatures:
        pass


class MLMPreprocessor(Preprocessor):

    def get_input_features(self, example: InputExample, labelled: bool, **kwargs) -> InputFeatures:
        input_ids, token_type_ids = self.pvp.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label]
        logits = example.logits if example.logits else [-1]

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits)


class SequenceClassifierPreprocessor(Preprocessor):

    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
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

        label = self.label_map[example.label]
        logits = example.logits if example.logits else [-1]

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits)
