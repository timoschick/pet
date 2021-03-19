# Pattern-Exploiting Training (PET)

This repository contains the code for [Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676) and [It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners](https://arxiv.org/abs/2009.07118). The papers introduce pattern-exploiting training (PET), a semi-supervised training procedure that reformulates input examples as cloze-style phrases. In low-resource settings, PET and iPET significantly outperform regular supervised training, various semi-supervised baselines and even GPT-3 despite requiring 99.9% less parameters. The iterative variant of PET (iPET) trains multiple generations of models and can even be used without any training data.

<table>
    <tr>
        <th>#Examples</th>
        <th>Training Mode</th>
        <th>Yelp (Full)</th>
        <th>AG's News</th>
        <th>Yahoo Questions</th>
        <th>MNLI</th>
    </tr>
    <tr>
        <td rowspan="2" align="center"><b>0</b></td>
        <td>unsupervised</td>
        <td align="right">33.8</td>
        <td align="right">69.5</td>
        <td align="right">44.0</td>
        <td align="right">39.1</td>
    </tr>
    <tr>
        <td>iPET</td>
        <td align="right"><b>56.7</b></td>
        <td align="right"><b>87.5</b></td>
        <td align="right"><b>70.7</b></td>
        <td align="right"><b>53.6</b></td>
    </tr>
    <tr>
        <td rowspan="3" align="center"><b>100</b></td>
        <td>supervised</td>
        <td align="right">53.0</td>
        <td align="right">86.0</td>
        <td align="right">62.9</td>
        <td align="right">47.9</td>
    </tr>
    <tr>
        <td>PET</td>
        <td align="right">61.9</td>
        <td align="right">88.3</td>
        <td align="right">69.2</td>
        <td align="right">74.7</td>
    </tr>
    <tr>
        <td>iPET</td>
        <td align="right"><b>62.9</b></td>
        <td align="right"><b>89.6</b></td>
        <td align="right"><b>71.2</b></td>
        <td align="right"><b>78.4</b></td>
    </tr>
</table>
    
<sup>*Note*: To exactly reproduce the above results, make sure to use v1.1.0 (`--branch v1.1.0`).</sup>

## 📑 Contents

**[🔧 Setup](#-setup)**

**[💬 CLI Usage](#-cli-usage)**

**[💻 API Usage](#-api-usage)**

**[🐶 Train your own PET](#-train-your-own-pet)**

**[📕 Citation](#-citation)**

## 🔧 Setup

All requirements for PET can be found in `requirements.txt`. You can install all required packages with `pip install -r requirements.txt`.

## 💬 CLI Usage

The command line interface `cli.py` in this repository currently supports three different training modes (PET, iPET, supervised training), two additional evaluation methods (unsupervised and priming) and 13 different tasks. For Yelp Reviews, AG's News, Yahoo Questions, MNLI and X-Stance, see [the original paper](https://arxiv.org/abs/2001.07676) for further details. For the 8 SuperGLUE tasks, see [this paper](https://arxiv.org/abs/2009.07118).

### PET Training and Evaluation

To train and evaluate a PET model for one of the supported tasks, simply run the following command:

    python3 cli.py \
	--method pet \
	--pattern_ids $PATTERN_IDS \
	--data_dir $DATA_DIR \
	--model_type $MODEL_TYPE \
	--model_name_or_path $MODEL_NAME_OR_PATH \
	--task_name $TASK \
	--output_dir $OUTPUT_DIR \
	--do_train \
	--do_eval
    
 where
 - `$PATTERN_IDS` specifies the PVPs to use. For example, if you want to use *all* patterns, specify `PATTERN_IDS 0 1 2 3 4` for AG's News and Yahoo Questions or `PATTERN_IDS 0 1 2 3` for Yelp Reviews and MNLI.
 - `$DATA_DIR` is the directory containing the train and test files (check `tasks.py` to see how these files should be named and formatted for each task).
 - `$MODEL_TYPE` is the name of the model being used, e.g. `albert`, `bert` or `roberta`.
 - `$MODEL_NAME` is the name of a pretrained model (e.g., `roberta-large` or `albert-xxlarge-v2`) or the path to a pretrained model.
 - `$TASK_NAME` is the name of the task to train and evaluate on.
 - `$OUTPUT_DIR` is the name of the directory in which the trained model and evaluation results are saved.
 
You can additionally specify various training parameters for both the ensemble of PET models corresponding to individual PVPs (prefix `--pet_`) and for the final sequence classification model (prefix `--sc_`). For example, the default parameters used for our SuperGLUE evaluation are:
 
 	--pet_per_gpu_eval_batch_size 8 \
	--pet_per_gpu_train_batch_size 2 \
	--pet_gradient_accumulation_steps 8 \
	--pet_max_steps 250 \
	--pet_max_seq_length 256 \
    --pet_repetitions 3 \
	--sc_per_gpu_train_batch_size 2 \
	--sc_per_gpu_unlabeled_batch_size 2 \
	--sc_gradient_accumulation_steps 8 \
	--sc_max_steps 5000 \
	--sc_max_seq_length 256 \
    --sc_repetitions 1
    
For each pattern `$P` and repetition `$I`, running the above command creates a directory `$OUTPUT_DIR/p$P-i$I` that contains the following files:
  - `pytorch_model.bin`: the finetuned model, possibly along with some model-specific files (e.g, `spiece.model`, `special_tokens_map.json`)
  - `wrapper_config.json`: the configuration of the model being used
  - `train_config.json`: the configuration used for training
  - `eval_config.json`: the configuration used for evaluation
  - `logits.txt`: the model's predictions on the unlabeled data
  - `eval_logits.txt`: the model's prediction on the evaluation data
  - `results.json`: a json file containing results such as the model's final accuracy
  - `predictions.jsonl`: a prediction file for the evaluation set in the SuperGlue format
  
The final (distilled) model for each repetition `$I` can be found in `$OUTPUT_DIR/final/p0-i$I`, which contains the same files as described above.

🚨 If your GPU runs out of memory during training, you can try decreasing both the `pet_per_gpu_train_batch_size` and the `sc_per_gpu_unlabeled_batch_size` while increasing both `pet_gradient_accumulation_steps` and `sc_gradient_accumulation_steps`.


### iPET Training and Evaluation

To train and evaluate an iPET model for one of the supported tasks, simply run the same command as above, but replace `--method pet` with `--method ipet`. There are various additional iPET parameters that you can modify; all of them are prefixed with `--ipet_`.

For each generation `$G`, pattern `$P` and iteration `$I`, this creates a directory `$OUTPUT_DIR/g$G/p$P-i$I` that is structured as for regular PET. The final (distilled) model can again be found in `$OUTPUT_DIR/final/p0-i$I`.

🚨 If you use iPET with zero training examples, you need to specify how many examples for each label should be chosen in the first generation and you need to change the reduction strategy to mean: `--ipet_n_most_likely 100 --reduction mean`.

### Supervised Training and Evaluation

To train and evaluate a regular sequence classifier in a supervised fashion, simply run the same command as above, but replace `--method pet` with `--method sequence_classifier`. There are various additional parameters for the sequence classifier that you can modify; all of them are prefixed with `--sc_`.

### Unsupervised Evaluation

To evaluate a pretrained language model with the default PET patterns and verbalizers, but without fine-tuning, remove the argument `--do_train` and add `--no_distillation` so that no final distillation is performed.

### Priming

If you want to use priming, remove the argument `--do_train` and add the arguments `--priming --no_distillation` so that all training examples are used for priming and no final distillation is performed. 

🚨 Remember that you may need to increase the maximum sequence length to a much larger value, e.g. `--pet_max_seq_length 5000`. This only works with language models that support such long sequences, e.g. XLNet. For using XLNet, you can specify `--model_type xlnet --model_name_or_path xlnet-large-cased --wrapper_type plm`.

## 💻 API Usage

Instead of using the command line interface, you can also directly use the PET API, most of which is defined in `pet.modeling`. By including `import pet`, you can access methods such as `train_pet`, `train_ipet` and `train_classifier`. Check out their documentation for more information.

## 🐶 Train your own PET

To use PET for custom tasks, you need to define two things: 

- a **DataProcessor**, responsible for loading training and test data. See `examples/custom_task_processor.py` for an example.
- a **PVP**, responsible for applying patterns to inputs and mapping labels to natural language verbalizations. See `examples/custom_task_pvp.py` for an example.

After having implemented the DataProcessor and the PVP, you can train a PET model using the command line as [described above](#pet-training-and-evaluation). Below, you can find additional information on how to define the two components of a PVP, *verbalizers* and *patterns*.

### Verbalizers

Verbalizers are used to map task labels to words in natural language. For example, in a binary sentiment classification task, you could map the positive label (`+1`) to the word `good` and the negative label (`-1`) to the word `bad`. Verbalizers are realized through a PVP's `verbalize()` method. The simplest way of defining a verbalizer is to use a dictionary:

```python
VERBALIZER = {"+1": ["good"], "-1": ["bad"]}
    
def verbalize(self, label) -> List[str]:
    return self.VERBALIZER[label]       
```

Importantly, in PET's current version, verbalizers are by default restricted to **single tokens** in the underlying LMs vocabulary (for using more than one token, [see below](#pet-with-multiple-masks)). Given a language model's tokenizer, you can easily check whether a word corresponds to a single token by verifying that `len(tokenizer.tokenize(word)) == 1`.

You can also define multiple verbalizations for a single label. For example, if you are unsure which words best represent the labels in a binary sentiment classification task, you could define your verbalizer as follows:

```python
VERBALIZER = {"+1": ["great", "good", "wonderful", "perfect"], "-1": ["bad", "terrible", "horrible"]}
```

### Patterns

Patterns are used to make the language model understand a given task; they must contain exactly one `<MASK>` token which is to be filled using the verbalizer. For binary sentiment classification based on a review's summary (`<A>`) and body (`<B>`), a suitable pattern may be `<A>. <B>. Overall, it was <MASK>.` Patterns are realized through a PVP's `get_parts()` method, which returns a pair of text sequences (where each sequence is represented by a list of strings):

```python
def get_parts(self, example: InputExample):
    return [example.text_a, '.', example.text_b, '.'], ['Overall, it was ', self.mask]
```

If you do not want to use a pair of sequences, you can simply leave the second sequence empty:

```python
def get_parts(self, example: InputExample):
    return [example.text_a, '.', example.text_b, '. Overall, it was ', self.mask], []
```
            
If you want to define several patterns, simply use the `PVP`s `pattern_id` attribute:

```python
def get_parts(self, example: InputExample):
    if self.pattern_id == 1:
        return [example.text_a, '.', example.text_b, '.'], ['Overall, it was ', self.mask]
    elif self.pattern_id == 2:
        return ['It was just ', self.mask, '!', example.text_a, '.', example.text_b, '.'], []
```

When training the model using the command line, specify all patterns to be used (e.g., `--pattern_ids 1 2`).

Importantly, if a sequence is longer than the specified maximum sequence length of the underlying LM, PET must know which parts of the input can be shortened and which ones cannot (for example, the mask token must always be there). Therefore, `PVP` provides a `shortenable()` method to indicate that a piece of text can be shortened:

```python
def get_parts(self, example: InputExample):
    text_a = self.shortenable(example.text_a)
    text_b = self.shortenable(example.text_b)
    return [text_a, '.', text_b, '. Overall, it was ', self.mask], []
```

### PET with Multiple Masks

By default, the current implementation of PET and iPET only supports a fixed set of labels that is shared across all examples and verbalizers that correspond to a single token. 
However, for some tasks it may be necessary to use verbalizers that correspond to multiple tokens ([as described here](http://arxiv.org/abs/2009.07118)).
To do so, you simply need the following two modifications:

1) Add the following lines in your task's **DataProcessor** (see `examples/custom_task_processor.py`):
 
   ```python
   from pet.tasks import TASK_HELPERS
   from pet.task_helpers import MultiMaskTaskHelper
   TASK_HELPERS['my_task'] = MultiMaskTaskHelper
   ```
   where ```'my_task'``` is the name of your task. 

2) In your **PVP**, make sure that the ``get_parts()`` method always inserts **the maximum number of mask tokens** required for any verbalization. For example, if your verbalizer maps ``+1`` to "really awesome" and ``-1`` to "terrible" and if those are tokenized as ``["really", "awe", "##some"]`` and ``["terrible"]``, respectively, your ``get_parts()`` method should always return a sequence that contains exactly 3 mask tokens.

With this modification, you can now use verbalizers consisting of multiple tokens:
```python
VERBALIZER = {"+1": ["really good"], "-1": ["just bad"]}
```
However, there are several limitations to consider:

- When using a ``MultiMaskTaskHelper``, the maximum batch size for evaluation is 1.
- As using multiple masks requires multiple forward passes during evaluation, the time required for evaluation scales about linearly with the length of the longest verbalizer. If you require verbalizers that consist of 10 or more tokens, [using a generative LM](https://arxiv.org/abs/2012.11926) might be a better approach.
- The ``MultiMaskTaskHelper`` class is an experimental feature that is not thoroughly tested. In particular, this feature has only been tested for PET and not for iPET. If you observe something strange, please raise an issue.

For more flexibility, you can also write a custom `TaskHelper`. As a starting point, you can check out the classes `CopaTaskHelper`, `WscTaskHelper` and `RecordTaskHelper` in `pet/task_helpers.py`.

## 📕 Citation

If you make use of the code in this repository, please cite the following papers:

    @article{schick2020exploiting,
      title={Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2001.07676},
      url={http://arxiv.org/abs/2001.07676},
      year={2020}
    }

    @article{schick2020small,
      title={It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2009.07118},
      url={http://arxiv.org/abs/2009.07118},
      year={2020}
    }
