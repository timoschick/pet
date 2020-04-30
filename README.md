# Pattern-Exploiting Training (PET)

This repository contains the code for [Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676). The paper introduces pattern-exploiting training (PET), a semi-supervised training
procedure that reformulates input examples as cloze-style phrases and significantly outperforms regular supervised training in low-resource settings. The iterative variant of PET (iPET) trains multiple generations of models and can even be used without any training data.

##### Zero-Shot Learning

|              | Yelp (Full) | AG's News | Yahoo    | MNLI     |
| ------------ | -----------:| ---------:| --------:| --------:|
| unsupervised |        33.8 |      69.5 |     44.0 |     39.1 |
| iPET         |    **56.7** |  **87.5** | **70.7** | **53.6** |

##### 10 Training Examples

|              | Yelp (Full) | AG's News | Yahoo    | MNLI     |
| ------------ | -----------:| ---------:| --------:| --------:|
| supervised   |        21.1 |      25.0 |     10.1 |     34.2 |
| PET          |        52.9 |      87.5 |     63.8 |     41.8 |
| iPET         |    **57.6** |  **89.3** | **70.7** | **43.2** |

##### 100 Training Examples

|              | Yelp (Full) | AG's News | Yahoo    | MNLI     |
| :----------- | -----------:| ---------:| --------:| --------:|
| supervised   |        53.0 |      86.0 |     62.9 |     47.9 |
| PET          |        61.9 |      88.3 |     69.2 |     74.7 |
| iPET         |    **62.9** |  **89.6** | **71.2** | **78.4** |

<sup>*Note*: The current release of PET does not support iterative training. iPET will be added in the next release.</sup>

## 📑 Contents

**[⚙️ Setup](#%EF%B8%8F-setup)**

**[💬 Usage](#-usage)**

**[🐶 Train your own PET](#-train-your-own-pet)**

**[📕 Citation](#-citation)**

## ⚙️ Setup

PET requires `Python>=3.6`, `numpy==1.17`, `jsonpickle==1.1`, `scikit-learn==0.19`, `pytorch==1.4` and `transformers==2.8`. If you use `conda`, you can simply create an environment with all required dependencies from the `environment.yml` file found in the root of this repository. 

## 💬 Usage

The code in this repository currently supports 3 different training modes (supervised, unsupervised and PET) and 4 different tasks (Yelp Reviews, AG's News, Yahoo Questions and MNLI). For details, please refer to [the original paper](https://arxiv.org/abs/2001.07676).

### Supervised Training and Evaluation

To fine-tune a pretrained language model on one of the four tasks with regular supervised training, only `run_training.py` is required. Training and evaluation can be performed as follows:

    python3 run_training.py \
    --wrapper_type sequence_classifier \
    --train_examples TRAIN_EXAMPLES \
    --data_dir DATA_DIR \
    --model_type MODEL_TYPE \
    --model_name_or_path MODEL_NAME \
    --task_name TASK_NAME \
    --output_dir OUTPUT_DIR \
    --do_train \
    --do_eval

where
 - `TRAIN_EXAMPLES` is the number of train examples to use. The script will always distribute the number of examples evenly among all labels. For example, if you specify `TRAIN_EXAMPLES = 100` and the task has 3 different labels, the training set will contain 33, 33 and 34 examples for label 1, 2 and 3, respectively.
 - `DATA_DIR` is the directory containing the train and test csv files (check `tasks.py` to see how these files should be named for each task).
 - `MODEL_TYPE` is either `bert` or `roberta`.
 - `MODEL_NAME` is the name of a pretrained model (e.g., `roberta-large`) or the path to a pretrained model.
 - `TASK_NAME` is one of `yelp-full`, `agnews`, `yahoo` and `mnli`.
 - `OUTPUT_DIR` is the name of the directory in which the trained model and evaluation results are saved.

To reproduce the exact results from the paper, you need to additionally specify `--gradient_accumulation_steps 4 --max_steps 250` when running this script.

### Unsupervised Evaluation

To evaluate a pretrained language model with the default PET patterns and verbalizers, but without fine-tuning, use the following:

    python3 run_training.py \
    --wrapper_type mlm \
    --train_examples TRAIN_EXAMPLES \
    --data_dir DATA_DIR \
    --model_type MODEL_TYPE \ 
    --model_name_or_path MODEL_NAME \
    --task_name TASK_NAME \
    --output_dir OUTPUT_DIR \
    --do_train \
    --do_eval \
    --max_steps 0 \
    --repetitions 1 \
    --pattern_ids PATTERN_IDS

where
 - `TRAIN_EXAMPLES`, `DATA_DIR`, `MODEL_TYPE`, `MODEL_NAME`, `TASK_NAME` and `OUTPUT_DIR` are as in the previous section.
 - `PATTERN_IDS` specifies the PVPs to use. Pattern ID `n` corresponds to `P_(n+1)` in the paper. If you want to use *all* patterns, specify `PATTERN_IDS 0 1 2 3 4` for AG's News and Yahoo Questions or `PATTERN_IDS 0 1 2 3` for Yelp Reviews and MNLI.

### PET Training and Evaluation

PET Training consists of three steps:

 1) training the individual PVP models (see Section *PVP Training and Inference* in the paper)
 2) combining the outputs produced by the individual models (see Section *Combining PVPs* in the paper)
 3) training a new model on the soft labels produced in the previous step (see Section *Combining PVPs* in the paper)
 
#### Training Individual PVP Models

Training individual PVP models is very similar to the unsupervised setting above:

    python3 run_training.py \
    --wrapper_type mlm \
    --train_examples TRAIN_EXAMPLES \
    --data_dir DATA_DIR \
    --model_type MODEL_TYPE \ 
    --model_name_or_path MODEL_NAME \
    --task_name TASK_NAME \
    --output_dir OUTPUT_DIR \
    --do_train \
    --do_eval \
    --pattern_ids PATTERN_IDS \
    --lm_train_examples_per_label 10000 \
    --save_train_logits

where `TRAIN_EXAMPLES`, `DATA_DIR`, `MODEL_TYPE`, `MODEL_NAME`, `TASK_NAME`, `OUTPUT_DIR` and `PATTERN_IDS` are as in the previous section.

For auxiliary language modeling, the parameter `--lm_training` must additionally be set. The number passed to `--lm_train_examples_per_label` specifies the number of unlabelled examples per label to use for language modeling. The labels for these examples are **not** used during training.  

To reproduce the exact results from the paper, you need to additionally specify

    --gradient_accumulation_steps 4 --max_steps 250 
    
for results **without** auxiliary language modeling or

    --gradient_accumulation_steps 4 --max_steps 1000 --per_gpu_train_batch_size 1 --per_gpu_helper_batch_size 3
    
for results **with** auxiliary language modeling.

#### Combining PVPs

For each pattern id `<P>` and repetition `<R>`, the above script creates a folder `p<P>-i<R>` in `OUTPUT_DIR`.
The argument `--save_train_logits` causes the script to create a file `logits.txt` in each of these folders, which contains the logits that the trained model assigns to each example that was also used for language modeling (see above). The logits can be merged using `merge_logits.py` as follows:

    python3 merge_logits.py --logits_dir OUTPUT_DIR --output_file LOGITS_FILE --reduction REDUCTION
    
 where `LOGITS_FILE` is the file to which the merged logits are saved and `REDUCTION` is either `mean` or `wmean`, with `mean` corresponding to the *uniform* variant and `wmean` corresponding to the *weighted* variant described in the paper.
 
#### Training the Final Model

To train the final model based on the newly created logits file, `run_training.py` needs to be called again:

    python3 run_training.py \
    --wrapper_type sequence_classifier \
    --data_dir DATA_DIR \
    --model_type MODEL_TYPE \
    --model_name_or_path MODEL_NAME \ 
    --task_name TASK_NAME \
    --output_dir FINAL_OUTPUT_DIR \
    --do_train \
    --do_eval \
    --max_steps 5000 \
    --train_examples 1 \
    --lm_train_examples_per_label 10000 \
    --temperature 2 \
    --logits_file LOGITS_FILE
    
where `DATA_DIR`, `MODEL_TYPE`, `MODEL_NAME`, `TASK_NAME` and `LOGITS_FILE` are as before and `FINAL_OUTPUT_DIR` is the dir in which the final model and its evaluation result are saved.

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

Importantly, in PET's current version, verbalizers are restricted to **single tokens** in the underlying LMs vocabulary. Given a language model's tokenizer, you can easily check whether a word corresponds to a single token by verifying that `len(tokenizer.tokenize(word)) == 1`.

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

## 📕 Citation

If you make use of the code in this repository, please cite the following paper:

    @article{schick2020exploiting,
      title={Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference},
      author={Timo Schick and Hinrich Schütze},
      journal={Computing Research Repository},
      volume={arXiv:2001.07676},
      url={http://arxiv.org/abs/2001.07676},
      year={2020}
    }

