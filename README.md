# Pattern-Exploiting Training (PET)

This repository contains the code for [Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676).

## Usage

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

