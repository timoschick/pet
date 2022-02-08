#!/bin/bash

# based on bingo/esnli_pet
# Compares [mlm/cls] losses on two settings
# (1) mlm
# (2) no mlm (by setting alpha=1)
PET_DIR=/home/yimingz0/src/pet

DATA_DIR=/data/shared/data/e-SNLI/dataset
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=esnli

NUM_TRAIN=100
NUM_TEST=-1
NUM_UNLABEL=30000 


export CUDA_VISIBLE_DEVICES=0

# setting (1)
MSG=mlm
python3 ${PET_DIR}/cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir ${PET_DIR}/outputs/pet_${TASK}_${MSG} \
--do_train \
--do_eval \
--overwrite_output_dir \
--cache_dir ${PET_DIR}/cache/${MODEL}_${TASK}/ \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--unlabeled_examples $NUM_UNLABEL \
--pet_max_steps 1000 \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4 \
--split_examples_evenly \
--no_distillation \
--lm_training

# setting (2)
MSG=no_mlm
python3 ${PET_DIR}/cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir ${PET_DIR}/outputs/pet_${TASK}_${MSG} \
--do_train \
--do_eval \
--overwrite_output_dir \
--cache_dir ${PET_DIR}/cache/${MODEL}_${TASK}/ \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--unlabeled_examples $NUM_UNLABEL \
--pet_max_steps 1000 \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4 \
--split_examples_evenly \
--no_distillation \
--lm_training \
--alpha 1.0