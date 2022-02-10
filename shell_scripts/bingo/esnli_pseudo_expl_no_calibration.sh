#!/bin/bash
# PH baseline for esnli-100 dataset

PET_DIR=/home/yimingz0/src/pet

DATA_DIR=/data/shared/data/e-SNLI/dataset
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=esnli-100

NUM_TRAIN=100
NUM_TEST=-1
NUM_UNLABEL=100 # does not matter since we don't do lm_training 

export CUDA_VISIBLE_DEVICES=0

MSG=pseudo_expl_no_calibration
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
--no_distillation \
--wandb_run_name ${MSG} \
--save_train_logits \
--train_custom_expl_file /data/shared/data/e-SNLI/dataset/esnli_train_1_100_gpt-neo.jsonl \
--dev_custom_expl_file /data/shared/data/e-SNLI/dataset/esnli_dev_gpt-neo.jsonl \
--beta 1.0