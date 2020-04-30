#!/bin/bash
TASK=$1
DATA_DIR=$2
MODEL_DIR=$3
NUM_EXAMPLES=$4

echo Running iPET with the following parameters:
echo ------------------------------
echo TASK         = "$TASK"
echo DATA_DIR     = "$DATA_DIR"
echo MODEL_DIR    = "$MODEL_DIR"
echo NUM_EXAMPLES = "$NUM_EXAMPLES"
echo ------------------------------

cd ../

if [ $TASK = "agnews" ]; then
  PATTERN_IDS="0 1 2 3 4 5"
elif [ $TASK = "mnli" ]; then
  PATTERN_IDS="0 1 2 3"
elif [ $TASK = "yelp-full" ]; then
  PATTERN_IDS="0 1 2 3"
elif [ $TASK = "yelp-polarity" ]; then
  PATTERN_IDS="0 1 2 3"
elif [ $TASK = "yahoo" ]; then
  PATTERN_IDS="0 1 2 3 4 5"
else
  echo "Task " $TASK " is not supported by this script" 1>&2
  exit 1
fi

ITERS="1 2"
LAST_ITER="2"
FACTOR="5"
MAX_STEPS="1000"

if [ $NUM_EXAMPLES = "10" ]; then
  ITERS="1 2 3"
  LAST_ITER="3"
elif [ $NUM_EXAMPLES = "50" ]; then
  ITERS="1 2"
  LAST_ITER="2"
elif [ $NUM_EXAMPLES = "100" ]; then
  ITERS="1 2"
  LAST_ITER="2"
else
  echo "This script only supports training from 10, 50 or 100 examples" 1>&2
  exit 1
fi

for ITER in $ITERS; do

  if [ $ITER = "1" ]; then
    DIR_SUFFIX=""
    NEXT_DIR_SUFFIX="-i1"
    NEW_EXAMPLES=$((NUM_EXAMPLES * FACTOR - NUM_EXAMPLES))
  elif [ $ITER = "2" ]; then
    DIR_SUFFIX="-i1"
    NEXT_DIR_SUFFIX="-i2"
    NEW_EXAMPLES=$((NUM_EXAMPLES * FACTOR * FACTOR - NUM_EXAMPLES))
  elif [ $ITER = "3" ]; then
    DIR_SUFFIX="-i2"
    NEXT_DIR_SUFFIX="-i3"
    NEW_EXAMPLES=$((NUM_EXAMPLES * FACTOR * FACTOR * FACTOR - NUM_EXAMPLES))
  fi

  echo New examples = "$NEW_EXAMPLES"
  echo Running create_ipet_training_set.py to obtain ${NEW_EXAMPLES} additional training examples...

  python3 create_ipet_training_set.py \
    --logits_dir ${MODEL_DIR}${DIR_SUFFIX}/ \
    --output_dir ${MODEL_DIR}${DIR_SUFFIX}/next-gen-train-sets/ \
    --data_dir ${DATA_DIR} \
    --task_name ${TASK} \
    --lm_train_examples_per_label 10000 \
    --reduction wmean \
    --num_examples ${NEW_EXAMPLES} \
    --logits_percentage 0.25

  echo Running run_training.py to train generation ${ITER}...
  python3 run_training.py \
    --data_dir ${DATA_DIR} \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --overwrite_output_dir \
    --task_name ${TASK} \
    --output_dir ${MODEL_DIR}${NEXT_DIR_SUFFIX}/ \
    --do_train \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --do_eval \
    --per_gpu_train_batch_size 1 \
    --per_gpu_helper_batch_size 3 \
    --lm_training \
    --alpha 0.9999 \
    --gradient_accumulation_steps 4 \
    --test_examples -1 \
    --max_steps ${MAX_STEPS} \
    --train_examples ${NUM_EXAMPLES} \
    --dev_examples 0 \
    --max_seq_length 256 \
    --additional_data_dir ${MODEL_DIR}${DIR_SUFFIX}/next-gen-train-sets/ \
    --wrapper_type mlm \
    --repetitions 3 \
    --save_train_logits \
    --lm_train_examples_per_label 10000 \
    --pattern_ids ${PATTERN_IDS}

  echo Training complete

  if [ $ITER = $LAST_ITER ]; then
    echo Running merge_logits.py to obtain logits for training the final model...
    python3 merge_logits.py \
      --logits_dir ${MODEL_DIR}${NEXT_DIR_SUFFIX}/ \
      --output ${MODEL_DIR}${NEXT_DIR_SUFFIX}/logits-wmean.txt \
      --reduction wmean

    echo Running run_training.py to train the final model...

    python3 run_training.py \
      --repetitions 3 \
      --data_dir ${DATA_DIR} \
      --model_type roberta \
      --model_name_or_path roberta-large \
      --overwrite_output_dir \
      --task_name ${TASK} \
      --output_dir ${MODEL_DIR}${NEXT_DIR_SUFFIX}-distilled/ \
      --do_train \
      --weight_decay 0.01 \
      --learning_rate 1e-5 \
      --per_gpu_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --do_eval \
      --test_examples -1 \
      --max_steps 5000 \
      --dev_examples 0 \
      --train_examples 1 \
      --max_seq_length 256 \
      --temperature 2 \
      --logits_file ${MODEL_DIR}${NEXT_DIR_SUFFIX}/logits-wmean.txt \
      --lm_train_examples_per_label 10000

    echo Training complete
  fi

done
