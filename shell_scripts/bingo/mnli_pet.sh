PET_DIR=/data/rosa/pet
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=mnli
NUM_TRAIN=100
NUM_TEST=-1
NUM_UNLABEL=20
MSG=toy_filter_dev

python3 ${PET_DIR}/cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
--data_dir /data/rosa/data/multinli_1.0/ \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir /data/rosa/pet/outputs/${TASK}_${MODEL}_${MSG}_T${NUM_TRAIN}_D${NUM_UNLABEL}_Test${NUM_TEST} \
--do_train \
--do_eval \
--overwrite_output_dir \
--cache_dir /data/rosa/pet/cache/${MODEL}_${TASK}/ \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--unlabeled_examples $NUM_UNLABEL
