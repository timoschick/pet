PET_DIR=/data/rosa/pet

DATA_DIR=/data/rosa/data/e-SNLI/dataset/
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=esnli

NUM_TRAIN=100
NUM_TEST=-1
NUM_UNLABEL=30000 
MSG=e_always

python3 ${PET_DIR}/cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir ${PET_DIR}/outputs/pet_${TASK}_${MODEL}_${MSG}_T${NUM_TRAIN}_D${NUM_UNLABEL}_Test${NUM_TEST} \
--do_train \
--do_eval \
--overwrite_output_dir \
--cache_dir ${PET_DIR}/cache/${MODEL}_${TASK}/ \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--unlabeled_examples $NUM_UNLABEL \
--sc_max_steps 5000 \
--pet_max_steps 1000 \
--lm_training \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--sc_gradient_accumulation_steps 4 \
--pet_gradient_accumulation_steps 4 \
--split_examples_evenly \
--no_distillation \
--e_pet_pred \
--e_pet_lm \
--e_pet_test 
