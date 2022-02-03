NET_HOME=/net/scratch/zhouy1
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=mnli
NUM_TRAIN=100
NUM_TEST=-1
NUM_UNLABEL=30000
MSG=replicate_with_params

python3 ${NET_HOME}/pet/cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
--data_dir ${NET_HOME}/data/multinli_1.0/ \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir ${NET_HOME}/pet/outputs/pet_${TASK}_${MODEL}_${MSG}_T${NUM_TRAIN}_D${NUM_UNLABEL}_Test${NUM_TEST} \
--do_train \
--do_eval \
--overwrite_output_dir \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--unlabeled_examples $NUM_UNLABEL \
--sc_max_steps 5000 \
--lm_training \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--sc_gradient_accumulation_steps 4 \
--split_examples_evenly
