DATA_DIR=/data/shared/data/e-SNLI/dataset/
MODEL_TYPE=roberta
MODEL=roberta-large
TASK=esnli

NUM_TRAIN=100
NUM_TEST=-1
MSG=snli_roberta

METHOD=sequence_classifier
SC_STEPS=$1
SC_TRAIN_BATCH=$2
SC_EVAL_BATCH=$3

export CUDA_VISIBLE_DEVICES=1

python3 cli.py \
--method $METHOD \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL \
--task_name $TASK \
--output_dir outputs/${METHOD}_${TASK}_${MODEL}_${MSG}_T${NUM_TRAIN}_Test${NUM_TEST}_steps${SC_STEPS}_trainbs${SC_TRAIN_BATCH}_testbs${SC_EVAL_BATCH}  \
--do_train \
--do_eval \
--overwrite_output_dir \
--cache_dir cache/${MODEL}_${TASK}/ \
--train_examples $NUM_TRAIN \
--test_examples $NUM_TEST \
--sc_max_steps $SC_STEPS \
--sc_per_gpu_train_batch_size $SC_TRAIN_BATCH \
--sc_per_gpu_eval_batch_size $SC_EVAL_BATCH \
--sc_gradient_accumulation_steps 1 \
--split_examples_evenly \
--wandb_run_name snli_roberta_steps${SC_STEPS}_trainbs${SC_TRAIN_BATCH}_testbs${SC_EVAL_BATCH} \
--sc_eval_steps 100 \
--sc_eval_during_train