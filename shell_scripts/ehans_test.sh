NET_HOME=/net/scratch/zhouy1

python3 ${NET_HOME}/pet/cli.py \
--method pet \
--pattern_ids 0 1 2 3 \
--data_dir ${NET_HOME}/data/split_abundant_words_templates/seed0/partition0 \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name ehans \
--output_dir ${NET_HOME}/pet/outputs/ehans_10 \
--do_train \
--do_eval \
--overwrite_output_dir \
--train_examples 10 \
# --test_examples 10 \
# --unlabeled_examples 100 \
# --pet_repetitions 1 \
# --sc_repetitions 1
