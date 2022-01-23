NET_HOME=/net/scratch/zhouy1

python3 ${NET_HOME}/pet/cli.py \
--method pet \
--pattern_ids 0 1 2 3 4 \
--data_dir ${NET_HOME}/data/ag_news_csv/ \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name agnews \
--output_dir ${NET_HOME}/pet/outputs/bert_agnews_save_epoch_few_shot50 \
--do_train \
--do_eval \
--overwrite_output_dir \
--train_examples 50 \
--test_examples 500 \
--unlabeled_examples 1000 \
# --pet_repetitions 1 \
# --sc_repetitions 1
