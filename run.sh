#!/bin/bash 
python ./official/bert/run_classifier.py \
       0 \
       --mode='train_and_eval' \
       --input_meta_data_path=${DATASETS_DIR}/${TASK}_meta_data \
       --train_data_path=${DATASETS_DIR}/${TASK}_train.tf_record \
       --eval_data_path=${DATASETS_DIR}/${TASK}_eval.tf_record \
       --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
       --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
       --train_batch_size=4 \
       --eval_batch_size=4 \
       --steps_per_loop=1 \
       --learning_rate=2e-5 \
       --num_train_epochs=3 \
       --model_dir=${MODEL_DIR} \
       --strategy_type=multi_worker_mirror