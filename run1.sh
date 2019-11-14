#!/bin/bash 

export PYTHONPATH=/home/user/tf-training/models
export GLUE_DIR=/home/user/tf-training/glue_data
export BERT_BASE_DIR=/home/user/tf-training/uncased_L-12_H-768_A-12
export OUTPUT_DIR=/home/user/tf-training/datasets
export TASK_NAME=MRPC
export DATASETS_DIR=/home/user/tf-training/datasets
export MODEL_DIR=/home/user/tf-training/results
export TASK=MRPC

python ./official/bert/run_classifier1.py \
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