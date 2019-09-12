#!/bin/bash 
python ./official/bert/create_finetuning_data.py --input_data_dir=${GLUE_DIR}/${TASK_NAME}/  --vocab_file=${BERT_BASE_DIR}/vocab.txt  --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record  --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record  --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data  --fine_tuning_task_type=classification --max_seq_length=128  --classification_task_name=${TASK_NAME}