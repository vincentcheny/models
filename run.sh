#!/bin/bash 
python ./official/nlp/bert/run_classifier.py \
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
       --model_dir=hdfs://b10g1:8020/user/root/tftuner \
       --strategy_type=multi_worker_mirror \
       --node_list="[\"b10g4.bigc.dbg.private:2001\",\"b10g6.bigc.dbg.private:2002\"]" \
       --task_index=0
