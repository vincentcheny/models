#!/bin/bash
if [ "$1" == "local" ]
then
    export PYTHONPATH=/mnt/d/GitRepository/BERT/models
    export GLUE_DIR=/mnt/d/GitRepository/BERT/glue_data
    export BERT_BASE_DIR=/mnt/d/GitRepository/BERT/uncased_L-12_H-768_A-12
    export OUTPUT_DIR=/mnt/d/GitRepository/BERT/datasets
    export TASK_NAME=MRPC
    export DATASETS_DIR=/mnt/d/GitRepository/BERT/datasets
    export MODEL_DIR=/mnt/d/GitRepository/BERT/results
    export TASK=MRPC
elif [ "$1" == "server" ]
then
    export PYTHONPATH=/home/chen.yu/models
    export GLUE_DIR=/home/chen.yu/glue_data
    export BERT_BASE_DIR=/home/chen.yu/uncased_L-12_H-768_A-12
    export OUTPUT_DIR=/home/chen.yu/datasets
    export TASK_NAME=MRPC
    export DATASETS_DIR=/home/chen.yu/datasets
    export MODEL_DIR=/home/chen.yu/results
    export TASK=MRPC
else
    echo "Usage: . addExportPath.sh [local|server]"
fi
