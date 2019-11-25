#!/bin/bash
if [ "$1" == "local" ]
then
    export PYTHONPATH=/mnt/d/GitRepository/Imagenet/models
    export GLUE_DIR=/mnt/d/GitRepository/Imagenet/glue_data
    export BERT_BASE_DIR=/mnt/d/GitRepository/Imagenet/uncased_L-12_H-768_A-12
    export OUTPUT_DIR=/mnt/d/GitRepository/Imagenet/datasets
    export TASK_NAME=MRPC
    export DATASETS_DIR=/mnt/d/GitRepository/Imagenet/datasets
    export MODEL_DIR=/mnt/d/GitRepository/Imagenet/results
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
