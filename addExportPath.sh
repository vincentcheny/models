if [ "$1" == "local" ]
then
    export PYTHONPATH=/mnt/d/GitRepository/myBERT/models
    export GLUE_DIR=/mnt/d/GitRepository/myBERT/glue_data
    export BERT_BASE_DIR=/mnt/d/GitRepository/myBERT/uncased_L-12_H-768_A-12
    export OUTPUT_DIR=/mnt/d/GitRepository/myBERT/datasets
    export TASK_NAME=MRPC
    export DATASETS_DIR=/mnt/d/GitRepository/myBERT/datasets
    export MODEL_DIR=/mnt/d/GitRepository/myBERT/results
    export TASK=MRPC
elif [ "$1" == "server" ]
then
    export PYTHONPATH=/home/user/tf-training/models
    export GLUE_DIR=/home/user/tf-training/glue_data
    export BERT_BASE_DIR=/home/user/tf-training/uncased_L-12_H-768_A-12
    export OUTPUT_DIR=/home/user/tf-training/datasets
    export TASK_NAME=MRPC
    export DATASETS_DIR=/home/user/tf-training/datasets
    export MODEL_DIR=/home/user/tf-training/results
    export TASK=MRPC
else
    echo "Usage: . addExportPath.sh [local|server]"
fi