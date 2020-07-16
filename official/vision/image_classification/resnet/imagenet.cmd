#!/bin/bash
#SBATCH --job-name=nni-img
#SBATCH --mail-type=FAIL #NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=cy0906@163.com
#SBATCH --output=/lustre/project/EricLo/chen.yu/resnet-imagenet-1gpu-nni.log
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
export PYTHONPATH=$PYTHONPATH:/lustre/project/EricLo/chen.yu/models
DATA_DIR=/lustre/project/EricLo/cx/imagenet/tf_records
# python ./models/official/vision/image_classification/classifier_trainer.py \
#   --mode=train_and_eval \
#   --model_type=resnet \
#   --dataset=imagenet \
#   --model_dir=$MODEL_DIR \
#   --data_dir=$DATA_DIR \
#   --config_file=./models/official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml \
#   --params_override='runtime.num_gpus=1'


# NCCL_DEBUG=WARN 
nnictl create --config /lustre/project/EricLo/chen.yu/models/official/vision/image_classification/resnet/config.yml
# ./ngrok http 8080
# python ./models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py --data_dir=$DATA_DIR --num_gpus=1 --log_steps=1000 --train_epochs=1 --train_steps=4000 #--distribution_strategy=mirrored
