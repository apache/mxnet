#!/usr/bin/env bash

# run this experiment with
# nohup bash script/vgg_voc07.sh 0,1 &> vgg_voc07.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to vgg_voc07.log
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python train_end2end.py --gpu $1
python test.py --gpu $gpu
