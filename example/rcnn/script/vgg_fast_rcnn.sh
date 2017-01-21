#!/usr/bin/env bash

# run this experiment with
# nohup bash script/vgg_fast_rcnn.sh 0,1 &> vgg_fast_rcnn.log &
# to use gpu 0,1 to train, gpu 0 to test and write logs to vgg_fast_rcnn.log
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

python -m rcnn.tools.train_rcnn --proposal selective_search --gpu $1
python -m rcnn.tools.test_rcnn --proposal selective_search --gpu $gpu
