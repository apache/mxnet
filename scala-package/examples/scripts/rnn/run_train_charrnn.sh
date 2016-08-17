#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# which gpu card to use, -1 means cpu
GPU=$1
# you can get the training data file using the following command
# wget http://data.dmlc.ml/mxnet/data/lab_data.zip
# unzip -o lab_data.zip
# for example ./datas/obama.txt
DATA_PATH=$2
# for example ./models
SAVE_MODEL_PATH=$3

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnet.examples.rnn.TrainCharRnn \
	--data-path $DATA_PATH \
	--save-model-path $SAVE_MODEL_PATH \
	--gpu $GPU \
