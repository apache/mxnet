#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# more details please refer to
# https://github.com/Ldpe2G/mxnet/blob/develop/example/neural-style/end_to_end/README.md
TRAIN_DATA_PATH=$1 
STYLE_IMG=$2 
VGG_MODEL_PATH=$3
SAVE_MODEL_DIR=$4
GPU=0

java -Xmx1024m -cp $CLASS_PATH \
	ml.dmlc.mxnet.examples.neuralstyle.end2end.BoostTrain \
	--data-path $TRAIN_DATA_PATH  \
	--vgg--model-path  $VGG_MODEL_PATH \
	--save--model-path $SAVE_MODEL_DIR \
	--style-image $STYLE_IMG \
	--gpu $GPU