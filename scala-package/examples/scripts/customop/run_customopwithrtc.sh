#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# which gpu card to use
GPU=0

# the mnist data path
# you can get the mnist data using the script core/scripts/get_mnist_data.sh
DATA_PATH=$1

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.customop.ExampleCustomOpWithRtc \
	--data-path $DATA_PATH \
	--gpu $GPU
