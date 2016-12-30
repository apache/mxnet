#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
export MXNET_CPU_WORKER_NTHREADS=2

# which gpu card to use, -1 means cpu
GPU=$1

# the mnist data path
# you can get the mnist data using the script core/scripts/get_mnist_data.sh
DATA_PATH=$2

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnet.examples.customop.ExampleCustomOp \
	--data-path $DATA_PATH \
	--gpu $GPU \
