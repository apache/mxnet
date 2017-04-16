#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

INPUT_IMG=$1 
MODEL_DIR=$2
OUTPUT_DIR=$3
GPU=0

java -Xmx1024m -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.neuralstyle.end2end.BoostInference \
	--model-path $MODEL_DIR \
	--input-image $INPUT_IMG \
	--output-path $OUTPUT_DIR \
	--gpu $GPU