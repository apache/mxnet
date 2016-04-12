#!/bin/bash

MXNET_ROOT=/home/ldpe2g/Mxnet/mxnet_scala/mxnet/mxnet
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*
INPUT_IMG=$MXNET_ROOT/example/neural-style/input/IMG_4343.jpg 
STYLE_IMG=$MXNET_ROOT/example/neural-style/input/starry_night.jpg 
MODEL_PATH=$MXNET_ROOT/example/neural-style/model/vgg19.params
OUTPUT_DIR=$MXNET_ROOT/example/neural-style/output

java -Xmx1024m -cp $CLASS_PATH \
	ml.dmlc.mxnet.examples.neuralstyle.NeuralStyle \
	--content-image $INPUT_IMG  \
	--style-image  $STYLE_IMG \
	--model-path  $MODEL_PATH \
	--output-dir $OUTPUT_DIR 
