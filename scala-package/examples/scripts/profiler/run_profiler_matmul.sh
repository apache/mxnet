#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# which gpu card to use, -1 means cpu
GPU=0

MODE="symbolic"
OUTPUT_PATH="."
# Just load the trace file at chrome://tracing in your Chrome browser
FILE_NAME="profile_matmul_20iter.json"

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.profiler.ProfilerMatMul \
	--gpu $GPU \
	--profiler-mode $MODE \
	--output-path $OUTPUT_PATH \
	--profile-filename $FILE_NAME

