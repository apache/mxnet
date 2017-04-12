#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*


MODE="all"
OUTPUT_PATH="."
# Just load the trace file at chrome://tracing in your Chrome browser
FILE_NAME="profile_ndarray.json"

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.profiler.ProfilerNDArray \
	--profiler-mode $MODE \
	--output-path $OUTPUT_PATH \
	--profile-filename $FILE_NAME

