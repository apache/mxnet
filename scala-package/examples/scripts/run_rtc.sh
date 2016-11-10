#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

GPU=0

java -Xmx1024m -cp $CLASS_PATH \
	ml.dmlc.mxnet.examples.rtc.MxRtc \
	--gpu $GPU
