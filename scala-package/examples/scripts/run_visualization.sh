#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-cpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# please install the Graphviz library
# if you are using ubuntu, use the following command:
# sudo apt-get install graphviz

# path to save the generated visualization result
OUT_DIR=$1
# net to visualze, e.g. "LeNet", "AlexNet", "VGG", "GoogleNet", "Inception_BN", "Inception_V3", "ResNet_Small"
NET=$2

java -Xmx1024m -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.visualization.ExampleVis \
	--out-dir $OUT_DIR  \
	--net $NET 
