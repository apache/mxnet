#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# you can get the training data file using the following command
# wget http://data.dmlc.ml/mxnet/data/lab_data.zip
# unzip -o lab_data.zip
# for example ./datas/obama.txt
DATA_PATH=$1
# for example ./models/obama
MODEL_PREFIX=$2
# feel free to change the starter sentence
STARTER_SENTENCE="The joke"

java -Xmx4G -cp $CLASS_PATH \
	ml.dmlc.mxnet.examples.rnn.TestCharRnn \
	--data-path $DATA_PATH \
	--model-prefix $MODEL_PREFIX \
	--starter-sentence "$STARTER_SENTENCE"
