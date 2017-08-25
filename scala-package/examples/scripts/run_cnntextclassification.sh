#!/bin/bash

MXNET_ROOT=$(cd "$(dirname $0)/../../.."; pwd)
CLASS_PATH=$MXNET_ROOT/scala-package/assembly/linux-x86_64-gpu/target/*:$MXNET_ROOT/scala-package/examples/target/*:$MXNET_ROOT/scala-package/examples/target/classes/lib/*

# which gpu card to use, -1 means cpu
GPU=$1
# the mr dataset path, you should put the pos and neg file in the same folder
MR_DATASET_PATH=$2  
# the trained word2vec file path, binary or text format
W2V_FILE_PATH=$3  
# whether the format of the word2vec file is binary,1 means binary, 0 means text
W2V_FORMAT_BIN=$4 
BATCH_SIZE=$5
SAVE_MODEL_PATH=$6

java -Xmx8G -cp $CLASS_PATH \
	ml.dmlc.mxnetexamples.cnntextclassification.CNNTextClassification \
	--gpu $GPU \
	--mr-dataset-path $MR_DATASET_PATH \
	--w2v-file-path $W2V_FILE_PATH \
	--w2v-format-bin $W2V_FORMAT_BIN \
	--batch-size $BATCH_SIZE \
	--save-model-path $SAVE_MODEL_PATH
