#!/bin/bash
ROOT_DIR=$(cd `dirname $0`/../../..; pwd)
CLASSPATH=$ROOT_DIR/assembly/linux-x86_64-cpu/target/*:$ROOT_DIR/examples/target/*:$ROOT_DIR/examples/target/classes/lib/*

DATA_DIR=$ROOT_DIR/core/data

SAVE_MODEL_PATH=.

# LOAD_MODEL=seqModule-0001.params

java -Xmx4G -cp $CLASSPATH \
            ml.dmlc.mxnetexamples.module.SequentialModuleEx \
            --data-dir $DATA_DIR \
            --batch-size 10 \
            --num-epoch 2 \
            --lr 0.01 \
            --save-model-path $SAVE_MODEL_PATH \
            # --load-model-path $LOAD_MODEL
