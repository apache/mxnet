#!/bin/bash
ROOT_DIR=$(cd `dirname $0`/../../..; pwd)
CLASSPATH=$ROOT_DIR/assembly/osx-x86_64-cpu/target/*:$ROOT_DIR/examples/target/*:$ROOT_DIR/examples/target/classes/lib/*

mkdir -p model
java -Xmx4G -cp $CLASSPATH \
	ml.dmlc.mxnetexamples.module.MnistMlp \
  --data-dir "$ROOT_DIR/core/data/" \
  --batch-size 10 \
  --num-epoch 5
