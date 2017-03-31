#!/bin/bash
java -Xmx4G -cp scala-package/assembly/osx-x86_64-cpu/target/*:scala-package/examples/target/*:scala-package/examples/target/classes/lib/* \
  ml.dmlc.mxnetexamples.imclassification.TrainMnist \
  --data-dir=/Users/QYGong/Downloads/mnist/ \
  --num-epochs=10 \
  --network=mlp \
  --cpus=0,1,2,3
