#!/bin/bash
export PS_VERBOSE=1
tools/launch.py -n 2 --launcher local \
  java -Xmx3G -Djava.library.path=scala-package/native/osx-x86_64-cpu/target/ \
  -cp scala-package/assembly/osx-x86_64-cpu/target/*:scala-package/examples/target/*:scala-package/examples/target/classes/lib/* \
  ml.dmlc.mxnetexamples.imclassification.TrainMnist \
  --data-dir=/Users/QYGong/Downloads/mnist/ \
  --num-epochs=10 \
  --network=mlp \
  --cpus=0 \
  --kv-store=dist_sync \
#  --num-worker=2 \
#  --num-server=2 \
#  --scheduler-host=10.239.12.123 \
#  --scheduler-port=9099
