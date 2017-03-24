#!/bin/bash
export PS_VERBOSE=1
#tools/launch.py -n 2 --launch local \
ps-lite/tracker/dmlc_local.py -n 2 -s 2 \
  java -Xmx3G -Djava.library.path=/home/intel/mxnet/scala-package/native/linux-x86_64-cpu/target/ \
  -cp scala-package/assembly/linux-x86_64-cpu/target/*:scala-package/examples/target/*:scala-package/examples/target/classes/lib/* \
  ml.dmlc.mxnet.examples.imclassification.TrainMnist \
  --data-dir=/home/intel/Downloads/mnist/ \
  --num-epochs=10 \
  --network=mlp \
  --cpus=0 \
  --kv-store=dist_sync \
#  --num-worker=2 \
#  --num-server=2 \
#  --scheduler-host=10.239.12.123 \
#  --scheduler-port=9099
