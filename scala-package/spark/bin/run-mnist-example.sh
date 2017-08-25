#!/bin/bash
CURR_DIR=$(cd `dirname $0`; pwd)
MODULE_DIR=$(cd $CURR_DIR/../; pwd)
ROOT_DIR=$(cd $CURR_DIR/../../; pwd)


LIB_DIR=${MODULE_DIR}/target/classes/lib
JAR=${MODULE_DIR}/target/mxnet-spark_2.10-0.1.2-SNAPSHOT.jar

LIBS=${ROOT_DIR}/assembly/linux-x86_64-cpu/target/mxnet-full_2.10-linux-x86_64-cpu-0.1.2-SNAPSHOT.jar
LIBS="${LIBS},${LIB_DIR}/args4j-2.0.29.jar,${LIB_DIR}/scala-library-2.10.4.jar,${JAR}"

SPARK_OPTS+=" --name mxnet"
SPARK_OPTS+=" --driver-memory 1g"
SPARK_OPTS+=" --executor-memory 1g"
SPARK_OPTS+=" --num-executors 2"
SPARK_OPTS+=" --executor-cores 1"
SPARK_OPTS+=" --jars ${LIBS}"

# You can download these two files as training & validation set.
# They were converted from the MNIST dataset,
# in which each sample was simply flatterned to an array of floats.
# https://s3-us-west-2.amazonaws.com/mxnet.liuyz/data/mnist/train.txt
# https://s3-us-west-2.amazonaws.com/mxnet.liuyz/data/mnist/val.txt

# running opts
RUN_OPTS+=" --input ${INPUT_TRAIN}"
RUN_OPTS+=" --input-val ${INPUT_VAL}"
RUN_OPTS+=" --output ${OUTPUT}"
# These jars are required by the KVStores at runtime.
# They will be uploaded and distributed to each node automatically.
RUN_OPTS+=" --jars ${LIBS}"
RUN_OPTS+=" --num-server 1"
RUN_OPTS+=" --num-worker 2"
RUN_OPTS+=" --java /usr/local/jdk1.8.0_60/bin/java"
RUN_OPTS+=" --model mlp"
RUN_OPTS+=" --cpus 0,1"
RUN_OPTS+=" --num-epoch 5"

${SPARK_HOME}/bin/spark-submit --master spark://localhost:7077 \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.speculation=false \
  --class ml.dmlc.mxnet.spark.example.ClassificationExample \
  ${SPARK_OPTS} \
  ${JAR} \
  ${RUN_OPTS}
