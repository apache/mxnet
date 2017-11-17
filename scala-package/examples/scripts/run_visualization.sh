#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


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
