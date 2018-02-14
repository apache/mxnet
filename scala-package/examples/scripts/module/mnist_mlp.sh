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

ROOT_DIR=$(cd `dirname $0`/../../..; pwd)
CLASSPATH=$ROOT_DIR/assembly/osx-x86_64-cpu/target/*:$ROOT_DIR/examples/target/*:$ROOT_DIR/examples/target/classes/lib/*

mkdir -p model
java -Xmx4G -cp $CLASSPATH \
	ml.dmlc.mxnetexamples.module.MnistMlp \
  --data-dir "$ROOT_DIR/core/data/" \
  --batch-size 10 \
  --num-epoch 5
