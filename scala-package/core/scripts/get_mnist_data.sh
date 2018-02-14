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


set -e

if [ ! -z "$MXNET_DATA_DIR" ]; then
  data_path="$MXNET_DATA_DIR"
else
  data_path="./data"
fi

if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

mnist_data_path="$data_path/mnist.zip"
if [ ! -f "$mnist_data_path" ]; then
  wget http://data.mxnet.io/mxnet/data/mnist.zip -P $data_path
  cd $data_path
  unzip -u mnist.zip
fi
