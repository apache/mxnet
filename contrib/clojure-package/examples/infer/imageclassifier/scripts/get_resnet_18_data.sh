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

set -evx

MXNET_ROOT=$(cd "$(dirname $0)/.."; pwd)

data_path=$MXNET_ROOT/models/resnet-18/

image_path=$MXNET_ROOT/images/

if [ ! -d "$data_path" ]; then
    mkdir -p "$data_path"
fi

if [ ! -d "$image_path" ]; then
    mkdir -p "$image_path"
fi

if [ ! -f "$data_path/resnet-18-0000.params" ]; then
    wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-symbol.json -P $data_path
    wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-0000.params -P $data_path
    wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/synset.txt -P $data_path
fi

if [ ! -f "$image_path/kitten.jpg" ]; then
    wget https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/resnet152/kitten.jpg -P $image_path
    wget https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg -P $image_path
fi
