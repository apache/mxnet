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

if [ ! -z "$MXNET_HOME" ]; then
  data_path="$MXNET_HOME/data"
else
  MXNET_ROOT=$(cd "$(dirname $0)/../.."; pwd)
  data_path="$MXNET_ROOT/data"
fi

if [ ! -d "$data_path" ]; then
    mkdir -p "$data_path"
fi

resnet50_ssd_data_path="$data_path/resnet50_ssd"
if [ ! -f "$resnet50_ssd_data_path/resnet50_ssd_model-0000.params" ]; then
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json -P $resnet50_ssd_data_path
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params -P $resnet50_ssd_data_path
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt -P $resnet50_ssd_data_path
fi
