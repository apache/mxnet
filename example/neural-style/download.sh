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


mkdir -p model
cd model
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params
cd ..

mkdir -p input
cd input
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/input/IMG_4343.jpg
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/input/starry_night.jpg
cd ..

mkdir -p output
