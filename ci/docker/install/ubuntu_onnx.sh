#!/usr/bin/env bash

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
set -x

# install libraries for onnx's python package on ubuntu
apt-get install -y libprotobuf-dev protobuf-compiler

pip2 install pytest==3.4.0 pytest-cov==2.5.1 protobuf==3.0.0 Pillow==5.0.0 tabulate==0.7.5
pip3 install pytest==3.4.0 pytest-cov==2.5.1 protobuf==3.0.0 Pillow==5.0.0 tabulate==0.7.5

git clone --recursive https://github.com/onnx/onnx.git
cd onnx
git checkout 7e205b66190f4376c64741ba7705dc23e9fbf225
python setup.py install