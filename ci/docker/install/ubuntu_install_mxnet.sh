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

set -ex

YEAR=$(date +"%Y")
MONTH=$(date +"%m")
DATE=$(date +"%d")

pip3 install https://apache-mxnet.s3-us-west-2.amazonaws.com/dist/"${YEAR}"-"${MONTH}"-"${DATE}"/dist/mxnet_cu101mkl-1.6.0b"${YEAR}""${MONTH}""${DATE}"-py2.py3-none-manylinux1_x86_64.whl
