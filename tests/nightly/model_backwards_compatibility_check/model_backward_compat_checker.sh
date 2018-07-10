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

#Author: Piyush Ghai

echo "Invoking model_backwards_compat_test.sh script"
echo `pwd`
cd tests/nightly/model_backwards_compatibility_check
echo `pwd`

echo '=========================='
echo 'running mlp with module api'
python mnist_mlp_module_api_inference.py

echo '=========================='
echo 'running lenet with gluon api (non - hybridized)'
python lenet_cnn_gluon_inference.py

echo '=========================='
echo 'running lenet with gluon api (hybridized)'
python lenet_cnn_gluon_hybrid_inference.py

echo '=========================='
echo 'running rnn with gluon - save and load parameters'
python lm_rnn_gluon_inference.py

