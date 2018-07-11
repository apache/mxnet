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

run_models() {
	echo '=========================='
	echo "Running training files and preparing models"
	echo '=========================='
	python mnist_mlp_module_api_train.py
	echo '=========================='
	python lenet_cnn_gluon_hybrid_train.py
	echo '=========================='
	python lm_rnn_gluon_train.py
	echo '=========================='
	python lenet_cnn_gluon_train.py
	echo '=========================='
}

install_mxnet() {
	version=$1
	echo "Installing MXNet "$version
	pip install mxnet==$version
}

install_boto3(){
	echo "Intalling boto3"
	pip install boto3
}

echo `pwd`
cd tests/nightly/model_backwards_compatibility_check
echo `pwd`

install_boto3

install_mxnet 1.1.0
run_models

install_mxnet 1.2.0
run_models