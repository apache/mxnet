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

set -ex

echo "uploading model files to s3"

echo `pwd`
cd ./tests/nightly/model_backwards_compatibility_check/models/
echo `pwd`

# The directory structure will be as follows :
# <mxnet-version>/<model-files> eg :
# ls /tests/nightly/model_backwards_compatibility_check/models/
# 1.1.0/   1.2.0/   1.2.1/
# we upload these folders to S3 and the inference files understand them and pull of models off them
for dir in $(ls `pwd`/)
do
    echo $dir
    aws s3 cp $dir/ s3://mxnet-ci-prod-backwards-compatibility-models/$dir/ --recursive
done

echo "Deleting model files"
cd ../
rm -rf `pwd`/models
