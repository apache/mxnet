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
#
# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

# This script requires that APACHE_PASSWORD and APACHE_USERNAME are set
# environment variables. Also, artifacts must be previously uploaded to S3
# in the MXNet public bucket (mxnet-public.s3.us-east-2.amazonaws.com).

set -ex

api_list=("cpp" "clojure" "java" "julia" "python" "r" "scala")
version=v1.5.0
for i in "${api_list[@]}"
do
    tar cvf $i-artifacts.tgz $i && aws s3 cp $i-artifacts.tgz s3://mxnet-public/docs/$version/$i-artifacts.tgz
done
