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

#
# Execute command outside a docker container
#
# Usage: ci_deploy_doc.sh <PR_ID> <BUILD_ID>
#
# PR_ID: the PR number
#
# BUILD_ID: the current build ID for the specified PR
#
set -ex

aws s3 sync --delete docs/_build/html/ s3://mxnet-ci-doc/$1/$2 \
    && echo "Doc is hosted at http://mxnet-ci-doc.s3-accelerate.dualstack.amazonaws.com/$1/$2/index.html"
