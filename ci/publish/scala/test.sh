#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -ex

if [ -z "$JAVA_HOME" ]; then
    source /etc/profile
fi

# Test
cd scala-package/packageTest

if [[ $mxnet_variant == cu* ]]; then
    export SCALA_TEST_ON_GPU=1
    make testlocal USE_CUDA=1 CI=1
else
    make testlocal CI=1
fi
