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

# Disable brew auto-update to avoid long running updates while running tests in CI.
export HOMEBREW_NO_AUTO_UPDATE=1

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew install opencv
    # Restrict numpy version to < 1.19.0 due to https://github.com/apache/incubator-mxnet/issues/18600
    python -m pip install --user nose 'numpy>1.16.0,<1.19.0' cython scipy requests mock nose-timer nose-exclude mxnet-to-coreml
fi
