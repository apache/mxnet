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

# Add user in order to make sure the assumed user the container is running under
# actually exists inside the container to avoid problems like missing home dir


set -ex

# $USER_ID is coming from build.py:build_docker passed as --build-arg
if [[ "$USER_ID" -gt 0 ]]
then
    # -no-log-init required due to https://github.com/moby/moby/issues/5419
    if [[ -n "$GROUP_ID" ]] && [[ "$GROUP_ID" -gt 0 ]]
    then
        groupadd --gid $GROUP_ID --system jenkins_slave
        useradd -m --no-log-init --uid $USER_ID --gid $GROUP_ID --system jenkins_slave
    else
        useradd -m --no-log-init --uid $USER_ID --system jenkins_slave
    fi
    usermod -aG sudo jenkins_slave

    # By default, docker creates all WORK_DIRs with root owner
    mkdir /work/mxnet
    mkdir /work/build
    chown -R jenkins_slave /work/
fi
