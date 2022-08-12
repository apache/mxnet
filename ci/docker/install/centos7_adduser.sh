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
    useradd -m --no-log-init --uid $USER_ID --system jenkins_slave 
    usermod -aG wheel jenkins_slave

    # By default, docker creates all WORK_DIRs with root owner
    mkdir /work/mxnet
    mkdir /work/build
    chown -R jenkins_slave /work/

    # Later on, we have to override the links because underlying build systems ignore our compiler settings. Thus,
    # we have to give the process the proper permission to these files. This is hacky, but unfortunately 
    # there's no better way to do this without patching all our submodules.
    chown -R jenkins_slave /usr/local/bin
fi
