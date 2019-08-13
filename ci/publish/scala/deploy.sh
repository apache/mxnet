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

# On Jenkins, run python script to configure keys
if [[ $BUILD_ID ]]; then
    python3 ci/publish/scala/buildkey.py
fi

# Updating cache
mkdir -p ~/.gnupg
echo "default-cache-ttl 14400" > ~/.gnupg/gpg-agent.conf
echo "max-cache-ttl 14400" >> ~/.gnupg/gpg-agent.conf
echo "allow-loopback-pinentry" >> ~/.gnupg/gpg-agent.conf
echo "pinentry-mode loopback" >> ~/.gnupg/gpg-agent.conf
export GPG_TTY=$(tty)

cd scala-package

mvn -B deploy -Pnightly

# On Jenkins, clear all password .xml files, exp files, and gpg key files
if [[ $BUILD_ID ]]; then
    rm -rf ~/.m2/*.xml ~/.m2/key.asc ~/.m2/*.exp
fi
