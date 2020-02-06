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

#Install steps for the nightly tests

set -ex

# Install for Compilation warning Nightly Test
# Adding ppas frequently fails due to busy gpg servers, retry 5 times with 5 minute delays.
for i in 1 2 3 4 5; do add-apt-repository -y ppa:ubuntu-toolchain-r/test && break || sleep 300; done

apt-get update || true
apt-get -y install time

# Install for RAT License Check Nightly Test
apt-get install -y subversion maven -y #>/dev/null

# Packages needed for the Straight Dope Nightly tests.
pip3 install pandas scikit-image prompt_toolkit
