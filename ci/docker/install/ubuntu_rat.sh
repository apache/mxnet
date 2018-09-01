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

set -ex

echo "Install dependencies"
apt-get update
apt-get install -y subversion maven openjdk-8-jdk openjdk-8-jre

echo "download RAT"
#svn co http://svn.apache.org/repos/asf/creadur/rat/trunk/
svn co http://svn.apache.org/repos/asf/creadur/rat/branches/0.12-release/

echo "cd into directory"
cd 0.12-release

echo "mvn install"
mvn -Dmaven.test.skip=true install
