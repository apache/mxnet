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

# install libraries for mxnet's julia package on ubuntu

# the julia version shipped with ubuntu (version 0.4) is too low. so download a
# new version
# apt-get install -y julia

wget -q https://julialang.s3.amazonaws.com/bin/linux/x64/0.5/julia-0.5.1-linux-x86_64.tar.gz
tar -zxf julia-0.5.1-linux-x86_64.tar.gz
rm julia-0.5.1-linux-x86_64.tar.gz
ln -s $(pwd)/julia-6445c82d00/bin/julia /usr/bin/julia
