#!/usr/bin/env bash
# -*- coding: utf-8 -*-

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


mxnet_variant=${1:?"Please specify the mxnet variant as the first parameter"}
is_release=${RELEASE_BUILD:-false}
version=${VERSION:-nightly}

if [[ ${version} == "null" ]]; then
    version="nightly"
fi

# The docker tags will be in the form <version>_<hardware>
# Eg. nightly_cpu, 2.0.0_cpu, nightly_gpu_cu110, etc.

if [[ ${mxnet_variant} == "cpu" ]]; then
    tag_suffix="cpu"
elif [[ ${mxnet_variant} == "native" ]]; then
    tag_suffix="native"
elif [[ ${mxnet_variant} == cu* ]]; then
    tag_suffix="gpu_${mxnet_variant}"

else
    echo "Error: Unrecognized mxnet variant: '${mxnet_variant}'."
    exit 1
fi

echo "${version}_${tag_suffix}"
