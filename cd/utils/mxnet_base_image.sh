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

case ${mxnet_variant} in
    cu100*)
    echo "nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04"
    ;;
    cu101*)
    echo "nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04"
    ;;
    cu102*)
    echo "nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04"
    ;;
    cu110*)
    echo "nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04"
    ;;
    cu111*)
    echo "nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04"
    ;;
    cu112*)
    echo "nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04"
    ;;
    cu113*)
    echo "nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04"
    ;;
    cu114*)
    echo "nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu18.04"
    ;;
    cu115*)
    echo "nvidia/cuda:11.5.2-cudnn8-runtime-ubuntu18.04"
    ;;
    cu116*)
    echo "nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu18.04"
    ;;
    cu117*)
    echo "nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu18.04"
    ;;
    cpu)
    echo "ubuntu:18.04"
    ;;
    native)
    echo "ubuntu:18.04"
    ;;
    aarch64_cpu)
    echo "arm64v8/ubuntu:18.04"
    ;;
    *)
    echo "Error: Unrecognized mxnet-variant: '${mxnet_variant}'"
    exit 1
    ;;
esac
