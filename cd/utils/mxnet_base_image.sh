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
    cu80*)
    echo "nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04"
    ;;
    cu90*)
    echo "nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"
    ;;
    cu92*)
    echo "nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04"
    ;;
    cu100*)
    echo "nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04"
    ;;
    cu101*)
    echo "nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04"
    ;;
    cpu)
    echo "ubuntu:16.04"
    ;;
    mkl)
    echo "ubuntu:16.04"
    ;;
    *)
    echo "Error: Unrecognized mxnet-variant: '${mxnet_variant}'"
    exit 1
    ;;
esac
