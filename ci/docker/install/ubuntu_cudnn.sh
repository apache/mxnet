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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

# Assumes base image is from nvidia/cuda

set -ex

if [ -z ${CUDNN_VERSION} ]; then
	echo "Error: CUDNN_VERSION environment variable undefiend"
	exit 1
fi

apt-get update || true

case ${CUDA_VERSION} in
	10\.1*)
		export libcudnn7_version="${CUDNN_VERSION}-1+cuda10.1"
		export libcudnn7_dev_version="${CUDNN_VERSION}-1+cuda10.1"
		;;
	10\.0*)
		export libcudnn7_version="${CUDNN_VERSION}-1+cuda10.0"
		export libcudnn7_dev_version="${CUDNN_VERSION}-1+cuda10.0"
		;;
	9\.0*)
		export libcudnn7_version="${CUDNN_VERSION}-1+cuda9.0"
		export libcudnn7_dev_version="${CUDNN_VERSION}-1+cuda9.0"
		;;
	9\.2*)
		export libcudnn7_version="${CUDNN_VERSION}-1+cuda9.2"
		export libcudnn7_dev_version="${CUDNN_VERSION}-1+cuda9.2"
		;;
	*)
		echo "Unsupported CUDA version ${CUDA_VERSION}"
		exit 1
		;;
esac

apt-get install -y --allow-downgrades libcudnn7=${libcudnn7_version} libcudnn7-dev=${libcudnn7_dev_version}

