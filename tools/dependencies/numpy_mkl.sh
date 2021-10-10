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

# This script rebuilds numpy so that it will use MKL BLAS instead of OpenBLAS.
set -x

# check if numpy uses openblas
set +e
python3 -c "from numpy import show_config; show_config()" | grep 'openblas_info'
if [[ $? -eq 0 ]] && [[ -e /opt/intel/oneapi/mkl/ ]] && [[ ! -e ~/.numpy-site.cfg ]]; then
  # create file and add to it MKL configuration
  if [[ $PLATFORM == 'darwin' ]]; then
    echo "[mkl]
  library_dirs = /opt/intel/oneapi/compiler/$INTEL_MKL/mac/compiler/lib:/opt/intel/oneapi/mkl/$INTEL_MKL/lib
  include_dirs = /opt/intel/oneapi/mkl/$INTEL_MKL/include
  libraries = mkl_rt,iomp5
  extra_link_args = -Wl,-rpath,/opt/intel/oneapi/mkl/$INTEL_MKL/lib,-rpath,/opt/intel/oneapi/compiler/$INTEL_MKL/mac/compiler/lib" >> ~/.numpy-site.cfg
  else
    echo "[mkl]
  library_dirs = /opt/intel/oneapi/compiler/$INTEL_MKL/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/mkl/$INTEL_MKL/lib/intel64
  include_dirs = /opt/intel/oneapi/mkl/$INTEL_MKL/include
  libraries = mkl_rt,iomp5
  extra_link_args = -Wl,-rpath,/opt/intel/oneapi/mkl/$INTEL_MKL/lib/intel64,-rpath,/opt/intel/oneapi/compiler/$INTEL_MKL/linux/compiler/lib/intel64_lin" >> ~/.numpy-site.cfg
  fi

  # reinstall numpy to use MKL BLAS
  pip3 install numpy==1.19.5 --no-binary numpy --force-reinstall --no-cache-dir
fi
set -e
