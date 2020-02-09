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

set(CMAKE_BUILD_TYPE "Distribution" CACHE STRING "Build type")
set(CFLAGS "-mno-avx" CACHE STRING "CFLAGS")
set(CXXFLAGS "-mno-avx" CACHE STRING "CXXFLAGS")

set(USE_CUDA ON CACHE BOOL "Build with CUDA support")
set(USE_CUDNN ON CACHE BOOL "Build with CUDA support")
set(USE_OPENCV ON CACHE BOOL "Build with OpenCV support")
set(USE_OPENMP ON CACHE BOOL "Build with Openmp support")
set(USE_MKL_IF_AVAILABLE OFF CACHE BOOL "Use Intel MKL if found")
set(USE_MKLDNN ON CACHE BOOL "Build with MKL-DNN support")
set(USE_LAPACK ON CACHE BOOL "Build with lapack support")
set(USE_TVM_OP OFF CACHE BOOL "Enable use of TVM operator build system.")
set(USE_SSE ON CACHE BOOL "Build with x86 SSE instruction support")
set(USE_F16C OFF CACHE BOOL "Build with x86 F16C instruction support")

set(CUDACXX "/usr/local/cuda-7.5/bin/nvcc" CACHE STRING "Cuda compiler")
set(MXNET_CUDA_ARCH "3.0;3.5;5.0;5.2" CACHE STRING "Cuda architectures")
