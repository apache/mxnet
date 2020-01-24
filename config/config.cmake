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

#-------------------------------------------------------------------------------
#  Template configuration for compiling MXNet
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of mxnet. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp config/config.cmake config.cmake
#
#  Next modify the according entries, and then compile by
#
#  $ mkdir build; cd build
#  $ cmake -C ../config.cmake ..
#  $ cmake --build . --parallel 8
#
# You can increase the --parallel 8 argument to match the number of processor
# cores of your computer.
#
#-------------------------------------------------------------------------------

#---------------------
# Compilers
#--------------------
# Compilers are usually autodetected. Uncomment and modify the next 3 lines to
# choose manually:

# set(CMAKE_C_COMPILER "" CACHE BOOL "C compiler")
# set(CMAKE_CXX_COMPILER "" CACHE BOOL "C++ compiler")
# set(CMAKE_CUDA_COMPILER "" CACHE BOOL "Cuda compiler (nvcc)")

# Uncomment the following line to compile with debug information
# set(CMAKE_BUILD_TYPE Debug CACHE STRING "CMake build type")

# Whether to build operators written in TVM
set(USE_TVM_OP OFF CACHE BOOL "Enable use of TVM operator build system.")

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------
set(USE_CUDA ON CACHE BOOL "Build with CUDA support")

# Target NVIDIA GPU achitecture.
# Format: Auto | Common | All | LIST(ARCH_AND_PTX ...)
# - "Auto" detects local machine GPU compute arch at runtime.
# - "Common" and "All" cover common and entire subsets of architectures
# - ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
# - NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal Volta Turing
# - NUM: Any number. Only those pairs are currently accepted by NVCC though:
#        2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2 7.0 7.2 7.5
# When compiling on a machine without GPU, autodetection will fail and you
# should instead specify the target architecture manually. If no architecture is
# detected or specified, compilation will target all available architectures,
# which significantly slows down the build .
set(MXNET_CUDA_ARCH "Auto" CACHE STRING "Target NVIDIA GPU achitecture")

set(ENABLE_CUDA_RTC OFF CACHE BOOL "Build with CUDA runtime compilation support")
set(USE_CUDNN ON CACHE BOOL "Build with cudnn support, if found")
# set(USE_NVTX ON CACHE BOOL "Build with NVTX support")  # TODO Currently always autodetected in CMake
set(USE_NCCL "Use NVidia NCCL with CUDA" OFF)
set(NCCL_ROOT "" CACHE BOOL "NCCL install path. Supports autodetection.")

set(USE_OPENCV ON CACHE BOOL "Build with OpenCV support")
set(OPENCV_ROOT "" CACHE BOOL "OpenCV install path. Supports autodetection.")

# USE_LIBJPEG_TURBO = 0  # TODO Not yet supported in CMake build
# set(LIBJPEG_TURBO_ROOT "" CACHE BOOL "libjpeg-turbo install path. Supports autodetection.")

set(USE_OPENMP ON CACHE BOOL "Build with Openmp support")

set(USE_MKL_IF_AVAILABLE ON CACHE BOOL "Use Intel MKL if found")
set(USE_MKLDNN ON CACHE BOOL "Build with MKL-DNN support")

# USE_NNPACK = 0  # TODO Not yet supported in CMake build

# Building with lapack is only effective when compiled with any of
# openblas/apple/atlas/mkl blas versions
set(USE_LAPACK ON CACHE BOOL "Build with lapack support")


#---------------------------------------------
# CPU instruction sets: The support is autodetected if turned ON
#---------------------------------------------
set(USE_SSE ON CACHE BOOL "Build with x86 SSE instruction support")
set(USE_F16C ON CACHE BOOL "Build with x86 F16C instruction support")


#----------------------------
# distributed computing
#----------------------------
set(USE_DIST_KVSTORE OFF CACHE BOOL "Build with DIST_KVSTORE support")
# set(USE_HDFS OFF CACHE BOOL "Allow read and write to HDFS directly; requires hadoop")  # TODO Not yet supported in CMake build
# # path to libjvm.so. required if USE_HDFS=1
# LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server
# # whether or not allow to read and write AWS S3 directly. If yes, then
# # libcurl4-openssl-dev is required, it can be installed on Ubuntu by
# # sudo apt-get install -y libcurl4-openssl-dev
# USE_S3 = 0  # TODO Not yet supported in CMake build


#----------------------------
# performance settings
#----------------------------
set(USE_OPERATOR_TUNING ON CACHE BOOL  "Enable auto-tuning of operators")
set(USE_GPERFTOOLS OFF CACHE BOOL "Build with GPerfTools support")
set(USE_JEMALLOC OFF CACHE BOOL "Build with Jemalloc support")


#----------------------------
# additional operators
#----------------------------
# path to folders containing projects specific operators that you don't want to
# put in src/operators
SET(EXTRA_OPERATORS "" CACHE PATH "EXTRA OPERATORS PATH")


#----------------------------
# other features
#----------------------------
# Create C++ interface package
set(USE_CPP_PACKAGE OFF CACHE BOOL "Build C++ Package")

# Use int64_t type to represent the total number of elements in a tensor
# This will cause performance degradation reported in issue #14496
# Set to 1 for large tensor with tensor size greater than INT32_MAX i.e. 2147483647
# Note: the size of each dimension is still bounded by INT32_MAX
set(USE_INT64_TENSOR_SIZE OFF CACHE BOOL "Use int64_t to represent the total number of elements in a tensor")
