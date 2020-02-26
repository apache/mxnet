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

message(STATUS "Prepare external packages for TVM...")
execute_process(COMMAND "sh" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/tvmop/prepare_tvm.sh")

# Whether enable ROCM runtime
#
# Possible values:
# - ON: enable ROCM with cmake's auto search
# - OFF: disable ROCM
# - /path/to/rocm: use specific path to rocm
set(USE_ROCM OFF)

# Whether enable SDAccel runtime
set(USE_SDACCEL OFF)

# Whether enable Intel FPGA SDK for OpenCL (AOCL) runtime
set(USE_AOCL OFF)

# Whether enable OpenCL runtime
set(USE_OPENCL OFF)

# Whether enable Metal runtime
set(USE_METAL OFF)

# Whether enable Vulkan runtime
#
# Possible values:
# - ON: enable Vulkan with cmake's auto search
# - OFF: disable vulkan
# - /path/to/vulkan-sdk: use specific path to vulkan-sdk
set(USE_VULKAN OFF)

# Whether enable OpenGL runtime
set(USE_OPENGL OFF)

# Whether to enable SGX runtime
#
# Possible values for USE_SGX:
# - /path/to/sgxsdk: path to Intel SGX SDK
# - OFF: disable SGX
#
# SGX_MODE := HW|SIM
set(USE_SGX OFF)
set(SGX_MODE "SIM")
set(RUST_SGX_SDK "/path/to/rust-sgx-sdk")

# Whether enable RPC runtime
set(USE_RPC ON)

# Whether embed stackvm into the runtime
set(USE_STACKVM_RUNTIME OFF)

# Whether enable tiny embedded graph runtime.
set(USE_GRAPH_RUNTIME ON)

# Whether enable additional graph debug functions
set(USE_GRAPH_RUNTIME_DEBUG OFF)

# Whether build with LLVM support
# Requires LLVM version >= 4.0
#
# Possible values:
# - ON: enable llvm with cmake's find search
# - OFF: disable llvm
# - /path/to/llvm-config: enable specific LLVM when multiple llvm-dev is available.
set(USE_LLVM "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/build/llvm/bin/llvm-config")

#---------------------------------------------
# Contrib libraries
#---------------------------------------------
# Whether use BLAS, choices: openblas, mkl, atlas, apple
set(USE_BLAS none)

# /path/to/mkl: mkl root path when use mkl blas library
# set(USE_MKL_PATH /opt/intel/mkl) for UNIX
# set(USE_MKL_PATH ../IntelSWTools/compilers_and_libraries_2018/windows/mkl) for WIN32
set(USE_MKL_PATH none)

# Whether use contrib.random in runtime
set(USE_RANDOM OFF)

# Whether use NNPack
set(USE_NNPACK OFF)

# First-class Cuda in modern CMake provides us with CMAKE_CUDA_COMPILER But TVM
# uses the deprecated findCUDA functionality which requires
# CUDA_TOOLKIT_ROOT_DIR We follow the FindCUDAToolkit.cmake logic to compute
# CUDA_TOOLKIT_ROOT_DIR for TVM https://gitlab.kitware.com/cmake/cmake/merge_requests/4093/
if(USE_CUDA)
  get_filename_component(cuda_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  set(CUDA_BIN_DIR "${cuda_dir}" CACHE PATH "" FORCE)
  unset(cuda_dir)
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR ${CUDA_BIN_DIR} DIRECTORY ABSOLUTE)

  message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
  message("Inferred CUDA_TOOLKIT_ROOT_DIR for TVM as: ${CUDA_TOOLKIT_ROOT_DIR}")
  set(USE_CUDA ${CUDA_TOOLKIT_ROOT_DIR})
endif()

# Whether use cuBLAS
set(USE_CUBLAS OFF)

# Whether use MIOpen
set(USE_MIOPEN OFF)

# Whether use MPS
set(USE_MPS OFF)

# Whether use rocBlas
set(USE_ROCBLAS OFF)

# Whether use contrib sort
set(USE_SORT OFF)

# Build ANTLR parser for Relay text format
set(USE_ANTLR OFF)

# Build TSIM for VTA
set(USE_VTA_TSIM OFF)

# Whether use Relay debug mode
set(USE_RELAY_DEBUG OFF)

# Use OPENMP thread pool to be compatible with MXNet
set(USE_OPENMP ON)

# Disable USE_MKLDNN for TVM
set(USE_MKLDNN OFF)
