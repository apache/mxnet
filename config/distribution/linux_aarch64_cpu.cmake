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
set(CFLAGS "-march=armv8-a" CACHE STRING "CFLAGS")
set(CXXFLAGS "-march=armv8-a" CACHE STRING "CXXFLAGS")

set(USE_CUDA OFF CACHE BOOL "Build with CUDA support")
set(USE_OPENCV ON CACHE BOOL "Build with OpenCV support")
set(USE_OPENMP ON CACHE BOOL "Build with Openmp support")
set(USE_MKL_IF_AVAILABLE OFF CACHE BOOL "Use Intel MKL if found")
set(USE_LAPACK ON CACHE BOOL "Build with lapack support")
set(USE_TVM_OP OFF CACHE BOOL "Enable use of TVM operator build system.")
set(USE_SSE OFF CACHE BOOL "Build with x86 SSE instruction support")
set(USE_F16C OFF CACHE BOOL "Build with x86 F16C instruction support")
set(USE_DIST_KVSTORE OFF CACHE BOOL "Build with DIST_KVSTORE support")

set(USE_MKLDNN ON CACHE BOOL "Build with MKL-DNN support")
# Pre-built binaries are available from  https://github.com/ARM-software/ComputeLibrary/releases
# Make sure to copy and rename the appropriate binaries folder
# from <acl_root>/lib/<binaries_folder_for_your_arch> to <acl_root>/build
# The resulting acl root folder should look something like:
# LICENSE README.md arm_compute build examples include lib scripts support utils
set(ENV{ACL_ROOT_DIR} "")
set(MKLDNN_USE_ACL OFF CACHE BOOL "Integrate MKLDNN with Arm Compute Library")
# APL can be downloaded from https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries
# Note that APL needs to be added to LD_LIBRARY_PATH
set(MKLDNN_USE_APL ON CACHE BOOL "Integrate MKLDNN with Arm Performance Libraries")
