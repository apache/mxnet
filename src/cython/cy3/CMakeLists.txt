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

cmake_minimum_required(VERSION 3.0.2)

project(cy3 C CXX)

# Since this will start a new sub-build that's nbot related to the "100%" displayed above it
message(STATUS "=====================================================")
message(STATUS "= BUILDING CYTHON MODULES FOR PYTHON 3              =")
message(STATUS "=====================================================")

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CythonUtil.cmake)

set(cython_modules "")

mxnet_build_cython_module(3 cython_modules)
#message(STATUS "cython_modules: ${cython_modules}")
#if(INSTALL_CYTHON_INPLACE)
#  message(STATUS "Installing cython in-place")
#  cython_install_into_source_dir("${MXNET_BINARY_DIR}/python" "${MXNET_ROOT_DIR}/python")
#else()
#  message(STATUS "Not installing cython in-place")
#endif()
