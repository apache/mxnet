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

# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

if ($ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS
  ${NCCL_INCLUDE_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  $ENV{NCCL_DIR}/include
  )

if(CMAKE_BUILD_TYPE STREQUAL "Distribution" AND UNIX)
  set(NCCL_LIB_NAME "nccl_static")
else()
  set(NCCL_LIB_NAME "nccl")
endif()

find_library(NCCL_LIBRARIES
  NAMES ${NCCL_LIB_NAME}
  HINTS
  ${NCCL_LIB_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/lib
  ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
  ${NCCL_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  $ENV{NCCL_DIR}/lib
  )

# if not found in any of the above paths, finally, check in the /usr/local/cuda for UNIX systems
if (UNIX)
  set (search_paths "/usr/local/cuda")

  find_path(NCCL_INCLUDE_DIRS
    NAMES nccl.h
    PATHS ${search_paths}
    PATH_SUFFIXES include
  )

  find_library(NCCL_LIBRARIES
    NAMES ${NCCL_LIB_NAME}
    PATHS ${search_paths}
    PATH_SUFFIXES lib
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if(NCCL_FOUND)
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()

