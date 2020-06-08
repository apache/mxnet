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

# Find the nvml libraries
#
# The following variables are optionally searched for defaults
#  NVML_ROOT_DIR: Base directory where all NVML components are found
#  NVML_INCLUDE_DIR: Directory where NVML header is found
#  NVML_LIB_DIR: Directory where NVML library is found
#
# The following are set after configuration is done:
#  NVML_FOUND
#  NVML_INCLUDE_DIRS
#  NVML_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NVML in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

if ($ENV{NVML_ROOT_DIR})
  message(WARNING "NVML_ROOT_DIR is deprecated. Please set NVML_ROOT instead.")
endif()

find_path(NVML_INCLUDE_DIRS
  NAMES nvml.h
  HINTS
  ${NVML_INCLUDE_DIR}
  ${NVML_ROOT_DIR}
  ${NVML_ROOT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  $ENV{NVML_DIR}/include
  )

find_library(NVML_LIBRARIES
  NAMES nvidia-ml
  HINTS
  ${NVML_LIB_DIR}
  ${NVML_ROOT_DIR}
  ${NVML_ROOT_DIR}/lib
  ${NVML_ROOT_DIR}/lib/x86_64-linux-gnu
  ${NVML_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
  $ENV{NVML_DIR}/lib
  )

# if not found in any of the above paths, finally, check in the /usr/local/cuda for UNIX systems
if (UNIX)
  set (search_paths "/usr/local/cuda")

  find_path(NVML_INCLUDE_DIRS
    NAMES nvml.h
    PATHS ${search_paths}
    PATH_SUFFIXES include
  )

  find_library(NVML_LIBRARIES
    NAMES nvidia-ml
    PATHS ${search_paths}
    PATH_SUFFIXES lib64/stubs
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML DEFAULT_MSG NVML_INCLUDE_DIRS NVML_LIBRARIES)

if(NVML_FOUND)
  message(STATUS "Found NVML (include: ${NVML_INCLUDE_DIRS}, library: ${NVML_LIBRARIES})")
  mark_as_advanced(NVML_ROOT_DIR NVML_INCLUDE_DIRS NVML_LIBRARIES)
endif()

