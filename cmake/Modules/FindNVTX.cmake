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

set(NVTX_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA NVTX")

find_path(NVTX_INCLUDE_DIRS
  NAMES nvToolsExt.h
  PATHS $ENV{NVTOOLSEXT_PATH} ${NVTX_ROOT_DIR}  ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include
  )

find_library(NVTX_LIBRARIES
  NAMES nvToolsExt64_1.lib nvToolsExt32_1.lib nvToolsExt
  PATHS $ENV{NVTOOLSEXT_PATH} ${NVTX_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/Win32 lib/x64
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIRS NVTX_LIBRARIES)

if(NVTX_FOUND)
  message(STATUS "Found NVTX (include: ${NVTX_INCLUDE_DIRS}, library: ${NVTX_LIBRARIES})")
  mark_as_advanced(NVTX_ROOT_DIR NVTX_INCLUDE_DIRS NVTX_LIBRARIES)
endif()
