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

# Tries to find Gperftools.
#
# Usage of this module as follows:
#
#     find_package(Gperftools)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  Gperftools_ROOT_DIR  Set this variable to the root installation of
#                       Gperftools if the module has problems finding
#                       the proper installation path.
#
# Variables defined by this module:
#
#  GPERFTOOLS_FOUND              System has Gperftools libs/headers
#  GPERFTOOLS_LIBRARIES          The Gperftools libraries (tcmalloc & profiler)
#  GPERFTOOLS_INCLUDE_DIR        The location of Gperftools headers

find_library(GPERFTOOLS_TCMALLOC
  NAMES tcmalloc
  HINTS ${Gperftools_ROOT_DIR}/lib)

find_library(GPERFTOOLS_PROFILER
  NAMES profiler
  HINTS ${Gperftools_ROOT_DIR}/lib)

find_library(GPERFTOOLS_TCMALLOC_AND_PROFILER
  NAMES tcmalloc_and_profiler
  HINTS ${Gperftools_ROOT_DIR}/lib)

find_path(GPERFTOOLS_INCLUDE_DIR
  NAMES gperftools/heap-profiler.h
  HINTS ${Gperftools_ROOT_DIR}/include)

set(GPERFTOOLS_LIBRARIES ${GPERFTOOLS_TCMALLOC_AND_PROFILER})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gperftools
  DEFAULT_MSG
  GPERFTOOLS_LIBRARIES
  GPERFTOOLS_INCLUDE_DIR)

mark_as_advanced(
  Gperftools_ROOT_DIR
  GPERFTOOLS_TCMALLOC
  GPERFTOOLS_PROFILER
  GPERFTOOLS_TCMALLOC_AND_PROFILER
  GPERFTOOLS_LIBRARIES
  GPERFTOOLS_INCLUDE_DIR)

