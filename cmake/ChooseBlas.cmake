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

set(BLAS "Open" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

function(switch_lapack enable)
  if(${enable})
    message(STATUS "Enabling lapack functionality")
    add_definitions(-DMXNET_USE_LAPACK=1)
  else()
    message(WARNING "Lapack functionality not available")
  endif()
endfunction()

if(USE_MKL_IF_AVAILABLE)
  message(STATUS "Trying to find MKL library due to USE_MKL_IF_AVAILABLE=True...")

  if(NOT MKL_FOUND)
    find_package(MKL)
  endif()

  if(MKL_FOUND)
    message(STATUS "MKL library found, checking if USE_MKLDNN...")

    if(USE_MKLDNN)
      message(STATUS "USE_MKLDNN=True, setting to use OpenBLAS")
      set(BLAS "open")
    else()
      message(STATUS "USE_MKLDNN=False, setting to use MKL")
      set(BLAS "MKL")
    endif()
  else()
    message(STATUS "MKL library not found, BLAS=${BLAS}")
  endif()
endif()

# cmake regexp does not support case insensitive match (?i) or //i
if(BLAS MATCHES "[Aa][Tt][Ll][Aa][Ss]")
  message(STATUS "Using Atlas for BLAS")

  set(Atlas_NEED_LAPACK ${USE_LAPACK})
  find_package(Atlas REQUIRED)

  include_directories(SYSTEM ${Atlas_INCLUDE_DIRS})
  list(APPEND mshadow_LINKER_LIBS ${Atlas_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  switch_lapack(${USE_LAPACK} AND ${Atlas_LAPACK_FOUND})

  return()
endif()

if(BLAS MATCHES "[Oo][Pp][Ee][Nn]")
  message(STATUS "Using OpenBLAS for BLAS")

  set(OpenBLAS_NEED_LAPACK ${USE_LAPACK})

  find_package(OpenBLAS REQUIRED)

  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIRS})
  list(APPEND mshadow_LINKER_LIBS ${OpenBLAS_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  switch_lapack(${USE_LAPACK} AND ${OpenBLAS_LAPACK_FOUND})

  return()
endif()

if(BLAS MATCHES "[Mm][Kk][Ll]")
  message(STATUS "Using MKL for BLAS")

  find_package(MKL REQUIRED)

  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${MKL_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=0)
  add_definitions(-DMSHADOW_USE_MKL=1)

  if(USE_LAPACK)
    include(CheckFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES ${MKL_LIBRARIES})
    check_function_exists("cgees_" LAPACK_FOUND)

    switch_lapack(${LAPACK_FOUND} OR False)

    return()
  endif()
endif()

if(BLAS MATCHES "[Aa][Pp][Pp][Ll][Ee]")
  if(NOT APPLE)
    message(FATAL_ERROR "Apple BLAS framework is available only on MAC")
    return()
  endif()

  message(STATUS "Using Apple Accelerate for BLAS")

  # Accelerate framework documentation
  # https://developer.apple.com/documentation/accelerate?changes=_2
  set(Accelerate_NEED_LAPACK ${USE_LAPACK})
  find_package(Accelerate REQUIRED)

  include_directories(SYSTEM ${Accelerate_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${Accelerate_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  switch_lapack(${USE_LAPACK} AND ${Accelerate_LAPACK_FOUND})
  return()

endif()

message(FATAL_ERROR "BLAS ${BLAS} not recognized")
