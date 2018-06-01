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

  include(${CMAKE_CURRENT_LIST_DIR}/Modules/FindAtlas.cmake)

  include_directories(SYSTEM ${Atlas_INCLUDE_DIRS})
  list(APPEND mshadow_LINKER_LIBS ${Atlas_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  if(USE_LAPACK AND Atlas_LAPACK_FOUND)
    add_definitions(-DMXNET_USE_LAPACK=1)
  endif()

  return()
endif()

if(BLAS MATCHES "[Oo][Pp][Ee][Nn]")
  message(STATUS "Using OpenBLAS for BLAS")

  set(OpenBLAS_NEED_LAPACK ${USE_LAPACK})

  include(${CMAKE_CURRENT_LIST_DIR}/Modules/FindOpenBLAS.cmake)

  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIRS})
  list(APPEND mshadow_LINKER_LIBS ${OpenBLAS_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  if(USE_LAPACK AND OpenBLAS_LAPACK_FOUND)
    add_definitions(-DMXNET_USE_LAPACK=1)
  endif()

  return()
endif()

if(BLAS MATCHES "[Mm][Kk][Ll]")
  message(STATUS "Using MKL for BLAS")

  # todo(lebeg): include(${CMAKE_CURRENT_LIST_DIR}/Modules/FindMKL.cmake)

  find_package(MKL REQUIRED)

  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${MKL_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=0)
  add_definitions(-DMSHADOW_USE_MKL=1)

  if(USE_LAPACK)
    include(CheckFunctionExists)
    check_function_exists("cheev_" LAPACK_FOUND)

    if(LAPACK_FOUND)
      add_definitions(-DMXNET_USE_LAPACK=1)
    endif()
  endif()

  return()
endif()

if(BLAS MATCHES "[Aa][Pp][Pp][Ll][Ee]")
  if(NOT APPLE)
    message(FATAL_ERROR "Apple BLAS framework is available only on MAC")
    return()
  endif()

  message(STATUS "Using Apple Accelerate for BLAS")

  # Accelerate framework documentation
  # https://developer.apple.com/documentation/accelerate?changes=_2
  find_package(Accelerate REQUIRED)
  include_directories(SYSTEM ${Accelerate_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${Accelerate_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  if(USE_LAPACK)
    # Apples vecLib should contain lapack functionalities included in the Accelerate framework, but we will double check
    # https://developer.apple.com/documentation/accelerate/veclib?changes=_2
    include(CheckFunctionExists)
    check_function_exists("cheev_" LAPACK_FOUND)

    if(LAPACK_FOUND)
      add_definitions(-DMXNET_USE_LAPACK=1)
    endif()
  endif()

  return()
endif()
