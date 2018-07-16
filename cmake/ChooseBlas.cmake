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
set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL;Apple")

function(switch_lapack enable)
  if(${enable})
    message(STATUS "Enabling LAPACK functionality")
    add_definitions(-DMXNET_USE_LAPACK=1)
  else()
    message(WARNING "LAPACK functionality not available")
  endif()
endfunction()

function(try_mkldnn)

  if(NOT ${USE_MKLDNN})
    return()
  endif()

  if(${CMAKE_CROSSCOMPILING})
    message(WARNING "MKLDNN with cross compilation is not supported, MKLDNN will not be available")
    return()
  endif()

  if(NOT ${MKL_FOUND})
    message(WARNING "MKLDNN requires either MKL or MKLML, MKLDNN will not be available")
    return()
  endif()

  message(STATUS "Adding MKLDNN to the build")

  # CPU architecture (e.g., C5) can't run on another architecture (e.g., g3).
  if(NOT MSVC)
    set(ARCH_OPT_FLAGS ${ARCH_OPT_FLAGS} "-mtune=generic" PARENT_SCOPE)
  endif()

  set(WITH_TEST OFF)
  set(WITH_EXAMPLE OFF)
  add_subdirectory(3rdparty/mkldnn)

  include_directories(3rdparty/mkldnn/include)
  set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} mkldnn PARENT_SCOPE)

  add_definitions(-DUSE_MKL=1)
  add_definitions(-DCUB_MKL=1)
  add_definitions(-DMXNET_USE_MKLDNN=1)

endfunction()

function(try_mkl)
  if(${USE_MKL_IF_AVAILABLE} AND NOT ${USE_MKLML})
    message(STATUS "Trying to enable MKL framework due to USE_MKL_IF_AVAILABLE")
    find_package(MKL)
    if(${MKL_FOUND})
      message(STATUS "MKL framework found")

      set(MKL_FOUND ${MKL_FOUND} PARENT_SCOPE)
      set(MKLROOT ${MKLROOT} PARENT_SCOPE)

      set(BLAS MKL PARENT_SCOPE)
    else()
      message(STATUS "MKL framework not found")
    endif()
  endif()
endfunction()

function(try_mklml)
  if(NOT ${USE_MKLML})
    return()
  endif()

  if(${MKL_FOUND})
    return()
  endif()

  message(STATUS "Trying to enable MKLML framework due to USE_MKLML")

  include(${CMAKE_CURRENT_LIST_DIR}/DownloadMKLML.cmake)
  find_package(MKLML REQUIRED)
  include_directories(SYSTEM ${MKLML_INCLUDE_DIRS})
  set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${MKLML_LIBRARIES} PARENT_SCOPE)

  set(MKL_FOUND ${MKL_FOUND} PARENT_SCOPE)
  set(MKLROOT ${MKLROOT} PARENT_SCOPE)

  set(BLAS MKL PARENT_SCOPE)

endfunction()

function(try_accelerate)
  if(NOT APPLE)
    return()
  endif()

  if(BLAS MATCHES "[Mm][Kk][Ll]")
    return()
  endif()

  if(${USE_APPLE_ACCELERATE_IF_AVAILABLE})
    message(STATUS "Trying to enable Apple Accelerate framework due to USE_ACCELERATE_IF_AVAILABLE")
    find_package(Accelerate)
    if(${Accelerate_FOUND})
      message(STATUS "Apple Accelerate framework found")
      set(BLAS Accelerate PARENT_SCOPE)
    else()
      message(STATUS "Apple Accelerate framework not found")
    endif()
  endif()
endfunction()

try_mkl()
try_mklml()
try_mkldnn()
try_accelerate()

# cmake regexp does not support case insensitive match (?i) or //i
if(BLAS MATCHES "[Aa][Tt][Ll][Aa][Ss]")
  message(STATUS "Using Atlas for BLAS")

  set(Atlas_NEED_LAPACK ${USE_LAPACK})
  find_package(Atlas REQUIRED)

  include_directories(SYSTEM ${Atlas_INCLUDE_DIRS})
  set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${Atlas_LIBRARIES})

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
  set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${OpenBLAS_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  switch_lapack(${USE_LAPACK} AND ${OpenBLAS_LAPACK_FOUND})

  return()
endif()

if(BLAS MATCHES "[Mm][Kk][Ll]")
  message(STATUS "Using MKL for BLAS")

  find_package(MKL REQUIRED)

  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${MKL_LIBRARIES})

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

if(BLAS MATCHES "([Aa][Pp][Pp][Ll][Ee]|[Aa][Cc][Cc][Ee][Ll][Ee][Rr][Aa][Tt][Ee])")
  if(NOT APPLE)
    message(FATAL_ERROR "Apple Accelerate framework's BLAS feature is available only on macOS")
    return()
  endif()

  message(STATUS "Using Apple Accelerate for BLAS")

  # Accelerate framework documentation
  # https://developer.apple.com/documentation/accelerate?changes=_2
  set(Accelerate_NEED_LAPACK ${USE_LAPACK})
  find_package(Accelerate REQUIRED)

  include_directories(SYSTEM ${Accelerate_INCLUDE_DIR})
  set(mxnet_LINKER_LIBS ${mxnet_LINKER_LIBS} ${Accelerate_LIBRARIES})

  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)

  switch_lapack(${USE_LAPACK} AND ${Accelerate_LAPACK_FOUND})
  return()

endif()

message(FATAL_ERROR "BLAS ${BLAS} not recognized")
