# *******************************************************************************
# Copyright 2020 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

if(blas_cmake_included)
    return()
endif()
set(blas_cmake_included true)
include("cmake/options.cmake")

# Retains existing functionality of _DNNL_USE_MKL
if(_DNNL_USE_MKL)
    set(DNNL_BLAS_VENDOR "MKL")
endif()

if(DNNL_BLAS_VENDOR STREQUAL "NONE")
    return()
endif()

if (NOT "${DNNL_BLAS_VENDOR}" MATCHES "^(NONE|MKL|OPENBLAS|ARMPL|ANY)$")
    message(FATAL_ERROR "Unsupported DNNL_BLAS_VENDOR: ${DNNL_BLAS_VENDOR}.")
endif()

macro(expect_arch_or_generic arch)
    if(NOT "${DNNL_TARGET_ARCH}" MATCHES "^(${arch}|ARCH_GENERIC)$")
        message(FATAL_ERROR "DNNL_BLAS_VENDOR=${DNNL_BLAS_VENDOR} is not supported "
            "for DNNL_TARGET_ARCH=${DNNL_TARGET_ARCH}.")
        return()
    endif()
endmacro()

# Check chosen DNNL_BLAS_VENDOR is supported and set BLA_VENDOR accordingly
set(CBLAS_HEADERS "cblas.h")
if(DNNL_BLAS_VENDOR STREQUAL "MKL")
    expect_arch_or_generic("X64")
    set(BLA_VENDOR "Intel10_64_dyn")
    set(CBLAS_HEADERS "mkl_cblas.h")
elseif(DNNL_BLAS_VENDOR STREQUAL "OPENBLAS")
    set(BLA_VENDOR "OpenBLAS")
elseif(DNNL_BLAS_VENDOR STREQUAL "ARMPL")
    expect_arch_or_generic("AARCH64")
    if(DNNL_CPU_RUNTIME STREQUAL "OMP")
        set(BLA_VENDOR "Arm_mp")
    else()
        set(BLA_VENDOR "Arm")
    endif()
endif()

find_package(BLAS REQUIRED)

if(BLAS_FOUND)
     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
     list(APPEND EXTRA_SHARED_LIBS ${BLAS_LIBRARIES})

     # Check that the BLAS library supports the CBLAS interface.
     set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LINKER_FLAGS};${BLAS_LIBRARIES}")
     set(CMAKE_REQUIRED_FLAGS "${BLAS_COMPILER_FLAGS}")

     # Find and include  accompanying cblas.h
     list(GET BLAS_LIBRARIES 0 FIRST_BLAS_LIB)
     get_filename_component(BLAS_LIB_DIR ${FIRST_BLAS_LIB} PATH)
     find_path(BLAS_INCLUDE_DIR ${CBLAS_HEADERS} $ENV{CPATH} ${BLAS_LIB_DIR}/../include ${BLAS_LIB_DIR}/../../include)
     include_directories(${BLAS_INCLUDE_DIR})

     # Check we have a working CBLAS interface
     unset(CBLAS_WORKS CACHE)
     check_function_exists(cblas_sgemm CBLAS_WORKS)
     if(NOT CBLAS_WORKS)
         message(FATAL_ERROR "BLAS library does not support CBLAS interface.")
     endif()

     message(STATUS "Found CBLAS: ${BLAS_LIBRARIES}")
     message(STATUS "CBLAS include path: ${BLAS_INCLUDE_DIR}")
     add_definitions(-DUSE_CBLAS)
     if (DNNL_BLAS_VENDOR STREQUAL "MKL")
         add_definitions(-DUSE_MKL)
     endif()
endif()
