#===============================================================================
# Copyright 2019-2020 Intel Corporation
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
#===============================================================================

if(SYCL_cmake_included)
    return()
endif()
set(SYCL_cmake_included true)

if(NOT DNNL_WITH_SYCL)
    return()
endif()

include(FindPackageHandleStandardArgs)
include(CheckCXXCompilerFlag)

# Check if CXX is Intel oneAPI DPC++ Compiler
CHECK_CXX_COMPILER_FLAG(-fsycl DPCPP_SUPPORTED)
find_package(LevelZero)

if(DPCPP_SUPPORTED)
    if(LevelZero_FOUND)
        message(STATUS "DPC++ support is enabled (OpenCL and Level Zero)")
    else()
        message(STATUS "DPC++ support is enabled (OpenCL)")
    endif()

    # Explicitly link against sycl as Intel oneAPI DPC++ Compiler doesn't always do it implicitly
    list(APPEND EXTRA_SHARED_LIBS sycl)
    # Explicitly link against OpenCL as Intel oneAPI DPC++ Compiler doesn't do it implicitly
    list(APPEND EXTRA_SHARED_LIBS OpenCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

    if(LevelZero_FOUND)
        set(DNNL_WITH_LEVEL_ZERO TRUE)
        include_directories(${LevelZero_INCLUDE_DIRS})
    endif()

    set(DNNL_SYCL_DPCPP true CACHE INTERNAL "" FORCE)
    set(DNNL_SYCL_RUNTIME "DPCPP")
else()
    message(STATUS "SYCL support is enabled (OpenCL)")
    set(_computecpp_flags "-Wno-sycl-undef-func -no-serial-memop")
    set(COMPUTECPP_USER_FLAGS "${_computecpp_flags} ${COMPUTECPP_USER_FLAGS}"
            CACHE STRING "")

    find_package(ComputeCpp REQUIRED)
    if(ComputeCpp_FOUND)
         list(APPEND EXTRA_SHARED_LIBS ComputeCpp::ComputeCpp)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ComputeCpp_FLAGS}")
         include_directories(${ComputeCpp_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
         set(DNNL_SYCL_COMPUTECPP true CACHE INTERNAL "" FORCE)
         set(DNNL_SYCL_RUNTIME "COMPUTECPP")
     endif()
endif()
