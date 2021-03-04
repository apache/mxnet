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

# Manage OpenCL-related compiler flags
#===============================================================================

if(OpenCL_cmake_included)
    return()
endif()
set(OpenCL_cmake_included true)

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    message(STATUS "GPU support is enabled (OpenCL)")
else()
    return()
endif()

# Set the macro to look specifically for the minimal supported version. Some
# implementations ignore this macro but the version will be checked after
# find_package() anyway.
set(CMAKE_REQUIRED_DEFINITIONS "-DCL_TARGET_OPENCL_VERSION=120")
find_package(OpenCL REQUIRED)

if(OpenCL_VERSION_STRING VERSION_LESS "1.2")
    message(FATAL_ERROR
        "OpenCL version ${OpenCL_VERSION_STRING} is not supported, must be 1.2 or greater.")
endif()

add_definitions(-DCL_TARGET_OPENCL_VERSION=120)

set(DNNL_GPU_RUNTIME_CURRENT ${DNNL_GPU_RUNTIME})
include_directories(${OpenCL_INCLUDE_DIRS})
list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
