#===============================================================================
# Copyright 2016-2020 Intel Corporation
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

if (NOT DNNL_BUILD_TESTS)
    return()
endif()

if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

# propagate TEST specific flags
append(CMAKE_C_FLAGS "${CMAKE_TEST_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_TEST_CCXX_FLAGS}")

set(CMAKE_TEST_CCXX_NOWARN_FLAGS)

# propagate no warning flags
append(CMAKE_TEST_CCXX_NOWARN_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

# allow tests to include internal header files with, e.g.
# include "src/common/dnnl_thread.hpp" ...
include_directories(${PROJECT_SOURCE_DIR})
# ... and allow the included files transitively include other headers files
# e.g. include "common/dnnl_thread_parallel_nd.hpp" from "dnnl_thread.hpp"
include_directories(${PROJECT_SOURCE_DIR}/src)

if(WIN32 AND NOT MINGW)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # 2415: unused variable
        append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/Qdiag-disable:2415")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # c4244: conversion with possible loss of data
        # c4996: unsafe / deprecated functions
        append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/wd4996 /wd4244")
        if ("${DNNL_CPU_THREADING_RUNTIME}" STREQUAL "THREADPOOL" AND Eigen3_FOUND)
            # c4267: conversion with possible loss of data
            append(CMAKE_TEST_CCXX_NOWARN_FLAGS "/wd4267")
        endif()
    endif()
endif()

append(CMAKE_C_FLAGS "${CMAKE_TEST_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_TEST_CCXX_NOWARN_FLAGS}")

register_exe(api-c api.c "test")

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND (UNIX OR MINGW))
    get_directory_property(include_dirs INCLUDE_DIRECTORIES)
    set(test_c_symbols "${CMAKE_CURRENT_BINARY_DIR}/test_c_symbols.c")
    add_custom_command(
        OUTPUT ${test_c_symbols}
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_c_symbols_refs.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/.. ${test_c_symbols} ${include_dirs}
    )
    register_exe(test_c_symbols-c ${test_c_symbols} "test")
# elseif(WIN32)
# No Windows support for: test_c_symbols.c
endif()

add_definitions(-DNOMINMAX) # to allow std::max on Windows with parentheses

if(NOT DNNL_ENABLE_PRIMITIVE_CACHE)
    add_definitions(-DDNNL_DISABLE_PRIMITIVE_CACHE)
endif()

if ("${DNNL_CPU_THREADING_RUNTIME}" STREQUAL "THREADPOOL")
    # Only enable this for threadpool runtime to avoid complaints about empty
    # object files.
    set(TEST_THREAD ${CMAKE_CURRENT_SOURCE_DIR}/test_thread.cpp)
endif()

add_subdirectory(gtests)
add_subdirectory(benchdnn)

if(NOT DNNL_WITH_SYCL)
    add_subdirectory(mkldnn_compat)
    if(UNIX OR MINGW)
        add_subdirectory(noexcept)
    endif()
endif()
