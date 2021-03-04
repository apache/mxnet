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

# Manage code coverage compiler flags
#===============================================================================

if(Coverage_cmake_included)
    return()
endif()

set(Coverage_cmake_included true)

if(NOT DNNL_CODE_COVERAGE)
    return()
endif()

include("cmake/utils.cmake")

if("${DNNL_CODE_COVERAGE}" STREQUAL "GCOV")
    find_program(GCOV_PATH gcov)

    if(NOT GCOV_PATH)
        message(FATAL_ERROR "GCOV not found in path")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
        if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 3)
            message(FATAL_ERROR "Clang version must be 3.0.0 or greater! Aborting...")
        endif()
    elseif(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -ftest-coverage"
        CACHE INTERNAL "")

    if(NOT CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
        message(WARNING "Code coverage results with an optimised (non-Debug) build may be misleading")
    endif() 

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        link_libraries(gcov)
    else()
        append(CMAKE_EXE_LINKER_FLAGS "--coverage")
    endif()
endif()

message(STATUS "Code coverage enabled")
append(CMAKE_SRC_CCXX_FLAGS "-O0 ${COVERAGE_COMPILER_FLAGS}")
# With coverage flags testing require a lot of machine time
append(CMAKE_EXAMPLE_CCXX_FLAGS "-O3 ${COVERAGE_COMPILER_FLAGS}")
append(CMAKE_TEST_CCXX_FLAGS "-O3 ${COVERAGE_COMPILER_FLAGS}")
