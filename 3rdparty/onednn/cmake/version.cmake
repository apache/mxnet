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

# Control generating version file
#===============================================================================

if(version_cmake_included)
    return()
endif()
set(version_cmake_included true)

string(REPLACE "." ";" VERSION_LIST ${PROJECT_VERSION})
list(GET VERSION_LIST 0 DNNL_VERSION_MAJOR)
list(GET VERSION_LIST 1 DNNL_VERSION_MINOR)
list(GET VERSION_LIST 2 DNNL_VERSION_PATCH)

find_package(Git)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE DNNL_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
    set(DNNL_VERSION_HASH "N/A")
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/dnnl_version.h.in"
    "${PROJECT_BINARY_DIR}/include/oneapi/dnnl/dnnl_version.h"
)

if(WIN32)
    string(TIMESTAMP DNNL_VERSION_YEAR "%Y")
    set(VERSION_RESOURCE_FILE ${PROJECT_BINARY_DIR}/src/version.rc)
    configure_file(${PROJECT_SOURCE_DIR}/cmake/version.rc.in
        ${VERSION_RESOURCE_FILE})
else()
    set(VERSION_RESOURCE_FILE "")
endif()
