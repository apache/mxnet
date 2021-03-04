#===============================================================================
# Copyright 2020 Intel Corporation
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

# Manage Threadpool-related compiler flags
#===============================================================================

if(Threadpool_cmake_included)
    return()
endif()
set(Threadpool_cmake_included true)
include("cmake/Threading.cmake")

if("${DNNL_CPU_THREADING_RUNTIME}" STREQUAL "THREADPOOL")

    if("${_DNNL_TEST_THREADPOOL_IMPL}" STREQUAL "TBB")
        find_package_tbb(REQUIRED)
        if(TBB_FOUND)
            list(APPEND EXTRA_STATIC_LIBS TBB::tbb)
            message(STATUS "Threadpool testing: TBB (${_tbb_root})")
            set(_DNNL_TEST_THREADPOOL_IMPL "TBB")
        endif()
    endif()

    if("${_DNNL_TEST_THREADPOOL_IMPL}" STREQUAL "EIGEN")
        find_package(Eigen3 REQUIRED 3.3 NO_MODULE)
        if(Eigen3_FOUND)
            list(APPEND EXTRA_STATIC_LIBS Eigen3::Eigen)
            message(STATUS "Threadpool testing: Eigen (${EIGEN3_ROOT_DIR})")
        endif()
    endif()

    if("${_DNNL_TEST_THREADPOOL_IMPL}" STREQUAL "STANDALONE")
        message(STATUS "Threadpool testing: standalone")
    endif()

    add_definitions(-DDNNL_TEST_THREADPOOL_USE_${_DNNL_TEST_THREADPOOL_IMPL})
endif()

