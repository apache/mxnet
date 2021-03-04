#===============================================================================
# Copyright 2018-2020 Intel Corporation
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

# Utils for managing threading-related configuration
#===============================================================================

if(Threading_cmake_included)
    return()
endif()
set(Threading_cmake_included true)

# CPU threading runtime specifies the threading used by the library:
# sequential, OpenMP or TBB. In future it may be different from CPU runtime.
set(DNNL_CPU_THREADING_RUNTIME "${DNNL_CPU_RUNTIME}")

# Always require pthreads even for sequential threading (required for e.g.
# std::call_once that relies on mutexes)
find_package(Threads REQUIRED)
list(APPEND EXTRA_SHARED_LIBS "${CMAKE_THREAD_LIBS_INIT}")

# A macro to avoid code duplication
macro(find_package_tbb)
    if(WIN32)
        find_package(TBB ${ARGN} COMPONENTS tbb HINTS cmake/win)
    elseif(APPLE)
        find_package(TBB ${ARGN} COMPONENTS tbb HINTS cmake/mac)
    elseif(UNIX)
        find_package(TBB ${ARGN} COMPONENTS tbb HINTS cmake/lnx)
    endif()

    if(TBB_FOUND)
        # Check for TBB version, required >= 2017
        foreach (_tbb_ver_header tbb_stddef.h version.h)
        foreach (_tbb_include_subdir oneapi "")
            get_target_property(_tbb_include_dirs TBB::tbb
                INTERFACE_INCLUDE_DIRECTORIES)
            set(_tbb_ver_header_full_path
                "${_tbb_include_dirs}/${_tbb_include_subdir}/tbb/${_tbb_ver_header}")

            if(EXISTS ${_tbb_ver_header_full_path})
                file(READ "${_tbb_ver_header_full_path}" _tbb_ver)
                string(REGEX REPLACE
                    ".*#define TBB_INTERFACE_VERSION ([0-9]+).*" "\\1"
                    TBB_INTERFACE_VERSION "${_tbb_ver}")
                if (${TBB_INTERFACE_VERSION} VERSION_LESS 9100)
                    if("${mode}" STREQUAL REQUIRED)
                        message(FATAL_ERROR
                            "oneDNN requires TBB version 2017 or above")
                    endif()
                    unset(TBB_FOUND)
                else()
                    break()
                endif()
            endif()
        endforeach()
        endforeach()
    endif()
endmacro()

