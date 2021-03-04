#===============================================================================
# Copyright 2017-2020 Intel Corporation
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

# Manage OpenMP-related compiler flags
#===============================================================================

if(OpenMP_cmake_included)
    return()
endif()
set(OpenMP_cmake_included true)
include("cmake/Threading.cmake")

if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # OSX Clang doesn't have OpenMP by default.
    # But we still want to build the library.
    set(_omp_severity "WARNING")
else()
    set(_omp_severity "FATAL_ERROR")
endif()

macro(forbid_link_compiler_omp_rt)
    if (NOT WIN32)
        set_if(OpenMP_C_FOUND
            CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS
            "${OpenMP_C_FLAGS}")
        set_if(OpenMP_CXX_FOUND
            CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS
            "${OpenMP_CXX_FLAGS}")
        if (NOT APPLE)
            append(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed")
        endif()
    endif()
endmacro()

macro(set_openmp_values_for_old_cmake)
    #newer version for findOpenMP (>= v. 3.9)
    if(CMAKE_VERSION VERSION_LESS "3.9" AND OPENMP_FOUND)
        if(${CMAKE_MAJOR_VERSION} VERSION_LESS "3" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
            # Override FindOpenMP flags for Intel Compiler (otherwise deprecated)
            set(OpenMP_CXX_FLAGS "-fopenmp")
            set(OpenMP_C_FLAGS "-fopenmp")
        endif()
        set(OpenMP_C_FOUND true)
        set(OpenMP_CXX_FOUND true)
    endif()
endmacro()

find_package(OpenMP)
set_openmp_values_for_old_cmake()

# special case for clang-cl (not recognized by cmake up to 3.17)
if(NOT OpenMP_CXX_FOUND AND MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT DNNL_SYCL_DPCPP)
    # clang-cl and icx will fall under this condition
    # CAVEAT: undocumented variable, may be inappropriate
    if(CMAKE_BASE_NAME STREQUAL "icx")
        # XXX: Use `-Xclang --dependent-lib=libiomp5md` to workaround an issue
        # with linking OpenMP on Windows.
        # The ICX driver doesn't link OpenMP library even if `/Qopenmp`
        # was specified.
        set(OpenMP_FLAGS "/Qopenmp -Xclang --dependent-lib=libiomp5md")
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
            # version < 10 can't pass cl-style `/openmp` flag
            set(OpenMP_FLAGS "-Xclang -fopenmp")
            # ... and requires explicit linking against omp library
            set(OpenMP_CXX_LIBRARIES "libomp.lib")
        endif()
    endif()
    set(OpenMP_C_FLAGS ${OpenMP_FLAGS})
    set(OpenMP_CXX_FLAGS ${OpenMP_FLAGS})
    set(OpenMP_CXX_FOUND true)
endif()

# add flags unconditionally to always utilize openmp-simd for any threading runtime
if(OpenMP_CXX_FOUND)
    append(CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
    append(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif()

if(DNNL_CPU_THREADING_RUNTIME MATCHES "OMP")
    if(OpenMP_CXX_FOUND)
        if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            list(APPEND EXTRA_SHARED_LIBS ${OpenMP_CXX_LIBRARIES})
        endif()
    else()
        message(${_omp_severity} "OpenMP library could not be found. "
            "Proceeding might lead to highly sub-optimal performance.")
        # Override CPU threading to sequential if allowed to proceed
        set(DNNL_CPU_THREADING_RUNTIME "SEQ")
    endif()
else()
    # Compilation happens with OpenMP to enable `#pragma omp simd`
    # but during linkage OpenMP dependency should be avoided
    forbid_link_compiler_omp_rt()
    return()
endif()
