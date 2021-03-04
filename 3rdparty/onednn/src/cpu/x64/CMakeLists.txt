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

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

if(NOT DNNL_ENABLE_JIT_PROFILING)
    # XXX: the profiling interface will still be built and present
    add_definitions(-DDNNL_ENABLE_JIT_PROFILING=0)
    # Don't enable support for linux_perf and VTune Amplifier
    list(REMOVE_ITEM SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/jitprofiling/jitprofiling.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/linux_perf/linux_perf.cpp"
        )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # to make computations more stable and to align the jitted code
    # with the reference one use precise division and square root
    # by default
    file(GLOB FILES_REQUIRED_PREC_SQRT
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp
        )
    file(GLOB FILES_REQUIRED_PREC_DIV
        ${CMAKE_CURRENT_SOURCE_DIR}/*resampling*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp
        )
    if(WIN32)
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "/Qprec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "/Qprec-div")
    else()
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "-prec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "-prec-div")
    endif()
endif()

# remove optimizations of files that don't need them for faster build times.
file(GLOB FILES_WITHOUT_OPT
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*/*_kern_autogen.cpp)
if(MSVC)
    set_source_files_properties(${FILES_WITHOUT_OPT}
        PROPERTIES COMPILE_FLAGS "/Od")
else()
    set_source_files_properties(${FILES_WITHOUT_OPT}
        PROPERTIES COMPILE_FLAGS "-O0 -U_FORTIFY_SOURCE")
endif()

set(OBJ_LIB ${LIB_NAME}_cpu_x64)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
