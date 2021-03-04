#===============================================================================
# Copyright 2019-2020 Intel Corporation
# Copyright 2020 Arm Ltd. and affiliates
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

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp)
file(GLOB_RECURSE SOURCES_EXTRA
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn/*.[ch]pp
    )

foreach(SOURCE_FILE ${SOURCES_EXTRA})
    list(APPEND SOURCES "${SOURCE_FILE}")
endforeach()

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
        ${CMAKE_CURRENT_SOURCE_DIR}/ref_eltwise.cpp
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

if(MSVC AND (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel"))
    file(GLOB FILES_REQUIRED_BIGOBJ
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_reorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_convolution_list.cpp)
    set_source_files_properties(${FILES_REQUIRED_BIGOBJ}
        PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

if(DNNL_SYCL_DPCPP)
    set(FILES_WITHNO_OPT)
    foreach(src in ${SOURCES})
        string(REGEX MATCH ".*jit.*kern.cpp" match ${src})
        if(match)
            list(APPEND FILES_WITHNO_OPT ${src})
        endif()
    endforeach()

    set_source_files_properties(${FILES_WITHNO_OPT}
        PROPERTIES COMPILE_FLAGS "-O0")
endif()

set(OBJ_LIB ${LIB_NAME}_cpu)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

if (DNNL_TARGET_ARCH STREQUAL "X64")
    add_subdirectory(x64)
endif()
if (DNNL_TARGET_ARCH STREQUAL "AARCH64")
    add_subdirectory(aarch64)
endif()
