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

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

include("${PROJECT_SOURCE_DIR}/cmake/gen_gpu_kernel_list.cmake")

file(GLOB_RECURSE CL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cl)

set(kernel_list_templ "${PROJECT_SOURCE_DIR}/src/gpu/ocl/ocl_kernel_list.cpp.in")
set(kernel_list_src "${PROJECT_BINARY_DIR}/src/gpu/ocl/ocl_kernel_list.cpp")

gen_gpu_kernel_list(
    "${kernel_list_templ}"
    "${kernel_list_src}"
    "${CL_SOURCES}")

list(APPEND SOURCES ${kernel_list_src})

set(OBJ_LIB ${LIB_NAME}_gpu_ocl)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
