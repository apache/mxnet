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

# Test oneDNN for embeddability
# by imitating a project that includes the library
#
# To test run:
# mkdir -p build && cd build && cmake .. && make -j && ./project_app

cmake_minimum_required(VERSION 2.8.11)

set(PROJECT_NAME "Project")

# force CPU runtime
set(DNNL_CPU_RUNTIME "SEQ" CACHE STRING "" FORCE)

# force GPU runtime
set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)

if(POLICY CMP0054)
    # Try to catch cases of double-expansion if CMake version is recent enough
    # to support a policy that controls it.
    set(NONE "some-value")
endif()

# include oneDNN
set(DNNL_DIR "../../..")
add_subdirectory(${DNNL_DIR} oneDNN)
include_directories(${DNNL_DIR}/include)

add_executable(project_app main.c)
target_link_libraries(project_app dnnl)
