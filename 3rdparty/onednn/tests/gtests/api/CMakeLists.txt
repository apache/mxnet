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

set(TEST_EXE test_api)

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
list(APPEND TEST_SOURCES ${MAIN_SRC_GTEST})

register_exe(${TEST_EXE} "${TEST_SOURCES}" "test" "dnnl_gtest")

# Create DPC++ USM target.
if(DNNL_SYCL_DPCPP)
    register_exe(${TEST_EXE}_buffer "${TEST_SOURCES}" "test" "dnnl_gtest")
    target_compile_definitions(${TEST_EXE}_buffer PUBLIC -DTEST_DNNL_DPCPP_BUFFER)
endif()
