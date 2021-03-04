#*******************************************************************************
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
#*******************************************************************************

# This allows compiling other arm, non-ACL source files
if (DNNL_AARCH64_USE_ACL)
    file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/acl_*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/acl_*.[ch]pp
        )

    set(OBJ_LIB ${LIB_NAME}_cpu_aarch64)
    add_library(${OBJ_LIB} OBJECT ${SOURCES})
    set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
        $<TARGET_OBJECTS:${OBJ_LIB}>)
endif()
