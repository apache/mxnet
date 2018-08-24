# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Determines whether hardware and compiler support F16C 
# instruction set
#
# The following are set after configuration is done:
#  SUPPORT_F16C

if(AUTO_DETECT_F16_CMAKE_INCLUDED)
  return()
endif()
set(AUTO_DETECT_F16_CMAKE_INCLUDED True)

set(SUPPORT_F16C False)
if(MSVC)
    message("F16C instruction set is not yet supported for MSVC")
    return()
endif()
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORT_MF16C)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(COMMAND cat /proc/cpuinfo
            COMMAND grep flags
            COMMAND grep f16c
            OUTPUT_VARIABLE CPU_SUPPORT_F16C)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    execute_process(COMMAND sysctl -a
            COMMAND grep machdep.cpu.features
            COMMAND grep F16C
            OUTPUT_VARIABLE CPU_SUPPORT_F16C)
endif()
if(NOT CPU_SUPPORT_F16C)
    message("CPU does not support F16C instructions")
    return()
endif()
if(CPU_SUPPORT_F16C AND COMPILER_SUPPORT_MF16C)	
    set(SUPPORT_F16C TRUE)
endif()
