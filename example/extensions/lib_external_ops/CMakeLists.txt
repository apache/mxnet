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

# specify CXX sources
FILE(GLOB CXX_SRCS
  # Required files
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/lib_api.cc
  # Your custom files
  ${CMAKE_CURRENT_SOURCE_DIR}/init_lib.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/min_ex.cc
  )

# create library & set libraries
add_library(external_lib SHARED ${CXX_SRCS})
target_link_libraries(external_lib PUBLIC mxnet)

if(USE_CUDA)
  # specify GPU sources (optional)
  FILE(GLOB CU_SRCS "*.cu")
  target_sources(external_lib PUBLIC ${CU_SRCS})
endif(USE_CUDA)
