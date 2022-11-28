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

# ---[ Google Test
if(NOT GTEST_ROOT)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
    set(GTEST_FOUND ON)
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
    set(GTEST_FOUND ON)
  else()
    find_package(GTest)
  endif()
endif()

# FIXME MSVC unit test linking issue
if(GTEST_FOUND AND NOT MSVC)
  enable_testing()

  file(GLOB_RECURSE UNIT_TEST_SOURCE "cpp/*.cc" "cpp/*.h")

  include_directories(${GTEST_INCLUDE_DIR})
  include_directories(cpp/include)

  if(NOT PRIVATE_RUNTIME_DIR)
   set(PRIVATE_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()

  add_executable(${PROJECT_NAME}_unit_tests ${UNIT_TEST_SOURCE})
  set_property(TARGET ${PROJECT_NAME}_unit_tests
               PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PRIVATE_RUNTIME_DIR})

  target_link_libraries(${PROJECT_NAME}_unit_tests
    ${GTEST_LIBRARY}
    dmlc
    ${nnvm_LINKER_LIBS}
    ${mxnet_LINKER_LIBS}
    mxnet
    ${pslite_LINKER_LIBS}
    )

  add_test(AllTestsIn${PROJECT_NAME}UnitTests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}_unit_tests)
else()
  message(STATUS "Google Test not found")
endif()

