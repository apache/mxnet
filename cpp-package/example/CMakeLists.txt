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

# Explicitly set GENERATED property https://gitlab.kitware.com/cmake/cmake/issues/18399
set_property(SOURCE ${CMAKE_CURRENT_LIST_DIR}/../include/mxnet-cpp/op.h PROPERTY GENERATED 1)

add_executable(test_regress_label test_regress_label.cpp)
target_link_libraries(test_regress_label mxnet_cpp)

add_executable(lenet lenet.cpp)
target_link_libraries(lenet mxnet_cpp)

add_executable(lenet_with_mxdataiter lenet_with_mxdataiter.cpp)
target_link_libraries(lenet_with_mxdataiter mxnet_cpp)

add_executable(alexnet alexnet.cpp)
target_link_libraries(alexnet mxnet_cpp)

add_executable(charRNN charRNN.cpp)
target_link_libraries(charRNN mxnet_cpp)

add_executable(googlenet googlenet.cpp)
target_link_libraries(googlenet mxnet_cpp)

add_executable(inception_bn inception_bn.cpp)
target_link_libraries(inception_bn mxnet_cpp)

add_executable(mlp mlp.cpp)
target_link_libraries(mlp mxnet_cpp)

add_executable(mlp_cpu mlp_cpu.cpp)
target_link_libraries(mlp_cpu mxnet_cpp)

add_executable(mlp_gpu mlp_gpu.cpp)
target_link_libraries(mlp_gpu mxnet_cpp)

add_executable(resnet resnet.cpp)
target_link_libraries(resnet mxnet_cpp)


if(MSVC)
  add_custom_target(cpp_package_deploy_library ALL
    DEPENDS mxnet
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mxnet> $<TARGET_FILE_DIR:mlp>)
endif()
