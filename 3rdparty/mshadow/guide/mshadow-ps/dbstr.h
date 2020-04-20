/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once
#include <mshadow/tensor.h>
#include <sstream>

template<typename DType>
std::string dbstr(mshadow::Tensor<mshadow::cpu, 1, DType> ts) {
  std::stringstream ss;
  for (mshadow::index_t i = 0; i < ts.size(0); ++i)
    ss << ts[i] << " ";
  ss << "\n";
  return ss.str();
}

template<typename DType>
std::string dbstr(mshadow::Tensor<mshadow::cpu, 2, DType> ts) {
  std::stringstream ss;
  for (mshadow::index_t i = 0; i < ts.size(0); ++i) {
    for (mshadow::index_t j = 0; j < ts.size(1); ++j) {
      ss << ts[i][j] << " ";
    }
    ss << "\n";
  }
  ss << "\n";
  return ss.str();
}

template<typename DType>
std::string dbstr(mshadow::Tensor<mshadow::cpu, 3, DType> ts) {
  std::stringstream ss;
  for (mshadow::index_t i = 0; i < ts.size(0); ++i) {
    ss << dbstr(ts[i]) << "\n";
  }
  ss << "\n";
  return ss.str();
}
