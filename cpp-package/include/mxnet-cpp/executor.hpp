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

/*!
 * \file executor.hpp
 * \brief implementation of the executor
 * \author Zhang Chen, Chuntao Hong
 */

#ifndef MXNET_CPP_EXECUTOR_HPP_
#define MXNET_CPP_EXECUTOR_HPP_

#include <vector>
#include <map>
#include <string>
#include "mxnet-cpp/executor.h"
#include "mxnet-cpp/optimizer.h"


namespace mxnet {
namespace cpp {
inline Executor::Executor(const Symbol &symbol, Context context,
                          const std::vector<NDArray> &arg_arrays,
                          const std::vector<NDArray> &grad_arrays,
                          const std::vector<OpReqType> &grad_reqs,
                          const std::vector<NDArray> &aux_arrays,
                          const std::map<std::string, Context> &group_to_ctx,
                          Executor *shared_exec) {
  this->arg_arrays = arg_arrays;
  this->grad_arrays = grad_arrays;
  this->aux_arrays = aux_arrays;
  this->symbol_ = symbol;
  this->device_type = context.GetDeviceType();
  this->device_id = context.GetDeviceId();

  std::vector<NDArrayHandle> arg_handles;
  std::vector<NDArrayHandle> grad_handles;
  std::vector<NDArrayHandle> aux_handles;

  for (const auto &array : arg_arrays) {
    arg_handles.push_back(array.GetHandle());
  }
  for (const auto &array : grad_arrays) {
    grad_handles.push_back(array.GetHandle());
  }

  this->require_grad = false;
  std::vector<mx_uint> grad_reqs_uint;
  for (auto s : grad_reqs) {
    if (s != OpReqType::kNullOp) {
      this->require_grad = true;
    }
    grad_reqs_uint.push_back(s);
  }
  CHECK_EQ(MXAutogradMarkVariables(arg_handles.size(), arg_handles.data(),
                                   grad_reqs_uint.data(), grad_handles.data()),0);
  // std::vector<const char *> map_keys;
  // std::vector<int> dev_types, dev_ids;
  // for (const auto &s : group_to_ctx) {
  //   map_keys.push_back(s.first.c_str());
  //   dev_types.push_back(s.second.GetDeviceType());
  //   dev_ids.push_back(s.second.GetDeviceId());
  // }

  CHECK_EQ(MXCreateCachedOp(symbol.GetHandle(), 0, nullptr, nullptr, &handle_, false), 0);
}

inline mx_uint GradType2Int(OpReqType t) {
  if (t == OpReqType::kNullOp) {
    return 0;
  } else if (t == OpReqType::kWriteTo) {
    return 1;
  } else if (t == OpReqType::kWriteInplace) {
    return 2;
  } else if (t == OpReqType::kAddTo) {
    return 3;
  } else {
    LOG(FATAL) << "unknown grad type " << t;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_EXECUTOR_HPP_
