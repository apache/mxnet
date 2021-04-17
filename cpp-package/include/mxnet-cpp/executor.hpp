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
                          bool require_grad,
                          const std::map<std::string, Context> &group_to_ctx,
                          Executor *shared_exec) {
  this->arg_arrays = arg_arrays;
  this->symbol_ = symbol;
  this->require_grad = require_grad;
  this->device_type = context.GetDeviceType();
  this->device_id = context.GetDeviceId();

  CHECK_EQ(MXCreateCachedOp(symbol.GetHandle(), 0, nullptr, nullptr, &handle_, false), 0);
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_EXECUTOR_HPP_
