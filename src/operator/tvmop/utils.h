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
 * Copyright (c) 2017 by Contributors
 * \file utils.h
 * \brief
 * \author Haozheng Fan
*/
#ifndef MXNET_OPERATOR_TVMOP_UTILS_H_
#define MXNET_OPERATOR_TVMOP_UTILS_H_

#if MXNET_USE_TVM_OP
#include <mxnet/op_attr_types.h>
#include <string>

namespace tvm {
namespace runtime {

template<::mxnet::OpReqType req>
std::string set_req();

template<> inline
std::string set_req<::mxnet::kWriteTo>() {
  return "req_kWriteTo";
}

template<> inline
std::string set_req<::mxnet::kAddTo>() {
  return "req_kAddTo";
}

}  // namespace runtime
}  // namespace tvm

#endif  // MXNET_USE_TVM_OP
#endif  // MXNET_OPERATOR_TVMOP_UTILS_H_
