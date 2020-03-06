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
 * \file np_choice_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_choice_op.cc
 */
#include "../../utils.h"
#include "../../../../operator/numpy/random/np_choice_op.h"
#include <algorithm>

namespace mxnet {

MXNET_REGISTER_API("_npi.choice")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_choice");
  nnvm::NodeAttrs attrs;
  op::NumpyChoiceParam param;

  if (args[0].type_code() == kDLInt) {
    param.a = 100;
  } else {
    param.a = 100;
  }

  // if (args[1].type_code() == kNull) {
  //   param.size = 10;
  // } else {
  //   param.size = 10;
  // }

  int num_input = 0;
  if (param.weighted) num_input += 1;
  if (!param.a.has_value()) num_input += 1;
  
  attrs.parsed = std::move(param);
  attrs.op = op;
  if (args[3].type_code() != kNull) {
    attrs.dict["ctx"] = args[3].operator std::string();
  }
  
  NDArray* out = args[5].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke<op::NumpyChoiceParam>(op, &attrs, 0, nullptr, &num_outputs, outputs);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
