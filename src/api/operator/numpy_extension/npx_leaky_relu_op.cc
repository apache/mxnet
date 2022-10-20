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
 * \file npx_leaky_relu_op.cc
 * \brief Implementation of the API of functions in
 * src/operator/numpy_extension/npx_leaky_relu_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/leaky_relu-inl.h"

namespace mxnet {

inline int String2ActType(const std::string& s) {
  using namespace op;
  if (s == "rrelu") {
    return leakyrelu::kRReLU;
  } else if (s == "leaky") {
    return leakyrelu::kLeakyReLU;
  } else if (s == "prelu") {
    return leakyrelu::kPReLU;
  } else if (s == "elu") {
    return leakyrelu::kELU;
  } else if (s == "selu") {
    return leakyrelu::kSELU;
  } else if (s == "gelu" || s == "gelu_erf") {
    return leakyrelu::kGELU_ERF;
  } else if (s == "gelu_tanh") {
    return leakyrelu::kGELU_TANH;
  } else {
    LOG(FATAL) << "unknown activation type " << s;
  }
  LOG(FATAL) << "should not reach here ";
  return 0;
}

MXNET_REGISTER_API("_npx.leaky_relu")
    .set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
      using namespace runtime;
      nnvm::NodeAttrs attrs;
      const nnvm::Op* op = Op::Get("_npx_leaky_relu");
      op::LeakyReLUParam param = {};
      int args_size = args.size();
      // act_type
      param.act_type = String2ActType(args[args_size - 4].operator std::string());
      // inputs
      int num_inputs  = param.act_type == op::leakyrelu::kPReLU ? 2 : 1;
      int num_outputs = param.act_type == op::leakyrelu::kPReLU ? 2 : 1;
      std::vector<NDArray*> inputs;
      inputs.reserve(num_inputs);
      for (int i = 0; i < num_inputs; ++i) {
        inputs.push_back(args[i].operator mxnet::NDArray*());
      }
      // slope
      if (args[args_size - 3].type_code() == kNull) {
        param.slope = 0.25f;
      } else {
        param.slope = args[args_size - 3].operator double();
      }
      // lower_bound
      if (args[args_size - 2].type_code() == kNull) {
        param.lower_bound = 0.125f;
      } else {
        param.lower_bound = args[args_size - 2].operator double();
      }
      // upper_bound
      if (args[args_size - 1].type_code() == kNull) {
        param.upper_bound = 0.334f;
      } else {
        param.upper_bound = args[args_size - 1].operator double();
      }
      attrs.parsed = param;
      attrs.op     = op;
      SetAttrDict<op::LeakyReLUParam>(&attrs);

      auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, nullptr);
      if (num_outputs == 1) {
        *ret = ndoutputs[0];
      } else {
        std::vector<NDArrayHandle> ndarray_handles;
        ndarray_handles.reserve(num_outputs);
        for (int i = 0; i < num_outputs; ++i) {
          ndarray_handles.emplace_back(ndoutputs[i]);
        }
        *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
      }
    });

}  // namespace mxnet
