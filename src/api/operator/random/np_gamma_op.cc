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
 * \file np_gamma_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/random/np_gamma_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <vector>
#include "../utils.h"
#include "../../../operator/numpy/random/np_gamma_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.gamma")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_gamma");
  nnvm::NodeAttrs attrs;
  op::NumpyGammaParam param;
  int num_inputs = 0;
  std::vector<NDArray*> inputs;
  if (args[0].type_code() == kDLFloat || args[0].type_code() == kDLInt) {
    if (args[0].type_code() == kNull) {
      param.shape = dmlc::nullopt;
    } else {
      param.shape = args[0].operator double();
    }
    if (args[1].type_code() == kDLFloat || args[1].type_code() == kDLInt) {
      // both 'shape' and 'scale' are numeric types
      num_inputs = 0;
      if (args[1].type_code() == kNull) {
        param.scale = dmlc::nullopt;
      } else {
        param.scale = args[1].operator double();
      }
    } else {
      // 'shape' is numeric types but 'scale' is not
      num_inputs = 1;
      param.scale = dmlc::nullopt;
      inputs.push_back(args[1].operator mxnet::NDArray*());
    }
  } else {
    param.shape = dmlc::nullopt;
    inputs.push_back(args[0].operator mxnet::NDArray*());
    if (args[1].type_code() == kDLFloat || args[1].type_code() == kDLInt) {
      // 'shape' is not numeric types but 'scale' is numeric types
      num_inputs = 1;
      if (args[1].type_code() == kNull) {
        param.scale = dmlc::nullopt;
      } else {
        param.scale = args[1].operator double();
      }
    } else {
      // nither 'shape' or 'scale' is numeric types
      num_inputs = 2;
      param.scale = dmlc::nullopt;
      inputs.push_back(args[1].operator mxnet::NDArray*());
    }
  }
  if (args[2].type_code() == kNull) {
    param.size = dmlc::optional<mxnet::Tuple<int>>();
  } else if (args[2].type_code() == kDLInt ||
             args[2].type_code() == kDLFloat) {
    param.size = Tuple<int>(1, args[2].operator int64_t());
  } else {
    param.size = Tuple<int>(args[2].operator ObjectRef());
  }
  if (args[4].type_code() == kNull) {
    param.dtype = mshadow::kFloat32;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[4].operator std::string());
  }
  NDArray* out = args[5].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  attrs.parsed = std::move(param);
  attrs.op = op;
  if (args[3].type_code() != kNull) {
    attrs.dict["ctx"] = args[3].operator std::string();
  }
  SetAttrDict<op::NumpyGammaParam>(&attrs);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs.data(), &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(5);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

}  // namespace mxnet
