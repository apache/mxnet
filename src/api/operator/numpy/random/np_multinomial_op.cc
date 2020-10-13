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
 * \file np_multinomial_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/random/np_multinomial_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <vector>
#include "../../utils.h"
#include "../../../../operator/numpy/random/np_multinomial_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.multinomial")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_multinomial");
  nnvm::NodeAttrs attrs;
  op::NumpyMultinomialParam param;
  NDArray** inputs = new NDArray*[1]();
  int num_inputs = 0;

  // parse int
  param.n = args[0].operator int();

  // parse pvals
  if (args[1].type_code() == kNull) {
    param.pvals = dmlc::nullopt;
  } else if (args[1].type_code() == kNDArrayHandle) {
    param.pvals = dmlc::nullopt;
    inputs[0] = args[1].operator mxnet::NDArray*();
    num_inputs = 1;
  } else {
    param.pvals = Obj2Tuple<double, Float>(args[1].operator ObjectRef());
  }

  // parse size
  if (args[2].type_code() == kNull) {
    param.size = dmlc::nullopt;
  } else {
    if (args[2].type_code() == kDLInt) {
      param.size = mxnet::Tuple<index_t>(1, args[2].operator int64_t());
    } else {
      param.size = mxnet::Tuple<index_t>(args[2].operator ObjectRef());
    }
  }

  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyMultinomialParam>(&attrs);
  inputs = num_inputs == 0 ? nullptr : inputs;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
