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
 * \file np_bincount_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_bincount_op.cc
 */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/numpy/np_bincount_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.bincount")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_bincount");
  nnvm::NodeAttrs attrs;
  op::NumpyBincountParam param;

  int num_outputs = 0;
  if (args[1].type_code() == kNull) {
    param.minlength = args[2].operator int64_t();
    param.has_weights = false;
    NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
    int num_inputs = 1;
    attrs.parsed = param;
    attrs.op = op;
    SetAttrDict<op::NumpyBincountParam>(&attrs);
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  } else {
    param.minlength = args[2].operator int64_t();
    param.has_weights = true;
    NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
    int num_inputs = 2;
    attrs.parsed = param;
    attrs.op = op;
    SetAttrDict<op::NumpyBincountParam>(&attrs);
    auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
    }
});

}  // namespace mxnet
