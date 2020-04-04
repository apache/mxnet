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
 * \file np_unique_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_unique_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <vector>
#include "../utils.h"
#include "../../../operator/numpy/np_unique_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.unique")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_unique");
  nnvm::NodeAttrs attrs;
  op::NumpyUniqueParam param;
  // param
  param.return_index = args[1].operator bool();
  param.return_inverse = args[2].operator bool();
  param.return_counts = args[3].operator bool();
  if (args[4].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[4].operator int();
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyUniqueParam>(&attrs);
  // inputs
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  // outputs
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

}  // namespace mxnet
