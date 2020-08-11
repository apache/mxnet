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
 * \file np_ordering_op.cc
 * \brief Implementation of the API of functions in src/operator/tensor/ordering_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/ordering_op-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.sort")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_sort");
  nnvm::NodeAttrs attrs;
  op::SortParam param;

  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[1].operator int();
  }
  param.is_ascend = true;

  attrs.parsed = std::move(param);
  attrs.op = op;

  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};

  int num_outputs = 0;
  SetAttrDict<op::SortParam>(&attrs);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

MXNET_REGISTER_API("_npi.argsort")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_argsort");
  nnvm::NodeAttrs attrs;
  op::ArgSortParam param;

  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    param.axis = args[1].operator int();
  }
  param.is_ascend = true;
  if (args[3].type_code() == kNull) {
    param.dtype = mshadow::kFloat32;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[3].operator std::string());
  }

  attrs.parsed = std::move(param);
  attrs.op = op;

  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};

  int num_outputs = 0;
  SetAttrDict<op::ArgSortParam>(&attrs);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
});

}  // namespace mxnet
