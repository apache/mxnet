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
 * \file np_tensordot_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_tensordot_op.cc
 */
#include <mxnet/api_registry.h>
#include "../utils.h"
#include "../../../operator/numpy/np_tensordot_op-inl.h"

namespace mxnet {

inline static void _npi_tensordot_int_axes(runtime::MXNetArgs args,
                                           runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_tensordot_int_axes");
  op::TensordotIntAxesParam param;
  nnvm::NodeAttrs attrs;
  param.axes = args[2].operator int();
  attrs.op = op;
  // we directly copy TensordotIntAxesParam, which is trivially-copyable
  attrs.parsed = param;
  SetAttrDict<op::TensordotIntAxesParam>(&attrs);
  int num_outputs = 0;
  int num_inputs = 2;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

inline static void _npi_tensordot(runtime::MXNetArgs args,
                                  runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_tensordot");
  op::TensordotParam param;
  nnvm::NodeAttrs attrs;
  ADT adt = Downcast<ADT, ObjectRef>(args[2].operator ObjectRef());
  if (const IntegerObj* lop = adt[0].as<IntegerObj>()) {
    // axes is a tuple of int, like axes=(0, 1)
    param.a_axes_summed = Tuple<int>(1, lop->value);
    param.b_axes_summed = Tuple<int>(1, Downcast<Integer, ObjectRef>(adt[1])->value);
  } else {
    // axes is a tuple of tuples of int, like axes=((0, 1), (1, 0))
    param.a_axes_summed = Tuple<int>(adt[0]);
    param.b_axes_summed = Tuple<int>(adt[1]);
  }
  attrs.op = op;
  attrs.parsed = std::move(param);
  SetAttrDict<op::TensordotParam>(&attrs);
  int num_outputs = 0;
  int num_inputs = 2;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

MXNET_REGISTER_API("_npi.tensordot")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  if (args[2].type_code() == kDLInt) {
    _npi_tensordot_int_axes(args, ret);
  } else {
    _npi_tensordot(args, ret);
  }
});

}  // namespace mxnet
