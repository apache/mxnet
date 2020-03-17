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
 * \file np_where_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_where_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/numpy/np_where_op-inl.h"

namespace mxnet {

inline static bool isScalar(const runtime::MXNetArgValue& arg) {
  return arg.type_code() == kDLInt ||
         arg.type_code() == kDLUInt ||
         arg.type_code() == kDLFloat;
}

inline static void _npi_where(runtime::MXNetArgs args,
                              runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_where");
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  int num_inputs = 3;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(),
                       args[1].operator mxnet::NDArray*(),
                       args[2].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

inline static void _npi_where_scalar1(runtime::MXNetArgs args,
                                      runtime::MXNetRetValue* ret,
                                      bool isl) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = isl ? Op::Get("_npi_where_lscalar") : Op::Get("_npi_where_rscalar");
  op::NumpyWhereScalarParam param;
  param.scalar = isl ? args[1].operator double() : args[2].operator double();
  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::NumpyWhereScalarParam>(&attrs);
  int num_inputs = 2;
  int num_outputs = 0;
  NDArray* inputs[] =
    {args[0].operator mxnet::NDArray*(),
     isl ? args[2].operator mxnet::NDArray*() : args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

inline static void _npi_where_scalar2(runtime::MXNetArgs args,
                                      runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_where_scalar2");
  op::NumpyWhereScalar2Param param;
  nnvm::NodeAttrs attrs;
  param.x = args[1].operator double();
  param.x = args[2].operator double();
  attrs.op = op;
  attrs.parsed = param;
  SetAttrDict<op::NumpyWhereScalar2Param>(&attrs);
  int num_inputs = 1;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

MXNET_REGISTER_API("_npi.where")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  if (isScalar(args[1]) && isScalar(args[2])) {
    _npi_where_scalar2(args, ret);
  } else if (!isScalar(args[1]) && !isScalar(args[2])) {
    _npi_where(args, ret);
  } else {
    _npi_where_scalar1(args, ret, isScalar(args[1]));
  }
});

}  // namespace mxnet
