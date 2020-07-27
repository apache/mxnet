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
 * \file np_pinv.cc
 * \brief Implementation of the API of functions in src/operator/numpy/linalg/np_pinv.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../../utils.h"
#include "../../../../operator/numpy/linalg/np_pinv-inl.h"

namespace mxnet {

inline static void _npi_pinv(runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_pinv");
  op::PinvParam param;
  nnvm::NodeAttrs attrs;
  param.hermitian = args[2].operator bool();
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::PinvParam>(&attrs);
  int num_inputs = 2;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*(), args[1].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

inline static void _npi_pinv_scalar_rcond(runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_pinv_scalar_rcond");
  op::PinvScalarRcondParam param;
  nnvm::NodeAttrs attrs;
  param.rcond = args[1].operator double();
  param.hermitian = args[2].operator bool();
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::PinvScalarRcondParam>(&attrs);
  int num_inputs = 1;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
}

MXNET_REGISTER_API("_npi.pinv")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  if (args[1].type_code() == kDLFloat || args[1].type_code() == kDLInt) {
    _npi_pinv_scalar_rcond(args, ret);
  } else {
    _npi_pinv(args, ret);
  }
});

}  // namespace mxnet
