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
 * \file np_eig.cc
 * \brief Implementation of the API of functions in src/operator/numpy/linalg/np_eig.cc
 */

#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../../utils.h"
#include "../../../../operator/numpy/linalg/np_eig-inl.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.eig")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_eig");
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ADT(0, {NDArrayHandle(ndoutputs[0]),
                 NDArrayHandle(ndoutputs[1])});
});

MXNET_REGISTER_API("_npi.eigh")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_eigh");
  nnvm::NodeAttrs attrs;
  op::EighParam param;
  param.UPLO = *((args[1].operator std::string()).c_str());
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::EighParam>(&attrs);
  int num_inputs = 1;
  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ADT(0, {NDArrayHandle(ndoutputs[0]),
                 NDArrayHandle(ndoutputs[1])});
});

}  // namespace mxnet
