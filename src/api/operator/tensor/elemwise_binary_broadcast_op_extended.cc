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
 * \file elemwise_binary_broadcast_op_extended.cc
 * \brief Implementation of the API of functions in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../ufunc_helper.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.maximum")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_maximum");
  const nnvm::Op* op_scalar = Op::Get("_npi_maximum_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.minimum")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_minimum");
  const nnvm::Op* op_scalar = Op::Get("_npi_minimum_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

}  // namespace mxnet
