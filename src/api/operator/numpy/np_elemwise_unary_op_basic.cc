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
 * \file np_elemwise_broadcast_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_elemwise_unary_op_basic.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../ufunc_helper.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.sin")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  const nnvm::Op* op = Op::Get("_npi_sin");
  UFuncHelper(args, ret, op);
});

}  // namespace mxnet
