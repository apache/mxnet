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
 * \brief Implementation of the API of functions in src/operator/numpy/np_elemwise_broadcast_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../ufunc_helper.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.add")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_add");
  const nnvm::Op* op_scalar = Op::Get("_npi_add_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.subtract")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_subtract");
  const nnvm::Op* op_scalar = Op::Get("_npi_subtract_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_rsubtract_scalar");
  UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.multiply")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_multiply");
  const nnvm::Op* op_scalar = Op::Get("_npi_multiply_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.true_divide")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_true_divide");
  const nnvm::Op* op_scalar = Op::Get("_npi_true_divide_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_rtrue_divide_scalar");
  UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.mod")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_mod");
  const nnvm::Op* op_scalar = Op::Get("_npi_mod_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_rmod_scalar");
  UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.power")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_power");
  const nnvm::Op* op_scalar = Op::Get("_npi_power_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_rpower_scalar");
  UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.lcm")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_lcm");
  const nnvm::Op* op_scalar = Op::Get("_npi_lcm_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.logical_and")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_logical_and");
  const nnvm::Op* op_scalar = Op::Get("_npi_logical_and_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.logical_or")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_logical_or");
  const nnvm::Op* op_scalar = Op::Get("_npi_logical_or_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.logical_xor")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_logical_xor");
  const nnvm::Op* op_scalar = Op::Get("_npi_logical_xor_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.bitwise_or")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_bitwise_or");
  const nnvm::Op* op_scalar = Op::Get("_npi_bitwise_or_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.bitwise_xor")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_bitwise_xor");
  const nnvm::Op* op_scalar = Op::Get("_npi_bitwise_xor_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.bitwise_and")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_bitwise_and");
  const nnvm::Op* op_scalar = Op::Get("_npi_bitwise_and_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.copysign")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_copysign");
  const nnvm::Op* op_scalar = Op::Get("_npi_copysign_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_rcopysign_scalar");
  UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.arctan2")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_arctan2");
  const nnvm::Op* op_scalar = Op::Get("_npi_arctan2_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_rarctan2_scalar");
  UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

MXNET_REGISTER_API("_npi.hypot")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_hypot");
  const nnvm::Op* op_scalar = Op::Get("_npi_hypot_scalar");
  UFuncHelper(args, ret, op, op_scalar, nullptr);
});

MXNET_REGISTER_API("_npi.ldexp")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_ldexp");
  const nnvm::Op* op_scalar = Op::Get("_npi_ldexp_scalar");
  const nnvm::Op* op_rscalar = Op::Get("_npi_rldexp_scalar");
  UFuncHelper(args, ret, op, op_scalar, op_rscalar);
});

}  // namespace mxnet
