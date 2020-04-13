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

#define MXNET_REGISTER_UNARY_API(op_name)                                      \
MXNET_REGISTER_API("_npi." #op_name)                                           \
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {           \
  const nnvm::Op* op = Op::Get("_npi_" #op_name);                              \
  UFuncHelper(args, ret, op);                                                  \
})

MXNET_REGISTER_UNARY_API(negative);
MXNET_REGISTER_UNARY_API(reciprocal);
MXNET_REGISTER_UNARY_API(abs);
MXNET_REGISTER_UNARY_API(sign);
MXNET_REGISTER_UNARY_API(rint);
MXNET_REGISTER_UNARY_API(ceil);
MXNET_REGISTER_UNARY_API(floor);
MXNET_REGISTER_UNARY_API(bitwise_not);
MXNET_REGISTER_UNARY_API(trunc);
MXNET_REGISTER_UNARY_API(fix);
MXNET_REGISTER_UNARY_API(square);
MXNET_REGISTER_UNARY_API(sqrt);
MXNET_REGISTER_UNARY_API(cbrt);
MXNET_REGISTER_UNARY_API(exp);
MXNET_REGISTER_UNARY_API(log);
MXNET_REGISTER_UNARY_API(log10);
MXNET_REGISTER_UNARY_API(log2);
MXNET_REGISTER_UNARY_API(log1p);
MXNET_REGISTER_UNARY_API(expm1);
MXNET_REGISTER_UNARY_API(logical_not);
MXNET_REGISTER_UNARY_API(isnan);
MXNET_REGISTER_UNARY_API(isinf);
MXNET_REGISTER_UNARY_API(isposinf);
MXNET_REGISTER_UNARY_API(isneginf);
MXNET_REGISTER_UNARY_API(isfinite);
MXNET_REGISTER_UNARY_API(sin);
MXNET_REGISTER_UNARY_API(cos);
MXNET_REGISTER_UNARY_API(tan);
MXNET_REGISTER_UNARY_API(arcsin);
MXNET_REGISTER_UNARY_API(arccos);
MXNET_REGISTER_UNARY_API(arctan);
MXNET_REGISTER_UNARY_API(degrees);
MXNET_REGISTER_UNARY_API(radians);
#if MXNET_USE_TVM_OP
MXNET_REGISTER_UNARY_API(rad2deg);  // from src/operator/contrib/tvmop/ufunc.cc
MXNET_REGISTER_UNARY_API(deg2rad);  // from src/operator/contrib/tvmop/ufunc.cc
#else  // MXNET_USE_TVM_OP
MXNET_REGISTER_API("_npi.rad2deg")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  const nnvm::Op* op = Op::Get("_npi_degrees");
  UFuncHelper(args, ret, op);
});
MXNET_REGISTER_API("_npi.deg2rad")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  const nnvm::Op* op = Op::Get("_npi_radians");
  UFuncHelper(args, ret, op);
});
#endif  // MXNET_USE_TVM_OP
MXNET_REGISTER_UNARY_API(sinh);
MXNET_REGISTER_UNARY_API(cosh);
MXNET_REGISTER_UNARY_API(tanh);
MXNET_REGISTER_UNARY_API(arcsinh);
MXNET_REGISTER_UNARY_API(arccosh);
MXNET_REGISTER_UNARY_API(arctanh);

}  // namespace mxnet
