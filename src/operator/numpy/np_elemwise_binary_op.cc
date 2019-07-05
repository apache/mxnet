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
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_binary_op.cc
 * \brief CPU Implementation for element-wise binary operators.
 */


#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {
    
bool NumpyBinaryScalarTypeInfer(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int itype = in_attrs->at(0);
  if (itype == -1) return false;
  auto is_float = [](const int dtype) {
    return dtype == mshadow::kFloat32 || dtype == mshadow::kFloat64 || dtype == mshadow::kFloat16;
  };
  CHECK(is_float(itype)) << "numpy binary scalar op currently only supports float dtype";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, itype);
  return true;
}

#define MXNET_OPERATOR_REGISTER_FOR_NP_BINARY_SCALAR(name)              \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr_parser([](NodeAttrs* attrs) {                           \
      attrs->parsed = std::stod(attrs->dict["scalar"]);             \
    })                                                              \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>) \
  .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryScalarTypeInfer)  \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "source input")        \
  .add_argument("scalar", "float", "scalar input")

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(ldexp)
.describe(R"code(
    ldexp(x1, x2, out=None)

    Returns x1 * 2**x2, element-wise.

    The mantissas `x1` and twos exponents `x2` are used to construct
    floating point numbers ``x1 * 2**x2``.

    Parameters
    ----------
    x1 : ndarray or scalar
        Array of multipliers.
    x2 : ndarray or scalar, int
        Array of twos exponents.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        The result of ``x1 * 2**x2``.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.
    Different from numpy, we allow x2 to be float besides int. 

    `ldexp` is useful as the inverse of `frexp`, if used by itself it is
    more clear to simply use the expression ``x1 * 2**x2``.

    Examples
    --------
    >>> np.ldexp(5, np.arange(4))
    array([  5.,  10.,  20.,  40.])
)code" ADD_FILELINE)
.add_alias("_npi_ldexp")  
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_ldexp"});

MXNET_OPERATOR_REGISTER_FOR_NP_BINARY_SCALAR(ldexp_scalar)
.add_alias("_npi_ldexp_scalar")  
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::ldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_ldexp_scalar"});

MXNET_OPERATOR_REGISTER_FOR_NP_BINARY_SCALAR(rldexp_scalar)
.add_alias("_npi_rldexp_scalar") 
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::rldexp>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rldexp_scalar"});

NNVM_REGISTER_OP(_backward_ldexp)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::ldexp_grad,
                                                              mshadow_op::ldexp_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_ldexp_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::ldexp_grad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_rldexp_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::rldexp_grad>);

}  // namespace op
}  // namespace mxnet
