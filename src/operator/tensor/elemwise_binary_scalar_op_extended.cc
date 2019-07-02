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
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op_extended.cc
 * \brief CPU Implementation of extended binary scalar functions.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_maximum_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::maximum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_maximum_scalar"})
.add_alias("_MaximumScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_maximum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::ge>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_minimum_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::minimum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_minimum_scalar"})
.add_alias("_MinimumScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_minimum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::le>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_power_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_power_scalar"})
.add_alias("_PowerScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_power_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<
  cpu, mshadow_op::power_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rpower_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<
  cpu, mshadow_op::rpower>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_rpower_scalar"})
.add_alias("_RPowerScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_rpower_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<
  cpu, mshadow_op::rpower_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_hypot_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<
  cpu, mshadow_op::hypot>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_hypot_scalar" })
.add_alias("_HypotScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_hypot_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs *attrs) { attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<
  cpu, mshadow_op::hypot_grad_left>);

NNVM_REGISTER_OP(smooth_l1)
.add_alias("_npx_smooth_l1")
.describe(R"code(Calculate Smooth L1 Loss(lhs, scalar) by summing

.. math::

    f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}

where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.

Example::

  smooth_l1([1, 2, 3, 4]) = [0.5, 1.5, 2.5, 3.5]
  smooth_l1([1, 2, 3, 4], scalar=1) = [0.5, 1.5, 2.5, 3.5]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
    if (attrs->dict.find("scalar") != attrs->dict.end()) {
      attrs->parsed = std::stod(attrs->dict["scalar"]);
    } else {
      attrs->parsed = 1.0;
    }
  })
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs& attrs){
                                  return std::vector<std::pair<int, int> >{{0, 0}};
                                })
.add_argument("data", "NDArray-or-Symbol", "source input")
.add_argument("scalar", "float", "scalar input")
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::smooth_l1_loss>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_smooth_l1" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_smooth_l1)
  .set_attr_parser([](NodeAttrs *attrs) {
      if (attrs->dict.find("scalar") != attrs->dict.end()) {
        attrs->parsed = std::stod(attrs->dict["scalar"]);
      } else {
        attrs->parsed = 1.0;
      }
})
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryScalarOp::Backward<cpu, mshadow_op::smooth_l1_gradient>);

}  // namespace op
}  // namespace mxnet
