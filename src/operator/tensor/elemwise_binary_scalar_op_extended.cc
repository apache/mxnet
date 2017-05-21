/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_maximum_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::maximum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_maximum_scalar"})
.add_alias("_MaximumScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_maximum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::ge>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_minimum_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::minimum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_minimum_scalar"})
.add_alias("_MinimumScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_minimum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::le>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_power_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_power_scalar"})
.add_alias("_PowerScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_power_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::power_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rpower_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rpower>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_rpower_scalar"})
.add_alias("_RPowerScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_rpower_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::rpower_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_hypot_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::hypot>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_hypot_scalar" })
.add_alias("_HypotScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_hypot_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::hypot_grad_left>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(smooth_l1)
.describe(R"code(Calculate Smooth L1 Loss(lhs, scalar) by summing

.. math::

    f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}

where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.

Example::

  a = mx.nd.array([1, 2, 3, 4]), sigma=1,
  smooth_l1(a, sigma) = [0.5, 1.5, 2.5, 3.5]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::smooth_l1_loss>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_smooth_l1" });

MXNET_OPERATOR_REGISTER_BINARY(_backward_smooth_l1)
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]); })
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::smooth_l1_gradient>);

}  // namespace op
}  // namespace mxnet
