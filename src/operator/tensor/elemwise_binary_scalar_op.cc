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
MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_plus_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
.add_alias("_PlusScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_minus_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
.add_alias("_MinusScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rminus_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rminus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"})
.add_alias("_RMinusScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_mul_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_mul_scalar"})
.add_alias("_MulScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_div_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::div>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_div_scalar"})
.add_alias("_DivScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rdiv_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rdiv>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rdiv_scalar"})
.add_alias("_RDivScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_rdiv_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::rdiv_grad>);

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

}  // namespace op
}  // namespace mxnet
