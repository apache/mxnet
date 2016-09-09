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
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::plus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_copy"})
.add_alias("_PlusScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_minus_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::minus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_copy"})
.add_alias("_MinusScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rminus_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rminus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"negative"})
.add_alias("_RMinusScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_mul_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::mul>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_mul_scalar"})
.add_alias("_MulScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_div_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::div>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_div_scalar"})
.add_alias("_DivScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rdiv_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rdiv>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_rdiv_scalar"})
.add_alias("_RDivScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_rdiv_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::rdiv_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_maximum_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::maximum>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_maximum_scalar"})
.add_alias("_MaximumScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_maximum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::maximum_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_minimum_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::minimum>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_minimum_scalar"})
.add_alias("_MinimumScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_minimum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::minimum_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_power_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::power>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_power_scalar"})
.add_alias("_PowerScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_power_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::power_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rpower_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rpower>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseOut{"_backward_rpower_scalar"})
.add_alias("_RPowerScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_rpower_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::rpower_grad>);

}  // namespace op
}  // namespace mxnet
