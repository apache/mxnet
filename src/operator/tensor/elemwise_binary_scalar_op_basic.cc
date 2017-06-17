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

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_mod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod_scalar"})
.add_alias("_ModScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_mod_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::mod_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rmod_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rmod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rmod_scalar"})
.add_alias("_RModScalar");

MXNET_OPERATOR_REGISTER_BINARY(_backward_rmod_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::rmod_grad>);

}  // namespace op
}  // namespace mxnet
