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
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_PlusScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::plus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_minus_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::minus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_MinusScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::minus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rminus_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rminus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"negative"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_RMinusScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rminus>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"negative"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_mul_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::mul>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_mul_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_MulScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::mul>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_mul_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_div_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::div>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_div_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_DivScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow::op::div>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseNone{"_div_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rdiv_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rdiv>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_rdiv_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_RDivScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rdiv>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_rdiv_scalar"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_rdiv_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::rdiv_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_maximum_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::maximum>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_maximum_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_MaximumScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::maximum>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_maximum_scalar"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_maximum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::maximum_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_minimum_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::minimum>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_minimum_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_MinimumScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::minimum>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_minimum_scalar"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_minimum_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::minimum_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_power_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::power>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_power_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_PowerScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::power>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseIn{"_backward_power_scalar"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_power_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::power_grad>);

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_rpower_scalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rpower>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseOut{"_backward_rpower_scalar"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_RPowerScalar)
.attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::rpower>)
.attr<nnvm::FGradient>("FGradient", UnaryGradUseOut{"_backward_rpower_scalar"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_rpower_scalar)
.add_argument("scalar", "float", "scalar value")
.set_attr_parser([](NodeAttrs* attrs) {attrs->parsed = std::stod(attrs->dict["scalar"]);})
.attr<FCompute>("FCompute<cpu>", BinaryScalarBackward<cpu, mshadow_op::rpower_grad>);

}  // namespace op
}  // namespace mxnet
