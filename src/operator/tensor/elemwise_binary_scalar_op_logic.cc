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

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::eq>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_EqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_not_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::ne>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_NotEqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_greater_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::gt>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_GreaterScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_greater_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::ge>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_GreaterEqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_lesser_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::lt>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LesserScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_lesser_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarCompute<cpu, mshadow_op::le>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LesserEqualScalar");

}  // namespace op
}  // namespace mxnet
