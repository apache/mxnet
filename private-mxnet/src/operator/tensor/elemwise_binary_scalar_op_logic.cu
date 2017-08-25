/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(_not_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::ne>);

NNVM_REGISTER_OP(_greater_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::gt>);

NNVM_REGISTER_OP(_greater_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::ge>);

NNVM_REGISTER_OP(_lesser_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::lt>);

NNVM_REGISTER_OP(_lesser_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarCompute<gpu, mshadow_op::le>);

}  // namespace op
}  // namespace mxnet
