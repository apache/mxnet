/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(elemwise_add)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_grad_add)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_backward_add)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseNoneWithHalf2<gpu, mshadow_op::identity,
                    mshadow_op::identity>);

NNVM_REGISTER_OP(elemwise_sub)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow::op::minus>);

NNVM_REGISTER_OP(_backward_sub)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseNoneWithHalf2<gpu, mshadow_op::identity,
                    mshadow_op::negation>);

NNVM_REGISTER_OP(elemwise_mul)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_backward_mul)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::right,
                    mshadow_op::left>);

NNVM_REGISTER_OP(elemwise_div)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_backward_div)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::div_grad,
                    mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(_mod)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_mod)
.set_attr<FCompute>("FCompute<gpu>",
  ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
