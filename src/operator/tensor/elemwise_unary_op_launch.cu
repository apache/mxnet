/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise_unary_op_launch.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op_launch.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(relu)
.set_attr<FCompute>("FCompute<gpu>",
    UnaryLaunch<gpu, kernel_launch_op::relu>);

NNVM_REGISTER_OP(_backward_relu)
.set_attr<FCompute>("FCompute<gpu>",
    UnaryLaunch<gpu, kernel_launch_op::relu_grad>);

NNVM_REGISTER_OP(sigmoid)
.set_attr<FCompute>("FCompute<gpu>",
    UnaryLaunch<gpu, kernel_launch_op::sigmoid>);

NNVM_REGISTER_OP(_backward_sigmoid)
.set_attr<FCompute>("FCompute<gpu>",
    UnaryLaunch<gpu, kernel_launch_op::sigmoid_grad>);

}  // namespace op
}  // namespace mxnet
