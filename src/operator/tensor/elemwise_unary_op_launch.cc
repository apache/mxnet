/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise_unary_op_launch.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op_launch.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_UNARY(relu)
.MXNET_DESCRIBE("ReLU operator")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_relu"})
.set_attr<FCompute>("FCompute<cpu>",
    UnaryLaunch<cpu, kernel_launch_op::relu>);

MXNET_OPERATOR_REGISTER_UNARY(_backward_relu)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
    UnaryLaunch<cpu, kernel_launch_op::relu_grad>);

MXNET_OPERATOR_REGISTER_UNARY(sigmoid)
.MXNET_DESCRIBE("Sigmoid operator")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sigmoid"})
.set_attr<FCompute>("FCompute<cpu>",
    UnaryLaunch<cpu, kernel_launch_op::sigmoid>);

MXNET_OPERATOR_REGISTER_UNARY(_backward_sigmoid)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>",
    UnaryLaunch<cpu, kernel_launch_op::sigmoid_grad>);

}  // namespace op
}  // namespace mxnet
