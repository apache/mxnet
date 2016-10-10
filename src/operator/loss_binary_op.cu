/*!
 * Copyright (c) 2015 by Contributors
 * \file loss_binary_op.cu
 * \brief loss function that takes a data and label
*/
#include "./loss_binary_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(softmax_cross_entropy)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCrossEntropyForward<cpu>);

NNVM_REGISTER_OP(_backward_softmax_cross_entropy)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCrossEntropyBackward<cpu>);

}  // namespace op
}  // namespace mxnet
