/*!
 * Copyright (c) 2015 by Contributors
 * \file loss_binary_op.cc
 * \brief loss function that takes a data and label
*/
#include "./loss_binary_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(softmax_cross_entropy)
.MXNET_DESCRIBE("Calculate cross_entropy(data, one_hot(label))")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", SoftmaxCrossEntropyShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCrossEntropyForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_softmax_cross_entropy"})
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_argument("label", "NDArray-or-Symbol", "Input label");

NNVM_REGISTER_OP(_backward_softmax_cross_entropy)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCrossEntropyBackward<cpu>);

}  // namespace op
}  // namespace mxnet
