/*!
 *  Copyright (c) 2017 by Contributors
 * \file relu.cc
 * \brief
 * \author Ziheng Jiang
 */
#include "./relu-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(relu)
.MXNET_DESCRIBE("relu operator")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", ReluCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_relu"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`");

NNVM_REGISTER_OP(_backward_relu)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ReluBackward<cpu>);

}  // namespace op
}  // namespace mxnet
