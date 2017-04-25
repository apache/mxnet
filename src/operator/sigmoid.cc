/*!
 *  Copyright (c) 2017 by Contributors
 * \file sigmoid.cc
 * \brief
 * \author Ziheng Jiang
 */
#include "./sigmoid-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sigmoid)
.MXNET_DESCRIBE("sigmoid operator")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", SigmoidCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sigmoid"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`");

NNVM_REGISTER_OP(_backward_sigmoid)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SigmoidBackward<cpu>);

}  // namespace op
}  // namespace mxnet
