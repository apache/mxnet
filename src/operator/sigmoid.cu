/*!
 * Copyright (c) 2017 by Contributors
e* \file sigmoid.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./sigmoid-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sigmoid)
.set_attr<FCompute>("FCompute<gpu>", SigmoidCompute<gpu>);

NNVM_REGISTER_OP(_backward_sigmoid)
.set_attr<FCompute>("FCompute<gpu>", SigmoidBackward<gpu>);

}  // namespace op
}  // namespace mxnet

