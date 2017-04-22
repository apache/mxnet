/*!
 * Copyright (c) 2017 by Contributors
 * \file relu.cu
 * \brief
 * \author Ziheng Jiang
*/
#include "./relu-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(relu)
.set_attr<FCompute>("FCompute<gpu>", ReluCompute<gpu>);

NNVM_REGISTER_OP(_backward_relu)
.set_attr<FCompute>("FCompute<gpu>", ReluBackward<gpu>);

}  // namespace op
}  // namespace mxnet

