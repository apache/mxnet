/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cc
 * \brief CPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_REDUCE_AXIS(argmax)
.MXNET_DESCRIBE("Returns the indices of the maximum values along an axis.")
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::maximum>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_REDUCE_AXIS(argmin)
.MXNET_DESCRIBE("Returns the indices of the minimum values along an axis.")
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::minimum>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// Legacy support
NNVM_REGISTER_OP(argmax_channel)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser([](NodeAttrs* attrs) {
    ReduceAxisParam param;
    param.axis = 1;
    param.keepdims = false;
    attrs->parsed = param;
  })
.set_attr<nnvm::FInferShape>("FInferShape", ReduceAxisShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::maximum>)
.add_argument("src", "ndarray-or-symbol", "Source input");

}  // namespace op
}  // namespace mxnet
