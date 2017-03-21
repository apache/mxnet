/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cu
 * \brief GPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(argmax)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::maximum>);

NNVM_REGISTER_OP(argmin)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::minimum>);

// Legacy support
NNVM_REGISTER_OP(argmax_channel)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::maximum>);

}  // namespace op
}  // namespace mxnet
