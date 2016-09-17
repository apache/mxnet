/*!
 *  Copyright (c) 2016 by Contributors
 * \file optimizer_op.cu
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#include "./optimizer_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sgd_update)
.set_attr<FCompute>("FCompute<gpu>", SGDUpdate<gpu>);

NNVM_REGISTER_OP(sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", SGDMomUpdate<gpu>);

NNVM_REGISTER_OP(adam_update)
.set_attr<FCompute>("FCompute<gpu>", AdamUpdate<gpu>);

}  // namespace op
}  // namespace mxnet
