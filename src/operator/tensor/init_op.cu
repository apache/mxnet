/*!
 *  Copyright (c) 2016 by Contributors
 * \file init_op.cu
 * \brief GPU Implementation of init op
 */
#include "./init_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_zeros)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>);

NNVM_REGISTER_OP(_ones)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

NNVM_REGISTER_OP(_arange)
.set_attr<FCompute>("FCompute<gpu>", RangeCompute<gpu>);

NNVM_REGISTER_OP(zeros_like)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>);

NNVM_REGISTER_OP(ones_like)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

}  // namespace op
}  // namespace mxnet
