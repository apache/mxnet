/*!
 *  Copyright (c) 2016 by Contributors
 * \file init_op.cu
 * \brief GPU Implementation of init op
 */
#include "./init_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(zeros)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 0>);

NNVM_REGISTER_OP(ones)
.set_attr<FCompute>("FCompute<gpu>", FillCompute<gpu, 1>);

}  // namespace op
}  // namespace mxnet
