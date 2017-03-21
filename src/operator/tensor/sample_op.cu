/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.cu
 * \brief GPU Implementation of sample op
 */
#include "./sample_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(uniform)
.set_attr<FCompute>("FCompute<gpu>", SampleUniform_<gpu>);

NNVM_REGISTER_OP(normal)
.set_attr<FCompute>("FCompute<gpu>", SampleNormal_<gpu>);

}  // namespace op
}  // namespace mxnet
