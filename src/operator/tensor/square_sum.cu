/*!
 *  Copyright (c) 2017 by Contributors
 * \file square_sum.cu
 * \brief GPU Implementation of square_sum op.
 */
#include "./square_sum-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_square_sum)
.set_attr<FComputeEx>("FComputeEx<gpu>", SquareSumOpForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_square_sum)
.set_attr<FComputeEx>("FComputeEx<gpu>", SquareSumOpBackwardEx<gpu>);

}  // namespace op
}  // namespace mxnet
