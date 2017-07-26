/*!
 *  Copyright (c) 2017 by Contributors
 * \file square_sum.cc
 * \brief CPU Implementation of square_sum op.
 */
#include "./square_sum-inl.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_REDUCE(_square_sum)
.describe(R"code(Computes the square sum of array elements over a given axis
for row-sparse matrix. This is a temporary solution for fusing ops square and
sum together for row-sparse matrix to save memory for storing gradients.
It will become deprecated once the functionality of fusing operators is finished
in the future.

Example::

  dns = mx.nd.array([[0, 0], [1, 2], [0, 0], [3, 4], [0, 0]])
  rsp = mx.nd.cast_storage(dns, stype='row_sparse')
  sum = mx.nd._internal._square_sum(rsp, axis=1)
  sum = [0, 5, 0, 25, 0]
)code" ADD_FILELINE)
.set_attr<FInferStorageType>("FInferStorageType", SquareSumForwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SquareSumOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square_sum"});

MXNET_OPERATOR_REGISTER_REDUCE_BACKWARD(_backward_square_sum)
.set_num_inputs(2)
.set_attr<FInferStorageType>("FInferStorageType", SquareSumBackwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SquareSumOpBackwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
