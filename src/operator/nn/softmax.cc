/*!
 * Copyright (c) 2017 by Contributors
 * \file softmax.cc
 * \brief CPU Implementation of softmax
 */
#include "./softmax-inl.h"
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(SoftmaxParam);

MXNET_OPERATOR_REGISTER_UNARY(softmax)
.describe(R"code(Applies the softmax function.

The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.

.. math::
   softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}

for :math:`j = 1, ..., K`

Example::

  x = [[ 1.  1.  1.]
       [ 1.  1.  1.]]

  softmax(x,axis=0) = [[ 0.5  0.5  0.5]
                       [ 0.5  0.5  0.5]]

  softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
                       [ 0.33333334,  0.33333334,  0.33333334]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::softmax_fwd>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_softmax"})
.add_arguments(SoftmaxParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_BINARY(_backward_softmax)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxGradCompute<cpu, mshadow::op::mul,
                                                        mxnet_op::softmax_bwd>);

MXNET_OPERATOR_REGISTER_UNARY(log_softmax)
.describe(R"code(Computes the log softmax of the input.
This is equivalent to computing softmax followed by log.

Examples::

  >>> x = mx.nd.array([1, 2, .1])
  >>> mx.nd.log_softmax(x).asnumpy()
  array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)

  >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
  >>> mx.nd.log_softmax(x, axis=0).asnumpy()
  array([[-0.34115392, -0.69314718, -1.24115396],
         [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)


)code")
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::log_softmax_fwd>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_log_softmax"})
.add_arguments(SoftmaxParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_BINARY(_backward_log_softmax)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxGradCompute<cpu, mshadow_op::left,
                                                        mxnet_op::log_softmax_bwd>);

}  // namespace op
}  // namespace mxnet
