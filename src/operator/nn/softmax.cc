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
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::softmax_fwd>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_softmax"})
.add_argument("data", "ndarray-or-symbol", "Input data.")
.add_arguments(SoftmaxParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_BINARY(_backward_softmax)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxGradCompute<cpu, mshadow::op::mul,
                                                        mxnet_op::softmax_bwd>);

MXNET_OPERATOR_REGISTER_UNARY(log_softmax)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxCompute<cpu, mxnet_op::log_softmax_fwd>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_log_softmax"})
.add_argument("data", "ndarray-or-symbol", "Input data.")
.add_arguments(SoftmaxParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_BINARY(_backward_log_softmax)
.set_attr_parser(ParamParser<SoftmaxParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxGradCompute<cpu, mshadow_op::left,
                                                        mxnet_op::log_softmax_bwd>);

}  // namespace op
}  // namespace mxnet
