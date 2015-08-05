/*!
 * Copyright (c) 2015 by Contributors
 * \file static_operator-inl.h
 * \brief static device invarient code to create operators
 * \author Bing Xu
*/
#ifndef MXNET_STATIC_OPERATOR_STATIC_OPERATOR_INL_H_
#define MXNET_STATIC_OPERATOR_STATIC_OPERATOR_INL_H_
#include <mxnet/base.h>
#include <dmlc/logging.h>
#include <mxnet/static_operator.h>
#include "./mshadow_op.h"
#include "./activation_op-inl.h"
#include "./convolution_op-inl.h"
#include "./pooling_op-inl.h"
#include "./reshape_op-inl.h"
#include "./dropout_op-inl.h"

namespace mxnet {
namespace op {
/*!
 * \brief device invariant function to create operators
 * \param type the type of operator
 * \tparam xpu the device type we are at
 */
template<typename xpu>
inline StaticOperator *CreateOperator_(OpType type, mshadow::Random<xpu> *prnd) {
  switch (type) {
    case kReLU:
      return new ActivationOp<xpu, relu, relu_grad>();
    case kConv:
      return new ConvolutionOp<xpu>();
    case kMaxPooling:
      return new PoolingOp<xpu, mshadow::red::maximum, kMaxPooling>();
    case kAvgPooling:
      return new PoolingOp<xpu, mshadow::red::sum, kAvgPooling>();
    case kFlatten:
      return new ReshapeOp<xpu, true>();
    case kReshape:
      return new ReshapeOp<xpu, false>();
    case kDropout:
      return new DropoutOp<xpu>(prnd);
    default: LOG(FATAL) << "unknown OpType";
  }
  return NULL;
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_STATIC_OPERATOR_STATIC_OPERATOR_INL_H_
