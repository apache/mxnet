/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_gpu.cc
 * \brief caffe operator
*/
#include "./caffe_operator-inl.h"
namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(CaffeOperatorParam param) {
  return new CaffeOperator<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
