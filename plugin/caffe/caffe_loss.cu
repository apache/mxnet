/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_loss_gpu.cc
 * \brief caffe loss 
 * \author Haoran Wang 
*/
#include "./caffe_loss-inl.h"
namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(CaffeLossParam param) {
  return new CaffeLoss<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
