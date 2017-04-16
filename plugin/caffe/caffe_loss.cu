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
Operator* CreateOp<gpu>(CaffeLossParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new CaffeLoss<gpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new CaffeLoss<gpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 layer is not supported by caffe";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}

}  // namespace op
}  // namespace mxnet
