/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_gpu.cc
 * \brief caffe operator
 * \author Haoran Wang
*/
#include "./caffe_operator-inl.h"
namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(CaffeOperatorParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new CaffeOperator<gpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new CaffeOperator<gpu, double>(param);
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
