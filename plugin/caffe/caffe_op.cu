/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_gpu.cc
 * \brief caffe operator
 * \author Haoran Wang
*/
#include "./caffe_op-inl.h"
namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(CaffeOpParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new CaffeOp<gpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new CaffeOp<gpu, double>(param);
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
