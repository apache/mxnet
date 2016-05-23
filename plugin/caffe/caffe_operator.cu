/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_gpu.cc
 * \brief caffe operator
*/
#include "./caffe_operator-inl.h"
#include<caffe/common.hpp>
namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(CaffeOperatorParam param) {
  return new CaffeOperator<gpu>(param);
}

template<> void CaffeOperator<gpu>::CaffeForward(std::vector<caffe::Blob<float>*> bottom, std::vector<caffe::Blob<float>*> top){

  caffeOp_->Forward(bottom, top);
}

template<> void CaffeOperator<gpu>::CaffeBackward(std::vector<caffe::Blob<float>*> top, std::vector<bool> bp_flags, std::vector<caffe::Blob<float>*> bottom){
  caffeOp_->Backward(top, bp_flags, bottom);
}

}  // namespace op
}  // namespace mxnet
