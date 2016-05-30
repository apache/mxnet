/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_base.h
 * \brief Caffe basic elements
 * \author Haoran Wang 
*/
#include<mshadow/tensor.h>
#include<caffe/common.hpp>
#include"caffe_base.h"

namespace mxnet {

template<>
void CaffeMode::SetMode<mshadow::cpu>() {
  ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
}

template<>
void CaffeMode::SetMode<mshadow::gpu>() {
  ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}

}  // namespace mxnet
