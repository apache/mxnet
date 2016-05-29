/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_inl.h
 * \brief Caffe
 * \author Haoran Wang 
*/
#include"caffe_mode.h"
#include<iostream>

namespace mxnet{
namespace op{

template<> 
void CaffeMode::SetMode<mshadow::cpu>(){
  ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
}

template<> 
void CaffeMode::SetMode<mshadow::gpu>(){
  ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}

}
}
