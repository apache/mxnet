/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_common.h
 * \brief Common functions for caffeOp and caffeLoss symbols
 * \author Haoran Wang 
*/
#include<mshadow/tensor.h>
#include<caffe/common.hpp>
#include"caffe_common.h"

namespace mxnet {
// Cpu implementation of set_mode
template<>
void CaffeMode::SetMode<mshadow::cpu>() {
  ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
}

// Gpu implementation of set_mode
template<>
void CaffeMode::SetMode<mshadow::gpu>() {
  ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}

void InitCaffeBlobs(std::vector<Blob<float>*>& v, int n_num) {
  for (index_t i=0; i<n_num; ++i)
    v.push_back(new Blob<float>());
}

void DelCaffeBlobs(std::vector<Blob<float>*>& v, int n_num) {
  for (index_t i=0; i<n_num; ++i)
    delete v[i];
}

}  // namespace mxnet
