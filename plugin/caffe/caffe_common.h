/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_common.h
 * \brief Common functions for caffeOp and caffeLoss symbols
 * \author Haoran Wang
*/

#ifndef PLUGIN_CAFFE_CAFFE_COMMON_H_
#define PLUGIN_CAFFE_CAFFE_COMMON_H_

#include <mxnet/operator.h>
#include<caffe/blob.hpp>
#include<vector>

namespace mxnet {

using caffe::Blob;

/**
 * \brief The class sets caffe's mode before doing forward/backward
 * \tparam xpu The device that the op will be executed on.
 */
class CaffeMode {
 public:
  template<typename xpu> static void SetMode();
};

// Initialization funciton called by caffeOp & caffeLoss
template<typename Dtype>
void InitCaffeBlobs(std::vector<Blob<Dtype>*>* v, int n_num) {
  for (index_t i=0; i < n_num; ++i)
    v->push_back(new Blob<Dtype>());
}

template<typename Dtype>
void DelCaffeBlobs(std::vector<Blob<Dtype>*>* v, int n_num) {
  for (index_t i=0; i < n_num; ++i)
    delete v->at(i);
}

}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_COMMON_H_
