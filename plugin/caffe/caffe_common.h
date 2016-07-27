/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_common.h
 * \brief Common functions for caffeOp and caffeLoss symbols
 * \author Haoran Wang
*/

#ifndef PLUGIN_CAFFE_CAFFE_COMMON_H_
#define PLUGIN_CAFFE_CAFFE_COMMON_H_

#include <mxnet/operator.h>
#include <vector>
#include <iostream>
#include <exception>

#include <caffe/layer.hpp>
#include <caffe/blob.hpp>
#include <caffe/layer_factory.hpp>

namespace mxnet {
namespace op {
namespace caffe {

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
void InitCaffeBlobs(std::vector< ::caffe::Blob<Dtype>*>* v, int n_num) {
  for (index_t i=0; i < n_num; ++i)
    v->push_back(new ::caffe::Blob<Dtype>());
}

template<typename Dtype>
void DelCaffeBlobs(std::vector< ::caffe::Blob<Dtype>*>* v, int n_num) {
  for (index_t i=0; i < n_num; ++i)
    delete v->at(i);
}


struct NULLDeleter {template<typename T> void operator()(T*){}};

template <typename Dtype>
void Deleter(::caffe::Layer<Dtype> *ptr) {
}

template <typename Dtype>
class LayerRegistry {
 public:
  static ::caffe::Layer<Dtype> * CreateLayer(const ::caffe::LayerParameter& param) {
    ::caffe::shared_ptr< ::caffe::Layer<Dtype> > ptr =
      ::caffe::LayerRegistry<Dtype>::CreateLayer(param);
    // avoid caffe::layer destructor, which deletes the weights layer owns
    new ::caffe::shared_ptr< ::caffe::Layer<Dtype> >(ptr);
    return ptr.get();
  }
};

}  // namespace caffe
}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_COMMON_H_
