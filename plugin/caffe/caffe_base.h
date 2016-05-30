/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_base.h
 * \brief Caffe basic elements
 * \author Haoran Wang
*/

#ifndef PLUGIN_CAFFE_CAFFE_BASE_H_
#define PLUGIN_CAFFE_CAFFE_BASE_H_

namespace mxnet {

/**
 * \brief The class sets caffe's mode before doing forward/backward
 * \tparam xpu The device that the op will be executed on.
 */
class CaffeMode {
 public:
  template<typename xpu> static void SetMode();
};

}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_BASE_H_
