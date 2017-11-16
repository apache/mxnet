/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_stream.h
 * \brief define stream opertors >> and <<
 * \author Haoran Wang 
*/
#ifndef PLUGIN_CAFFE_CAFFE_STREAM_H_
#define PLUGIN_CAFFE_CAFFE_STREAM_H_

#include<caffe/proto/caffe.pb.h>
#include<iostream>
namespace dmlc {
namespace parameter {
  std::istringstream &operator>>(std::istringstream &is, ::caffe::LayerParameter &para_);
  std::ostream &operator<<(std::ostream &os, ::caffe::LayerParameter &para_);
}
}

#endif  // PLUGIN_CAFFE_CAFFE_STREAM_H_
