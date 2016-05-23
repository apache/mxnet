#ifndef CAFFE_LAYER_PARAMTER_H_
#define CAFFE_LAYER_PARAMTER_H_

#include<iostream>
#include<caffe/proto/caffe.pb.h> 
namespace dmlc{
  namespace parameter{

    std::istringstream &operator>>(std::istringstream &is, ::caffe::LayerParameter &para_);
    std::ostream &operator<<(std::ostream &os, ::caffe::LayerParameter &para_);
  }
}

#endif  // CAFFE_LAYER_PARAMTER_H_
