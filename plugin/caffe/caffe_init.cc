#include"caffe_init.h"
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
