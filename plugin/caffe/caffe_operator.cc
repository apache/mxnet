/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator.cc
 * \brief caffe operator
 * \author Haoran Wang 
*/
#include "./caffe_operator-inl.h"
#include "./caffe_operator_util.h"
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/tanh_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(CaffeOperatorParam param) {
  return new CaffeOperator<cpu>(param);
}

template<> void CaffeOperator<cpu>::CaffeForward(std::vector<::caffe::Blob<float>*> bottom, \
    std::vector<::caffe::Blob<float>*> top) {
  caffeOp_->Forward(bottom, top);
}

template<> void CaffeOperator<cpu>::CaffeBackward(std::vector<::caffe::Blob<float>*> top, \
    std::vector<bool> bp_flags, std::vector<::caffe::Blob<float>*> bottom) {
  caffeOp_->Backward(top, bp_flags, bottom);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* CaffeOperatorProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CaffeOperatorParam);

MXNET_REGISTER_OP_PROPERTY(CaffeOperator, CaffeOperatorProp)
.describe("Apply caffe operator")
.add_argument("data", "Symbol[]", "List of tensors")
.add_arguments(CaffeOperatorParam::__FIELDS__());


#define DEFINE_CAFFE_LAYER_FN(fn_name, layer_class) \
  static ::caffe::Layer<float>* fn_name(::caffe::LayerParameter layer_para) {\
    return new layer_class(layer_para);\
  }

// Wrap caffe's layer_class to layer-generator function
DEFINE_CAFFE_LAYER_FN(CaffeInnerProductFunc, ::caffe::InnerProductLayer<float>)
DEFINE_CAFFE_LAYER_FN(CaffeTanhFunc, ::caffe::TanHLayer<float>)
DEFINE_CAFFE_LAYER_FN(CaffeReluFunc, ::caffe::ReLULayer<float>)
DEFINE_CAFFE_LAYER_FN(CaffeConvFunc, ::caffe::ConvolutionLayer<float>)

MXNET_REGISTER_PLUGIN_CAFFE_INIT(fullyconnected, CaffeInnerProductFunc);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(tanh, CaffeTanhFunc);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(relu, CaffeReluFunc);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(conv, CaffeConvFunc);

}  // namespace op
}  // namespace mxnet
