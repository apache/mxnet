/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator.cc
 * \brief caffe operator
 * \author Haoran Wang 
*/
#include "./caffe_operator-inl.h"
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
DEFINE_CAFFE_LAYER_FN(GenFnCaffeInnerProductLayer, ::caffe::InnerProductLayer<float>)
DEFINE_CAFFE_LAYER_FN(GenFnCaffeTanhLayer, ::caffe::TanHLayer<float>)
DEFINE_CAFFE_LAYER_FN(GenFnCaffeReluLayer, ::caffe::ReLULayer<float>)
DEFINE_CAFFE_LAYER_FN(GenFnCaffeConvLayer, ::caffe::ConvolutionLayer<float>)

// Set init flag
bool CaffeTypeNameMap::init = false;
std::map<std::string, pFunc> CaffeTypeNameMap::gen_func_map;
std::map<std::string, int> CaffeTypeNameMap::enum_map,
                           CaffeTypeNameMap::in_num_map,
                           CaffeTypeNameMap::out_num_map;

// Add layer generate-function to dictionary
void CaffeTypeNameMap::DoInit() {
  init = true;
  gen_func_map["fullyconnected"] = GenFnCaffeInnerProductLayer;
  gen_func_map["tanh"] = GenFnCaffeTanhLayer;
  gen_func_map["relu"] = GenFnCaffeReluLayer;
  gen_func_map["conv"] = GenFnCaffeConvLayer;

  enum_map["fullyconnected"] = caffeEnum::fullyconnected;
  enum_map["tanh"] = caffeEnum::tanh;
  enum_map["relu"] = caffeEnum::relu;
  enum_map["conv"] = caffeEnum::conv;

  in_num_map["fullyconnected"] = 1;
  in_num_map["tanh"] = 1;
  in_num_map["relu"] = 1;
  in_num_map["conv"] = 1;

  out_num_map["fullyconnected"] = 1;
  out_num_map["tanh"] = 1;
  out_num_map["relu"] = 1;
  out_num_map["conv"] = 1;
}

pFunc CaffeTypeNameMap::GetInitFunc(std::string name) {
  if (!init)
    DoInit();
  CHECK(gen_func_map.count(name) > 0) << "Cannot find Caffe Type Name:" << name;
  return gen_func_map[name];
}

int CaffeTypeNameMap::GetType(std::string name) {
  if (!init)
    DoInit();
  CHECK(enum_map.count(name) > 0) << "Cannot find Caffe Type Name:" << name;
  return enum_map[name];
}

int CaffeTypeNameMap::GetInputNum(std::string name) {
  if (!init)
    DoInit();
  CHECK(in_num_map.count(name) > 0) << "Cannot find Caffe Type Name:" << name;
  return in_num_map[name];
}

int CaffeTypeNameMap::GetOutputNum(std::string name) {
  if (!init)
    DoInit();
  CHECK(out_num_map.count(name) > 0) << "Cannot find Caffe Type Name:" << name;
  return out_num_map[name];
}

}  // namespace op
}  // namespace mxnet
