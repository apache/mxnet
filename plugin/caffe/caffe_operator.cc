/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_op.cc
 * \brief caffe operator
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

template<> void CaffeOperator<cpu>::CaffeForward(std::vector<::caffe::Blob<float>*> bottom, std::vector<::caffe::Blob<float>*> top){
  caffeOp_->Forward(bottom, top);
}

template<> void CaffeOperator<cpu>::CaffeBackward(std::vector<::caffe::Blob<float>*> top, std::vector<bool> bp_flags, std::vector<::caffe::Blob<float>*> bottom){
  caffeOp_->Backward(top, bp_flags, bottom);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* CaffeOperatorProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CaffeOperatorParam);

MXNET_REGISTER_OP_PROPERTY(CaffeOperator, CaffeOperatorProp)
.describe("Apply caffe operator")
.add_argument("data", "Symbol", "Input data to the Caffe Op.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(CaffeOperatorParam::__FIELDS__());


#define DEFINE_CAFFE_LAYER_FN(fn_name, layer_class) \
  static ::caffe::Layer<float>* fn_name(::caffe::LayerParameter layer_para){\
    return new layer_class(layer_para);\
  }

DEFINE_CAFFE_LAYER_FN(GenFnCaffeInnerProductLayer, ::caffe::InnerProductLayer<float>)
DEFINE_CAFFE_LAYER_FN(GenFnCaffeTanhLayer, ::caffe::TanHLayer<float>)
DEFINE_CAFFE_LAYER_FN(GenFnCaffeReluLayer, ::caffe::ReLULayer<float>)
DEFINE_CAFFE_LAYER_FN(GenFnCaffeConvLayer, ::caffe::ConvolutionLayer<float>)

bool CaffeTypeNameMap::init = false;
std::map<std::string, pFunc> CaffeTypeNameMap::to_gen_func;
std::map<std::string, int> CaffeTypeNameMap::to_type_value;
//std::map<std::string, int> CaffeTypeNameMap::to_input_num;

void CaffeTypeNameMap::DoInit(){
  init = true; 
  to_gen_func["fullyconnected"] = GenFnCaffeInnerProductLayer;
  to_gen_func["tanh"] = GenFnCaffeTanhLayer;
  to_gen_func["relu"] = GenFnCaffeReluLayer;
  to_gen_func["conv"] = GenFnCaffeConvLayer;

  to_type_value["fullyconnected"] = caffeEnum::fullyconnected;
  to_type_value["tanh"] = caffeEnum::tanh;
  to_type_value["relu"] = caffeEnum::relu;
  to_type_value["conv"] = caffeEnum::conv;
}

pFunc CaffeTypeNameMap::toFn(std::string name){
  if (!init)
    DoInit();
  CHECK(to_gen_func.count(name)>0)<<"Cannot find Caffe Type Name:" << name;
  return to_gen_func[name];
}

int CaffeTypeNameMap::toVal(std::string name){
  if (!init)
    DoInit();
  CHECK(to_type_value.count(name)>0)<<"Cannot find Caffe Type Name:" << name;
  return to_type_value[name];
}


}  // namespace op
}  // namespace mxnet
