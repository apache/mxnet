/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator.cc
 * \brief caffe operator
 * \author Haoran Wang 
*/
#include "./caffe_operator-inl.h"
#include "./caffe_operator_util.h"
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

// Wrap caffe's layer_class to layer-generator function
MXNET_REGISTER_PLUGIN_CAFFE_INIT(InnerProduct, ::caffe::InnerProductLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Tanh, ::caffe::TanHLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Relu, ::caffe::ReLULayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Conv, ::caffe::ConvolutionLayer<float>);

MXNET_REGISTER_PLUGIN_CAFFE_INIT(AbsVal, ::caffe::AbsValLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Accuracy, ::caffe::AccuracyLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(BatchNormLayer, ::caffe::BatchNormLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(BatchReindex, ::caffe::BatchReindexLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Bias, ::caffe::BiasLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(BNLL, ::caffe::BNLLLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Concat, ::caffe::ConcatLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(ContrastiveLoss, ::caffe::ContrastiveLossLayer<float>);

MXNET_REGISTER_PLUGIN_CAFFE_INIT(Crop, ::caffe::CropLayer<float>);
/*
 * /brief: (TODO) Haoran: support cudnn compilation
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNConv, ::caffe::CuDNNConvolutionLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNLCN, ::caffe::CuDNNLCNLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNLRN, ::caffe::CuDNNLRNLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNPool, ::caffe::CuDNNPoolingLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNLCN, ::caffe::CuDNNLCNLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNReLULayer, ::caffe::CuDNNReLULayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNSigmoid, ::caffe::CuDNNSigmoidLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNSoftmax, ::caffe::CuDNNSoftmaxLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNTanH, ::caffe::CuDNNTanHLayer<float>);
*/

/*
 * /brief: Data layer hasn't been supported yet
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(Data, ::caffe::DataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(DummyData, ::caffe::DummyDataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(ImageData, ::caffe::ImageDataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(MemoryData, ::caffe::MemoryDataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(WindowData, ::caffe::WindowDataLayer<float>);
*/

MXNET_REGISTER_PLUGIN_CAFFE_INIT(Deconvolution, ::caffe::DeconvolutionLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Dropout, ::caffe::DropoutLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Eltwise, ::caffe::EltwiseLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(ELULayer, ::caffe::ELULayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(EmbedLayer, ::caffe::EmbedLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(EuclideanLoss, ::caffe::EuclideanLossLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Exp, ::caffe::ExpLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Filter, ::caffe::FilterLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Flatten, ::caffe::FlattenLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(HingeLoss, ::caffe::HingeLossLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Im2col, ::caffe::Im2colLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(InfogainLoss, ::caffe::InfogainLossLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Input, ::caffe::InputLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Log, ::caffe::LogLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(LRN, ::caffe::LRNLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(MultinomialLogisticLoss, 
                                 ::caffe::MultinomialLogisticLossLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(MVN, ::caffe::MVNLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Parameter, ::caffe::ParameterLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Pool, ::caffe::PoolingLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Power, ::caffe::PowerLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(PReLU, ::caffe::PReLULayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Reduction, ::caffe::ReductionLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Reshape, ::caffe::ReshapeLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Scale, ::caffe::ScaleLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(SigmoidCrossEntropyLoss,
                                 ::caffe::SigmoidCrossEntropyLossLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Sigmoid, ::caffe::SigmoidLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Silence, ::caffe::SilenceLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Slice, ::caffe::SliceLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Softmax, ::caffe::SoftmaxLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(SoftmaxWithLoss, ::caffe::SoftmaxWithLossLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Split, ::caffe::SplitLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(SPP, ::caffe::SPPLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Threshold, ::caffe::ThresholdLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Tile, ::caffe::TileLayer<float>);
}  // namespace op
}  // namespace mxnet
