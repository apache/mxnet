/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_util.h
 * \brief Caffe Operator Registry
 * \author Haoran Wang 
*/
#include "caffe_operator_util.h"

#include <caffe/layers/absval_layer.hpp>
#include <caffe/layers/accuracy_layer.hpp>
#include <caffe/layers/argmax_layer.hpp>
#include <caffe/layers/base_conv_layer.hpp>
#include <caffe/layers/base_data_layer.hpp>
#include <caffe/layers/batch_norm_layer.hpp>
#include <caffe/layers/batch_reindex_layer.hpp>
#include <caffe/layers/bias_layer.hpp>
#include <caffe/layers/bnll_layer.hpp>
#include <caffe/layers/concat_layer.hpp>
#include <caffe/layers/contrastive_loss_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/crop_layer.hpp>
#include <caffe/layers/data_layer.hpp>
#include <caffe/layers/deconv_layer.hpp>
#include <caffe/layers/dropout_layer.hpp>
#include <caffe/layers/dummy_data_layer.hpp>
#include <caffe/layers/eltwise_layer.hpp>
#include <caffe/layers/elu_layer.hpp>
#include <caffe/layers/embed_layer.hpp>
#include <caffe/layers/euclidean_loss_layer.hpp>
#include <caffe/layers/exp_layer.hpp>
#include <caffe/layers/filter_layer.hpp>
#include <caffe/layers/flatten_layer.hpp>
#include <caffe/layers/hdf5_data_layer.hpp>
#include <caffe/layers/hdf5_output_layer.hpp>
#include <caffe/layers/hinge_loss_layer.hpp>
#include <caffe/layers/im2col_layer.hpp>
#include <caffe/layers/image_data_layer.hpp>
#include <caffe/layers/infogain_loss_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/input_layer.hpp>
#include <caffe/layers/log_layer.hpp>
#include <caffe/layers/loss_layer.hpp>
#include <caffe/layers/lrn_layer.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/multinomial_logistic_loss_layer.hpp>
#include <caffe/layers/mvn_layer.hpp>
#include <caffe/layers/neuron_layer.hpp>
#include <caffe/layers/parameter_layer.hpp>
#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/power_layer.hpp>
#include <caffe/layers/prelu_layer.hpp>
#include <caffe/layers/reduction_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/reshape_layer.hpp>
#include <caffe/layers/scale_layer.hpp>
#include <caffe/layers/sigmoid_cross_entropy_loss_layer.hpp>
#include <caffe/layers/sigmoid_layer.hpp>
#include <caffe/layers/silence_layer.hpp>
#include <caffe/layers/slice_layer.hpp>
#include <caffe/layers/softmax_layer.hpp>
#include <caffe/layers/softmax_loss_layer.hpp>
#include <caffe/layers/split_layer.hpp>
#include <caffe/layers/spp_layer.hpp>
#include <caffe/layers/tanh_layer.hpp>
#include <caffe/layers/threshold_layer.hpp>
#include <caffe/layers/tile_layer.hpp>
#include <caffe/layers/window_data_layer.hpp>

#if MSHADOW_USE_CUDNN == 1
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

namespace mxnet {
namespace op {

CaffeOpInitEntry& CaffeOpInitEntry::SetBlobNum(int b_num) {
  b_num_ = b_num;
  return *this;
}

CaffeOpInitEntry& CaffeOpInitRegistry::__REGISTER__(const char* name_str,
                                                    pFunc f) {
  std::string name(name_str);
  CHECK(fmap_.count(name) == 0) << "Caffe initial param of " << name << " already existed";
  CaffeOpInitEntry *e = new CaffeOpInitEntry(name, f);
  fmap_[name] = e;
  return *e;
}

CaffeOpInitRegistry* CaffeOpInitRegistry::Get() {
  static CaffeOpInitRegistry inst;
  return &inst;
}

CaffeOpInitRegistry::~CaffeOpInitRegistry() {
  for (auto kv : fmap_) {
    delete kv.second;
  }
}

MXNET_REGISTER_CAFFE_LAYER(InnerProduct, caffe::InnerProductLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(TanH, caffe::TanHLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(ReLU, caffe::ReLULayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Convolution, caffe::ConvolutionLayer<float>);

MXNET_REGISTER_CAFFE_LAYER(AbsVal, caffe::AbsValLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Accuracy, caffe::AccuracyLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(BatchNorm, caffe::BatchNormLayer<float>).
SetBlobNum(3);
MXNET_REGISTER_CAFFE_LAYER(BatchReindex, caffe::BatchReindexLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Bias, caffe::BiasLayer<float>).
SetBlobNum(1);
MXNET_REGISTER_CAFFE_LAYER(BNLL, caffe::BNLLLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Concat, caffe::ConcatLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(ContrastiveLoss, caffe::ContrastiveLossLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Crop, caffe::CropLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Deconvolution, caffe::DeconvolutionLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Dropout, caffe::DropoutLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Eltwise, caffe::EltwiseLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(ELULayer, caffe::ELULayer<float>);
MXNET_REGISTER_CAFFE_LAYER(EmbedLayer, caffe::EmbedLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(EuclideanLoss, caffe::EuclideanLossLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Exp, caffe::ExpLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Filter, caffe::FilterLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Flatten, caffe::FlattenLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(HingeLoss, caffe::HingeLossLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Im2col, caffe::Im2colLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(InfogainLoss, caffe::InfogainLossLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Input, caffe::InputLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Log, caffe::LogLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(LRN, caffe::LRNLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(MultinomialLogisticLoss,
                                 caffe::MultinomialLogisticLossLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(MVN, caffe::MVNLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Parameter, caffe::ParameterLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Pooling, caffe::PoolingLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Power, caffe::PowerLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(PReLU, caffe::PReLULayer<float>).
SetBlobNum(1);
MXNET_REGISTER_CAFFE_LAYER(Reduction, caffe::ReductionLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Reshape, caffe::ReshapeLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Scale, caffe::ScaleLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(SigmoidCrossEntropyLoss,
                                 caffe::SigmoidCrossEntropyLossLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Sigmoid, caffe::SigmoidLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Silence, caffe::SilenceLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Slice, caffe::SliceLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Softmax, caffe::SoftmaxLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(SoftmaxWithLoss, caffe::SoftmaxWithLossLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Split, caffe::SplitLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(SPP, caffe::SPPLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Threshold, caffe::ThresholdLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(Tile, caffe::TileLayer<float>);

#if MSHADOW_USE_CUDNN == 1
MXNET_REGISTER_CAFFE_LAYER(CuDNNConvolution, caffe::CuDNNConvolutionLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(CuDNNLCN, caffe::CuDNNLCNLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(CuDNNLRN, caffe::CuDNNLRNLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(CuDNNPooling, caffe::CuDNNPoolingLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(CuDNNReLU, caffe::CuDNNReLULayer<float>);
MXNET_REGISTER_CAFFE_LAYER(CuDNNSigmoid, caffe::CuDNNSigmoidLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(CuDNNSoftmax, caffe::CuDNNSoftmaxLayer<float>);
MXNET_REGISTER_CAFFE_LAYER(CuDNNTanH, caffe::CuDNNTanHLayer<float>);
#endif
/*
 * /brief: Data layer hasn't been supported yet
 * MXNET_REGISTER_CAFFE_LAYER(Data, caffe::DataLayer<float>);
 * MXNET_REGISTER_CAFFE_LAYER(DummyData, caffe::DummyDataLayer<float>);
 * MXNET_REGISTER_CAFFE_LAYER(ImageData, caffe::ImageDataLayer<float>);
 * MXNET_REGISTER_CAFFE_LAYER(MemoryData, caffe::MemoryDataLayer<float>);
 * MXNET_REGISTER_CAFFE_LAYER(WindowData, caffe::WindowDataLayer<float>);
*/

}  // namespace op
}  // namespace mxnet
