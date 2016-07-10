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

MXNET_REGISTER_PLUGIN_CAFFE_INIT(InnerProduct, ::caffe::InnerProductLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(TanH, ::caffe::TanHLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(ReLU, ::caffe::ReLULayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Convolution, ::caffe::ConvolutionLayer<float>);

MXNET_REGISTER_PLUGIN_CAFFE_INIT(AbsVal, ::caffe::AbsValLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Accuracy, ::caffe::AccuracyLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(BatchNorm, ::caffe::BatchNormLayer<float>).
SetBlobNum(3);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(BatchReindex, ::caffe::BatchReindexLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Bias, ::caffe::BiasLayer<float>).
SetBlobNum(1);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(BNLL, ::caffe::BNLLLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Concat, ::caffe::ConcatLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(ContrastiveLoss, ::caffe::ContrastiveLossLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Crop, ::caffe::CropLayer<float>);
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
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Pooling, ::caffe::PoolingLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(Power, ::caffe::PowerLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(PReLU, ::caffe::PReLULayer<float>).
SetBlobNum(1);
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

#if MSHADOW_USE_CUDNN == 1
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNConvolution, ::caffe::CuDNNConvolutionLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNLCN, ::caffe::CuDNNLCNLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNLRN, ::caffe::CuDNNLRNLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNPooling, ::caffe::CuDNNPoolingLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNReLU, ::caffe::CuDNNReLULayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNSigmoid, ::caffe::CuDNNSigmoidLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNSoftmax, ::caffe::CuDNNSoftmaxLayer<float>);
MXNET_REGISTER_PLUGIN_CAFFE_INIT(CuDNNTanH, ::caffe::CuDNNTanHLayer<float>);
#endif
/*
 * /brief: Data layer hasn't been supported yet
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(Data, ::caffe::DataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(DummyData, ::caffe::DummyDataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(ImageData, ::caffe::ImageDataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(MemoryData, ::caffe::MemoryDataLayer<float>);
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(WindowData, ::caffe::WindowDataLayer<float>);
*/

}  // namespace op
}  // namespace mxnet
