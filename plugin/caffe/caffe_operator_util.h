/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_util.h
 * \brief Caffe Operator Registry
 * \author Haoran Wang 
*/

#include <dmlc/logging.h>
#include <caffe/proto/caffe.pb.h>
#include <caffe/layer.hpp>

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
#include <caffe/layers/cudnn_conv_layer.hpp>
#include <caffe/layers/cudnn_lcn_layer.hpp>
#include <caffe/layers/cudnn_lrn_layer.hpp>
#include <caffe/layers/cudnn_pooling_layer.hpp>
#include <caffe/layers/cudnn_relu_layer.hpp>
#include <caffe/layers/cudnn_sigmoid_layer.hpp>
#include <caffe/layers/cudnn_softmax_layer.hpp>
#include <caffe/layers/cudnn_tanh_layer.hpp>
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


#include <string>
#include <map>

#ifndef PLUGIN_CAFFE_CAFFE_OPERATOR_UTIL_H_
#define PLUGIN_CAFFE_CAFFE_OPERATOR_UTIL_H_
namespace mxnet {
namespace op {

typedef caffe::Layer<float>* (*pFunc) (caffe::LayerParameter);

class CaffeOpInitEntry {
 public:
  CaffeOpInitEntry(std::string name,
                   pFunc f) :
                   name_(std::string(name)),
                   gen_f_(f),
                   w_num_(0) {}
  int w_num_;
  CaffeOpInitEntry& SetWeightNum(int);
  pFunc gen_f_;
  std::string name_;
};

class CaffeOpInitRegistry {
 public:
  /*!
   * \brief Internal function to register the required initial params for caffe operator under name.
   * \param name name of the operator 
   * \return ref to the registered entry, used to set properties
   */
  CaffeOpInitEntry& __REGISTER__(const char*, pFunc);
  /*!
   * \brief Register the entry with corresponding name.
   * \param name name of the operator 
   * \return the corresponding function, can be NULL
   */
  static CaffeOpInitEntry* Find(const std::string &name) {
    CHECK(Get()->fmap_.count(name) != 0) << "No caffe type named" << name;
    return Get()->fmap_.at(name);
  }

  static CaffeOpInitEntry* Find(const char* name_str) {
    std::string name(name_str);
    return CaffeOpInitRegistry::Find(name);
  }
  /*! \return global singleton of the registry */
  static CaffeOpInitRegistry* Get();

 private:
  // destructor
  ~CaffeOpInitRegistry();
  /*! \brief internal registry map */
  std::map<std::string, CaffeOpInitEntry*> fmap_;
};

//--------------------------------------------------------------
// The following part are API Registration of Caffe Plugin Init Entry
//--------------------------------------------------------------
/*!
 * \brief Macro to register caffe init entry including generate function and params.
 * \brief Firstly it builds layer init function
 * \brief Then register the entry with name and generated function
 *
 * \code
 * // example of registering init entry of fullyconnected caffe-op
 * MXNET_REGISTER_PLUGIN_CAFFE_INIT(fullyconnected, ::caffe::InnerProductLayer<float>);
 *
 * \endcode
 */
#define MXNET_REGISTER_PLUGIN_CAFFE_INIT(Name, LayerClass) \
  static ::caffe::Layer<float>* __make_ ## CaffeOpGenFunc ## _ ## Name ##__\
                            (::caffe::LayerParameter layer_para) {\
    return new LayerClass(layer_para);\
  }\
\
  static ::mxnet::op::CaffeOpInitEntry & \
  __make_ ## CaffeOpInitEntry ## _ ## Name ##__ = \
      ::mxnet::op::CaffeOpInitRegistry::Get()->__REGISTER__(#Name, \
                        __make_ ## CaffeOpGenFunc ## _ ## Name ##__)

}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_OPERATOR_UTIL_H_
