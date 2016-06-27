/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_util.h
 * \brief Caffe Operator Registry
 * \author Haoran Wang 
*/

#include <dmlc/logging.h>
#include <caffe/proto/caffe.pb.h>
#include <caffe/layer.hpp>

#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/tanh_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>

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
                   in_num_(1),
                   out_num_(1) {}

  CaffeOpInitEntry& SetInNum(int in_num);
  CaffeOpInitEntry& SetOutNum(int out_num);

  pFunc GetInNum();

  int in_num_, out_num_;
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
