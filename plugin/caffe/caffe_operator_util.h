/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_operator_util.h
 * \brief Caffe Operator Registry
 * \author Haoran Wang 
*/

#include <dmlc/logging.h>
#include <caffe/proto/caffe.pb.h>
#include <caffe/layer.hpp>

#include <string>
#include <map>

#ifndef PLUGIN_CAFFE_CAFFE_OPERATOR_UTIL_H_
#define PLUGIN_CAFFE_CAFFE_OPERATOR_UTIL_H_
namespace mxnet {
namespace op {


template<typename Dtype>
class CaffeOpInitEntry {
 public:
  typedef caffe::Layer<Dtype>* (*pFunc) (caffe::LayerParameter);

  CaffeOpInitEntry(std::string name,
                   pFunc f) :
                   name_(std::string(name)),
                   gen_f_(f),
                   b_num_(0) {}

  CaffeOpInitEntry<Dtype>& SetBlobNum(int b_num) {
    b_num_ = b_num;
    return *this;
  }

  int b_num_;
  pFunc gen_f_;
  std::string name_;
};

template<typename Dtype>
class CaffeOpInitRegistry {
 public:
  ~CaffeOpInitRegistry() {
    typename std::map<std::string, CaffeOpInitEntry<Dtype>*>::iterator kv;
    for (kv = fmap_.begin(); kv != fmap_.end(); ++kv)
      delete kv->second;
  }
  typedef caffe::Layer<Dtype>* (*pFunc) (caffe::LayerParameter);
  /*!
   * \brief Internal function to register the required initial params for caffe operator under name.
   * \param name name of the operator 
   * \return ref to the registered entry, used to set properties
   */
  CaffeOpInitEntry<Dtype>& __REGISTER__(const char* name_str, pFunc f) {
    std::string name(name_str);
    CHECK(fmap_.count(name) == 0) << "Caffe initial param of " << name << " already existed";
    CaffeOpInitEntry<Dtype> *e = new CaffeOpInitEntry<Dtype>(name, f);
    fmap_[name] = e;
    return *e;
  }
  /*!
   * \brief Register the entry with corresponding name.
   * \param name name of the operator 
   * \return the corresponding function, can be NULL
   */
  static CaffeOpInitEntry<Dtype>* Find(const std::string &name) {
#if MSHADOW_USE_CUDNN == 1
    if (!((name.compare("Convolution")) &&
        (name.compare("LCN")) &&
        (name.compare("LRN")) &&
        (name.compare("Pooling")) &&
        (name.compare("ReLU")) &&
        (name.compare("Sigmoid")) &&
        (name.compare("Softmax")) &&
        (name.compare("TanH")))) {
      std::string cudnn_name = "CuDNN"+name;
      CHECK(Get()->fmap_.count(cudnn_name) != 0) << "Not found caffe layer:" << cudnn_name;
      return Get()->fmap_.at(cudnn_name);
    }
#endif
    CHECK(Get()->fmap_.count(name) != 0) << "Not found caffe layer:" << name;
    return Get()->fmap_.at(name);
  }

  static CaffeOpInitEntry<Dtype>* Find(const char* name_str) {
    std::string name(name_str);
    return CaffeOpInitRegistry::Find(name);
  }
  /*! \return global singleton of the registry */
  static CaffeOpInitRegistry<Dtype>* Get() {
    static CaffeOpInitRegistry<Dtype> inst;
    return &inst;
  }

 private:
  /*! \brief internal registry map */
  std::map<std::string, CaffeOpInitEntry<Dtype>*> fmap_;
};

//--------------------------------------------------------------
// The following part are API Registration of Caffe layer
//--------------------------------------------------------------
/*!
 * \brief Macro to register caffe layers
 *
 * \Macro arguments
 *
 * Two arguments are layer's name and class
 * Name string is the type of layer specified in prototxt
 * For example, name should be XXX given the prototxt="layer{type:\"XXX\"}"
 *
 * \end Macro arguments
 *
 * \Layer with blobs
 *
 * For layers use blobs(weights), please specify blob number by SetBlobNum()
 * If the number of blobs cannot be decided in registration but during runtime,
 *    then add your code in CaffeOperatorProp::GetBlobNum() in caffe_operator-inl.h
 *
 * \end Layer with blobs
 *
 * \Add new layers
 *
 * Refer more examples at caffe_operator_util.cc and writemacro with above rules
 * Place the macro in that same file. (Don't forget to add "#include <file_of_new_clas>")
 *
 * \end Add new layers
 *
 * \code
 * // example of registering tanh layer (from caffe_operator_util.cc)
 * MXNET_REGISTER_CAFFE_LAYER(TanH, caffe::TanHLayer);
 *
 * // example of registering batch norm layer which uses 3 blobs
 * MXNET_REGISTER_CAFFE_LAYER(BatchNorm, caffe::BatchNormLayer).
 * SetBlobNum(3);
 * \endcode
 *
 *
 */

#define MXNET_REGISTER_CAFFE_LAYER(Name, LayerClass) \
  static ::caffe::Layer<float>* __make_ ## CaffeOpGenFunc ## _ ## Name ##__float__\
                            (::caffe::LayerParameter layer_para) {\
    return new LayerClass<float>(layer_para);\
  }\
\
  static ::mxnet::op::CaffeOpInitEntry<float> & \
  __make_ ## CaffeOpInitEntry ## _ ## Name ##__float__ = \
      ::mxnet::op::CaffeOpInitRegistry<float>::Get()->__REGISTER__(#Name, \
                        __make_ ## CaffeOpGenFunc ## _ ## Name ##__float__);\
\
  static ::caffe::Layer<double>* __make_ ## CaffeOpGenFunc ## _ ## Name ##__double__\
                            (::caffe::LayerParameter layer_para) {\
    return new LayerClass<double>(layer_para);\
  }\
\
  static ::mxnet::op::CaffeOpInitEntry<double> & \
  __make_ ## CaffeOpInitEntry ## _ ## Name ##__double__ = \
      ::mxnet::op::CaffeOpInitRegistry<double>::Get()->__REGISTER__(#Name, \
                        __make_ ## CaffeOpGenFunc ## _ ## Name ##__double__)\


}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_OPERATOR_UTIL_H_
