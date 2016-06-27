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

// Enumeration for operator type
namespace caffeEnum {
enum CaffeOpType {fullyconnected, tanh, relu, conv};
}  // namespace caffeEnum

typedef caffe::Layer<float>* (*pFunc) (caffe::LayerParameter);

class CaffeOpInitEntry {
 public:
  CaffeOpInitEntry(std::string name,
                   int op_enum,
                   pFunc f) :
                   name_(std::string(name)),
                   op_enum_(op_enum),
                   gen_f_(f),
                   in_num_(1),
                   out_num_(1) {}

  CaffeOpInitEntry& SetInNum(int in_num);
  CaffeOpInitEntry& SetOutNum(int out_num);

  pFunc GetInNum();

  int in_num_, out_num_, op_enum_;
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
  CaffeOpInitEntry& __REGISTER__(const char*, int, pFunc);
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
 * \brief Macro to register simple operator to both imperative and symbolic API.
 *
 * see src/operator/elementwise_unary_op-inl.h for example
 *
 * \code
 * // example of registering a sigmoid operator on GPU
 * // MySigmoid is of type UnaryFunction,
 * // MySigmoidGrad is of type UnaryGradFunctionT2
 *
 * MXNET_REGISTER_SIMPLE_OP(sigmoid, cpu)
 * .set_function(MySigmoid<gpu>, true)
 * .set_gradient(MySigmoidGrad<gpu>, true)
 * .describe("Sigmoid function");
 *
 * \endcode
 */
#define MXNET_REGISTER_PLUGIN_CAFFE_INIT(Name, GenFunc) \
  static ::mxnet::op::CaffeOpInitEntry & \
  __make_ ## CaffeOpInitEntry ## _ ## Name ##__ = \
      ::mxnet::op::CaffeOpInitRegistry::Get()->__REGISTER__(#Name, caffeEnum::Name, GenFunc)

}  // namespace op
}  // namespace mxnet
#endif  // PLUGIN_CAFFE_CAFFE_OPERATOR_UTIL_H_
