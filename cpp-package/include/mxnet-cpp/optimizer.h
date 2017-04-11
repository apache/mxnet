/*!
*  Copyright (c) 2016 by Contributors
* \file optimizer.h
* \brief definition of optimizer
* \author Chuntao Hong, Zhang Chen
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_OPTIMIZER_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_OPTIMIZER_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/op_map.h"

namespace mxnet {
namespace cpp {

/*!
* \brief Optimizer interface
*/
class Optimizer {
 public:
  /*!
  * \brief get optimizer type
  * \return string of optimizer type
  */
  virtual std::string GetType() const = 0;
  /*!
  * \brief destructor
  */
  virtual ~Optimizer();
  /*!
  * \brief set config parameters
  * \param name name of the config parameter
  * \param value value of the config parameter
  * \return reference of self
  */
  template <typename T>
  Optimizer *SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return this;
  }
  /*!
  *  \brief Update a weight with gradient.
  *  \param index the unique index for the weight.
  *  \param weight the weight to update.
  *  \param grad gradient for the weight.
  *  \param lr learning rate.
  *  \param wd weight decay.
  */
  void Update(int index, NDArray weight, NDArray grad, mx_float lr,
              mx_float wd);
  /*!
  *  \brief Update a weight with gradient.
  *  \param index the unique index for the weight.
  *  \param weight the weight to update.
  *  \param grad gradient for the weight.
  */
  virtual void Update(int index, NDArray weight, NDArray grad) = 0;
  // TODO(zhangcheng-qinyinghua)
  // implement Update a list of arrays, maybe in the form of map
  // void Update(int index, std::vector<NDArray> weights, std::vector<NDArray>
  // grad, mx_float lr);

  /*!
  *  \brief Serialize the optimizer parameters to a string.
  *  \return serialization
  */
  std::string Serialize() const;

 protected:
  std::map<std::string, std::string> params_;
  static OpMap*& op_map();
  const std::vector<const char*> GetParamKeys_() const;
  const std::vector<const char*> GetParamValues_() const;
};

typedef std::function<Optimizer*()> OptimizerCreator;

class OptimizerRegistry {
 public:
  static Optimizer* Find(const std::string& name);
  static int __REGISTER__(const std::string& name, OptimizerCreator creator);
 private:
  static std::map<std::string, OptimizerCreator>& cmap();
  OptimizerRegistry() = delete;
  ~OptimizerRegistry() = delete;
};

#define MXNETCPP_REGISTER_OPTIMIZER(Name, OptimizerType)          \
  static int __make_ ## OptimizerType ## _ ## Name ## __ = \
       OptimizerRegistry::__REGISTER__(#Name, [](){return new OptimizerType();})

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer();
  virtual std::string GetType() const;
  virtual void Update(int index, NDArray weight, NDArray grad);
 private:
  virtual ~SGDOptimizer();
  virtual void CreateState_(int index, NDArray weight);
  std::map<int, NDArray*> states_;
  AtomicSymbolCreator update_handle_;
  AtomicSymbolCreator mom_update_handle_;
};


}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_OPTIMIZER_H_
