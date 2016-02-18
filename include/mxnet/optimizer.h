/*!
 *  Copyright (c) 2015 by Contributors
 * \file optimizer.h
 * \brief Operator interface of mxnet.
 * \author Junyuan Xie
 */
#ifndef MXNET_OPTIMIZER_H_
#define MXNET_OPTIMIZER_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <mshadow/tensor.h>
#include <string>
#include <vector>
#include <utility>
#include "./base.h"
#include "./resource.h"

#if DMLC_USE_CXX11
#include <mxnet/ndarray.h>
#endif

namespace mxnet {

#if !DMLC_USE_CXX11
class NDArray;
#endif

class Optimizer {
 public:
 	/*!
   * \brief virtual destructor
   */
  virtual ~Optimizer() {}
   /*!
   *  \brief Initialize the Optimizer by setting the parameters
   *  This function need to be called before all other functions.
   *  \param kwargs the keyword arguments parameters
   */
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) = 0;

  /*!
   *  \brief Create aux state for weigth with index
   *  \param index the unique index for the weight.
   *  \param weight the NDArray to associate created state to.
   */
  virtual void CreateState(const int index, const NDArray *weight) = 0;

  /*!
   *  \brief Update a weight with gradient.
   *  \param index the unique index for the weight.
   *  \param weight the weight to update.
   *  \param grad gradient for the weight.
   *  \param lr learning rate for this update.
   *  \param wd weight decay for this update.
   */
  virtual void Update(const int index, NDArray *weight,
                      const NDArray *grad, const float lr, const float wd) = 0;
  /*!
   * \brief create Optimizer
   * \param type_name the type string of the Optimizer
   * \return a new constructed Optimizer
   */
  static Optimizer *Create(const char* type_name);
};

#if DMLC_USE_CXX11

/*! \brief typedef the factory function of Optimizer */
typedef std::function<Optimizer *()> OptimizerFactory;
/*!
 * \brief Registry entry for Optimizer factory functions.
 */
struct OptimizerReg
    : public dmlc::FunctionRegEntryBase<OptimizerReg,
                                        OptimizerFactory> {
};

//--------------------------------------------------------------
// The following part are API Registration of Optimizers
//--------------------------------------------------------------
/*!
 * \brief Macro to register Optimizer
 *
 * \code
 * // example of registering a SGD optimizer
 * MXNET_REGISTER_OPTIMIZER(_SGD, SGDOptimizer)
 * .describe("Stochastic Gradient Decent optimizer");
 *
 * \endcode
 */
#define MXNET_REGISTER_OPTIMIZER(name, OptimizerType)             \
  DMLC_REGISTRY_REGISTER(::mxnet::OptimizerReg, OptimizerReg, name) \
  .set_body([]() { return new OptimizerType(); })

#endif  // DMLC_USE_CXX11

}  // namespace mxnet
#endif  // MXNET_OPTIMIZER_H_
