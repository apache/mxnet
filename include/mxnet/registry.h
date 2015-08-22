/*!
 *  Copyright (c) 2015 by Contributors
 * \file registry.h
 * \brief registry that registers all sorts of functions
 */
#ifndef MXNET_REGISTRY_H_
#define MXNET_REGISTRY_H_

#include <dmlc/base.h>
#include <dmlc/registry.h>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include "./base.h"
#include "./narray.h"
#include "./operator.h"

namespace mxnet {
/*! \brief definition of NArray function */
typedef std::function<void (NArray **used_vars,
                            real_t *scalars,
                            NArray **mutate_vars)> NArrayAPIFunction;
/*! \brief mask information on how functions can be exposed */
enum NArrayFunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};

/*! \brief Registry entry for NArrayFunction */
struct NArrayFunctionReg
    : public dmlc::FunctionRegEntryBase<NArrayFunctionReg,
                                        NArrayAPIFunction> {
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*!
   * \brief constructor
   */
  explicit NArrayFunctionReg()
      : num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0) {}
  /*!
   * \brief set the function body to a binary NArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_function(void fbinary(const NArray &lhs,
                                                      const NArray &rhs,
                                                      NArray *out)) {
    body = [fbinary] (NArray **used_vars,
                      real_t *s, NArray **mutate_vars) {
      fbinary(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NArray", "Left operand to the function.");
    this->add_argument("rhs", "NArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_function(void funary(const NArray &src,
                                                     NArray *out)) {
    body = [funary] (NArray **used_vars,
                     real_t *s, NArray **mutate_vars) {
      funary(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("src", "NArray", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
};  // NArrayFunctionReg

/*!
 * \brief Macro to register NArray function
 *
 * Example: the following code is example to register a plus
 * \code
 *
 * REGISTER_NARRAY_FUN(Plus)
 * .set_function(Plus);
 *
 * \endcode
 */
#define MXNET_REGISTER_NARRAY_FUN(name)                                 \
  DMLC_REGISTRY_REGISTER(::mxnet::NArrayFunctionReg, NArrayFunctionReg, name)

/*! \brief typedef the factory function of operator property */
typedef OperatorProperty *(*OperatorPropertyFactory)();
/*!
 * \brief Registry entry for OperatorProperty factory functions.
 */
struct OperatorPropertyReg
    : public dmlc::FunctionRegEntryBase<OperatorPropertyReg,
                                        OperatorPropertyFactory> {
};

/*!
 * \brief Macro to register OperatorProperty
 *
 * \code
 * // example of registering a fully connected operator
 * REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedOpProp)
 * .describe("Fully connected layer");
 *
 * \endcode
 */
#define MXNET_REGISTER_OP_PROPERTY(name, OperatorPropertyType)          \
  static ::mxnet::OperatorProperty* __create__ ## OperatorPropertyType ## __() { \
    return new OperatorPropertyType;                                    \
  }                                                                     \
  DMLC_REGISTRY_REGISTER(::mxnet::OperatorPropertyReg, OperatorPropertyReg, name) \
  .set_body(__create__ ## OperatorPropertyType ## __)

}  // namespace mxnet
#endif  // MXNET_REGISTRY_H_
