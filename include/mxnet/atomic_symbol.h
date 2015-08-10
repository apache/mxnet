/*!
 *  Copyright (c) 2015 by Contributors
 * \file atomic_symbol.h
 * \brief atomic symbol interface of mxnet
 */
#ifndef MXNET_ATOMIC_SYMBOL_H_
#define MXNET_ATOMIC_SYMBOL_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "./base.h"
#include "./tensor_blob.h"

namespace mxnet {
// forward declare StaticOperator
class StaticOperator;
/*!
 * \brief AtomicSymbol is the base class of all atomic symbols.
 *  This is not meant to be used by user, it should be wrapped in Symbol, so that the same instance
 *  of AtomicSymbol can be shared in the graphs of different Symbols
 */
class AtomicSymbol {
 public:
  /*!
   * \brief virtual destructor
   */
  virtual ~AtomicSymbol() {}
  /*! \brief get the descriptions of inputs for this symbol */
  virtual std::vector<std::string> ListArguments() const {
    // default implementation returns "data"
    return std::vector<std::string>(1, std::string("data"));
  }
  /*! \brief get the descriptions of outputs for this symbol */
  virtual std::vector<std::string> ListReturns() const {
    // default implementation returns "output"
    return std::vector<std::string>(1, std::string("output"));
  }
  /*!
   *  \brief set param for the symbol from string
   *  \param name parameter name
   *  \param val string for the configuration
   */
  virtual void SetParam(const char *name, const char *val) {}
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by DescribeArgs
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return if the shape inference is successful, return true, else return false.
   */
  virtual bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape) const = 0;
  /*!
   * \brief Copy this AtomicSymbol and returns a pointer to the copied object.
   *  this is a virtual function because different subclass of AtomicSymbol would copy differently.
   * \return a pointer to the copied atomic symbol
   */
  virtual AtomicSymbol* Copy() const = 0;
  /*!
   * \brief Bind this AtomicSymbol to a context and get back a static operator
   *  Bind function of AtomicSymbol does not return NArrayOperator, but static operator.
   *  Calling bind from the Symbol wrapper would generate a NArrayOperator.
   */
  template<typename xpu>
  StaticOperator* Bind(Context ctx) const;
  /*!
   * \brief return the type string of the atomic symbol
   *  subclasses override this function.
   */
  virtual std::string TypeString() const = 0;
  friend class Symbol;

  /*!
   * \brief create atomic symbol by type name
   * \param type_name the type string of the AtomicSymbol
   * \return a new constructed AtomicSymbol
   */
  static AtomicSymbol *Create(const char* type_name);
};

}  // namespace mxnet
#endif  // MXNET_ATOMIC_SYMBOL_H_
