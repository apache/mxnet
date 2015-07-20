 /*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator
 * \author Bing Xu
*/
#ifndef MXNET_SYMBOL_FULLY_CONNECT_SYM_INL_H_
#define MXNET_SYMBOL_FULLY_CONNECT_SYM_INL_H_

#include <dmlc/logging.h>
#include <mxnet/static_operator.h>
#include <mxnet/atomic_symbol.h>
#include <vector>
#include "../static_operator/fully_connect_op-inl.h"
#include "../static_operator/param.h"

namespace mxnet {
using namespace mxnet::op;

class FullyConnectSymbol : public AtomicSymbol {
 public:
  virtual std::vector<std::string> DescribeArguments() const;

  virtual std::vector<std::string> DescribeReturns() const;

  virtual void SetParam(const char *name, const char *val);

  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const;

  /*!
   * \brief Copy this AtomicSymbol and returns a pointer to the copied object.
   *  this is a virtual function because different subclass of AtomicSymbol would copy differently.
   * \return a pointer to the copied atomic symbol
   */
  virtual AtomicSymbol* Copy() const;

  template<typename xpu>
  StaticOperator* Bind(Context ctx) const;
  /*!
   * \brief Bind this AtomicSymbol to a context and get back a static operator
   *  Bind function of AtomicSymbol does not return Operator, but static operator.
   *  Calling bind from the Symbol wrapper would generate a Operator.
   */
  template<typename xpu>
  StaticOperator* Bind_(Context ctx) const {
    return new FullyConnectOp<xpu>(param_);
  }

  virtual std::string TypeString() const;
 private:
  Param param_;
};  // class FullyConnectSymbol
}  // namespace mxnet

#endif  // MXNET_SYMBOL_FULLY_CONNECT_SYM_INL_H_

