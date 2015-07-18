 /*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator
 * \author Bing Xu
*/
#ifndef MXNET_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_
#define MXNET_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_

#include <dmlc/logging.h>
#include <mxnet/static_operator.h>
#include <vector>
#include "./static_operator_common.h"
#include "./param.h"
#include "../static_operator/fully_connect_op-inl.h"

namespace mxnet {
template<typename xpu>
class FullyConnectSymbol : public AtomicSymbol {
 public:
  virtual std::vector<std::string> DescribeArguments() const {
    std::string ret[] = {"data", "weight", "bias"};
    if (param_.no_bias == 0) {
      return std::vector<std::string>(ret, ret + 3);
    } else {
      return std::vector<std::string>(ret, ret + 2);
    }
  }

  virtual std::vector<std::string> DescribeReturns() const {
  	return std::vector();
  }

  virtual void SetParam(const char *name, const char *val) const {
    param_.SetParam(name, val);
  }
  virtual void InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    using namespace mshadow;
    if (param_.no_bias == 0) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    CHECK_GT(param_.num_hidden, 0);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
        "Input data should be 4D in batch-1-1-hidden";
    CHECK_NE(dshape.ndim(), 0) << "Require data shape to be known";
    ShapeAssignCheck((*in_shape)[1], Shape2(param_.num_hidden, dshape[3]));
    if (param_.no_bias == 0) {
      ShapeAssignCheck((*in_shape)[2], Shape1(param_.num_hidden));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    (*out_shape)[0][3] = param_.num_hidden;
  }

  /*!
   * \brief Copy this AtomicSymbol and returns a pointer to the copied object.
   *  this is a virtual function because different subclass of AtomicSymbol would copy differently.
   * \return a pointer to the copied atomic symbol
   */
  virtual AtomicSymbol* Copy() const {
  	FullyConnectSymbol* fc_sym = new FullyConnectSymbol();
  	fc_sym->param = this->param;
  	return fc_sym;
  }
  /*!
   * \brief Bind this AtomicSymbol to a context and get back a static operator
   *  Bind function of AtomicSymbol does not return Operator, but static operator.
   *  Calling bind from the Symbol wrapper would generate a Operator.
   */
  virtual StaticOperator* Bind(Context ctx) const {
  	return new FullyConnectOp<xpu>(param_);
  }

  virtual std::string TypeString() const {
  	return "Fully Connected";
  }
 private:
  Param param_;
};  // class FullyConnectSymbol
}  // namespace mxnet

#endif  // MXNET_STATIC_OPERATOR_FULLY_CONNECT_OP_INL_H_

