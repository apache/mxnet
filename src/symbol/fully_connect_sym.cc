  /*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_sym.cc
 * \brief fully connect operator symbol
*/
#include "./fully_connect_sym-inl.h"

namespace mxnet {
  std::vector<std::string> FullyConnectSymbol::DescribeArguments() const {
    std::string ret[] = {"data", "weight", "bias"};
    if (param_.no_bias == 0) {
      return std::vector<std::string>(ret, ret + 3);
    } else {
      return std::vector<std::string>(ret, ret + 2);
    }
  }

  std::vector<std::string> FullyConnectSymbol::DescribeReturns() const {
    std::string temp = "output";
    std::vector<std::string> v;
    v.push_back(temp);
    return v;
  }

  void FullyConnectSymbol::SetParam(const char *name, const char *val) {              
    param_.SetParam(name, val);
  }

  bool FullyConnectSymbol::InferShape(std::vector<TShape> *in_shape,
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
    return true;
  }

  /*!
   * \brief Copy this AtomicSymbol and returns a pointer to the copied object.
   *  this is a function because different subclass of AtomicSymbol would copy differently.
   * \return a pointer to the copied atomic symbol
   */
  AtomicSymbol* FullyConnectSymbol::Copy() const {
    FullyConnectSymbol* fc_sym = new FullyConnectSymbol();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }
  std::string FullyConnectSymbol::TypeString() const {
    return "Fully Connected";
  }

  template<>
  StaticOperator* FullyConnectSymbol::Bind<cpu>(Context ctx) const {
    return Bind_<cpu>(ctx);
  }
}
