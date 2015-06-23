/*!
 * Copyright (c) 2015 by Contributors
 * \file operator_cpu.cc
 * \brief CPU specialization of operator codes
 * \author Bing Xu
*/
#include "./operator-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOperator<cpu>(OpType type,
                              mshadow::Random<cpu> *prnd) {
  return CreateOperator_<cpu>(type, prnd);
}

}  // namespace op
}  // namespace mxnet
