/*!
 * Copyright (c) 2015 by Contributors
 * \file operator_cpu.cc
 * \brief CPU specialization of operator codes
 * \author Bing Xu
*/
#include "./operator-inl.h"

namespace mxnet {
namespace op {
// todo add managing for prnd
mshadow::Random<cpu> prnd_cpu(0);

template<>
Operator *CreateOperator<cpu>(OpType type) {
  return CreateOperator_<cpu>(type, &prnd_cpu);
}

}  // namespace op
}  // namespace mxnet
