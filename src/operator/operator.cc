/*!
 * Copyright (c) 2015 by Contributors
 * \file operator.cc
 * \brief
 * \author Bing Xu
*/
#include "operator_helper.h"

namespace mxnet {
namespace op {

Operator * CreateOperator(OpType type) {
  return OperatorFactory<cpu>(type);
}

} // namespace op
} // namespace mxnet

