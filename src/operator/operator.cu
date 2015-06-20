/*!
 * Copyright (c) 2015 by Contributors
 * \file operator.cu
 * \brief
 * \author Bing Xu
*/


#include "operator_helper.h"

namespace mxnet {
namespace op {

Operator * CreateOperator(OpType type) {
  return OperatorFactory<gpu>(type);
}

} // namespace op
} // namespace mxnet

