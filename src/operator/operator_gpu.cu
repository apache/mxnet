/*!
 * Copyright (c) 2015 by Contributors
 * \file operator_gpu.cu
 * \brief GPU specialization of operator code
 * \author Bing Xu
*/
#include <mxnet/base.h>
#include <mxnet/tensor_blob.h>
#include "operator-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOperator<gpu>(OpType type,
                              mshadow::Random<gpu> *prnd) {
  return CreateOperator_<gpu>(type, prnd);
}

} // namespace op
} // namespace mxnet

