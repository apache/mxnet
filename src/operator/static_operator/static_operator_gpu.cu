/*!
 * Copyright (c) 2015 by Contributors
 * \file static_operator_gpu.cu
 * \brief GPU specialization of operator code
 * \author Bing Xu
*/
#include <mxnet/base.h>
#include "static_operator-inl.h"

namespace mxnet {
namespace op {

mshadow::Random<gpu> prnd_gpu(0);

template<>
StaticOperator *CreateOperator<gpu>(OpType type) {
  return CreateOperator_<gpu>(type, &prnd_gpu);
}

} // namespace op
} // namespace mxnet

