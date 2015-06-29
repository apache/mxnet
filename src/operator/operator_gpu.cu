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

mshadow::Random<gpu> prnd_gpu(0);

template<>
Operator *CreateOperator<gpu>(OpType type) {
  return CreateOperator_<gpu>(type, &prnd_gpu);
}

} // namespace op
} // namespace mxnet

