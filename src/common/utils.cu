/*!
 * Copyright (c) 2017 by Contributors
 * \file utils.cu
 * \brief gpu implementation of util functions
 */

#include "./utils.h"
#include "../operator/tensor/cast_storage-inl.h"

namespace mxnet {
namespace common {

template<>
void CastStorageDispatch<gpu>(const OpContext& ctx,
                              const NDArray& input,
                              const NDArray& output) {
  mxnet::op::CastStorageComputeImpl<gpu>(ctx, input, output);
}

}  // namespace common
}  // namespace mxnet
