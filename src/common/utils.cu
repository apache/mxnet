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
void CastStorageDispatch<gpu>(mshadow::Stream<gpu>* s,
                              const NDArray& input,
                              const NDArray& output) {
  mxnet::op::CastStorageComputeImpl(s, input, output);
}

}  // namespace common
}  // namespace mxnet
