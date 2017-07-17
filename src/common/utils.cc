/*!
 * Copyright (c) 2017 by Contributors
 * \file utils.cc
 * \brief cpu implementation of util functions
 */

#include "./utils.h"
#include "../operator/tensor/cast_storage-inl.h"

namespace mxnet {
namespace common {

template<>
void CastStorageDispatch<cpu>(const OpContext& ctx,
                              const NDArray& input,
                              const NDArray& output) {
  mxnet::op::CastStorageComputeImpl<cpu>(ctx, input, output);
}

}  // namespace common
}  // namespace mxnet
