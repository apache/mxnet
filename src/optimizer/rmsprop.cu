/*!
 * Copyright (c) 2015 by Contributors
 * \file rmsprop.cc
 * \brief rmsprop optimizer
*/
#include "./rmsprop-inl.h"

namespace mxnet {
namespace opt {

void call_rmsprop_update_gpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob cache,
                              float lr, const RMSPropParam& param) {
  rmsprop_update<gpu>(ctx, weight, grad, cache, lr, param);
}

}  // namespace opt
}  // namespace mxnet
