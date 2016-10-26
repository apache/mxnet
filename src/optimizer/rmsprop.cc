/*!
 * Copyright (c) 2015 by Contributors
 * \file rmsprop.cc
 * \brief rmsprop optimizer
*/
#include <mxnet/ndarray.h>
#include "./rmsprop-inl.h"


namespace mxnet {
namespace opt {

void call_rmsprop_update_cpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob cache,
                              float lr, const RMSPropParam& param) {
  rmsprop_update<cpu>(ctx, weight, grad, cache, lr, param);
}

DMLC_REGISTER_PARAMETER(RMSPropParam);

MXNET_REGISTER_OPTIMIZER(ccrmsprop, RMSPropOpt)
.describe("RMSProp adaptive learning rate method implemented in C++.");

}  // namespace opt
}  // namespace mxnet
