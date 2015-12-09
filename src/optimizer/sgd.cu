/*!
 * Copyright (c) 2015 by Contributors
 * \file sgd.cc
 * \brief sgd optimizer
*/
#include "./sgd-inl.h"

namespace mxnet {
namespace opt {

void call_sgd_mom_update_gpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                              float lr, const SGDParam& param) {
  sgd_mom_update<gpu>(ctx, weight, grad, mom, lr, param);
}
void call_sgd_update_gpu(RunContext ctx, TBlob weight, const TBlob grad,
                          float lr, const SGDParam& param) {
  sgd_update<gpu>(ctx, weight, grad, lr, param);
}

}  // namespace opt
}  // namespace mxnet
