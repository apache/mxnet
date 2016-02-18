/*!
 * Copyright (c) 2015 by Contributors
 * \file sgd.cc
 * \brief sgd optimizer
*/
#include "./sgd-inl.h"

namespace mxnet {
namespace opt {

void call_sgd_mom_update_gpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                              float lr, float wd, const SGDParam& param) {
  sgd_mom_update<gpu>(ctx, weight, grad, mom, lr, wd, param);
}
void call_sgd_update_gpu(RunContext ctx, TBlob weight, const TBlob grad,
                          float lr, float wd, const SGDParam& param) {
  sgd_update<gpu>(ctx, weight, grad, lr, wd, param);
}

}  // namespace opt
}  // namespace mxnet
