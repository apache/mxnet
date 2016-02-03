/*!
 * Copyright (c) 2015 by Contributors
 * \file sgd.cc
 * \brief sgd optimizer
*/
#include <mxnet/ndarray.h>
#include "./sgd-inl.h"


namespace mxnet {
namespace opt {

void call_sgd_mom_update_cpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                              float lr, float wd, const SGDParam& param) {
  sgd_mom_update<cpu>(ctx, weight, grad, mom, lr, wd, param);
}
void call_sgd_update_cpu(RunContext ctx, TBlob weight, const TBlob grad,
                          float lr, float wd, const SGDParam& param) {
  sgd_update<cpu>(ctx, weight, grad, lr, wd, param);
}

DMLC_REGISTER_PARAMETER(SGDParam);

MXNET_REGISTER_OPTIMIZER(ccsgd, SGDOpt)
.describe("Stochastic gradient decent optimizer implemented in C++.");

}  // namespace opt
}  // namespace mxnet
