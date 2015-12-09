/*!
 * Copyright (c) 2015 by Contributors
 * \file sgd.cc
 * \brief sgd optimizer
*/
#include <mxnet/ndarray.h>
#include "./sgd-inl.h"


namespace mxnet {
namespace opt {

DMLC_REGISTER_PARAMETER(SGDParam);

MXNET_REGISTER_OPTIMIZER(ccsgd, SGDOpt)
.describe("Stochastic gradient decent optimizer implemented in C++.");

}  // namespace opt
}  // namespace mxnet
