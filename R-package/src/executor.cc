/*!
 *  Copyright (c) 2015 by Contributors
 * \file executor.h
 * \brief Rcpp Symbol of MXNet.
 */
#include <Rcpp.h>
#include <string>
#include <algorithm>
#include "./base.h"
#include "./executor.h"

namespace mxnet {
namespace R {

void Executor::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  class_<Executor>("MXExecutor")
      .finalizer(&Executor::Finalizer);
}

}  // namespace R
}  // namespace mxnet
