/*!
 *  Copyright (c) 2015 by Contributors
 * \file mxnet.cc
 * \brief The registry of all module functions and objects
 */
#include <Rcpp.h>
#include <fstream>
#include "./base.h"
#include "./ndarray.h"
#include "./symbol.h"
#include "./executor.h"
#include "./io.h"
#include "./kvstore.h"
#include "./export.h"

namespace mxnet {
namespace R {
void SetSeed(int seed) {
  MX_CALL(MXRandomSeed(seed));
}

void NotifyShutdown(int seed) {
  MX_CALL(MXNotifyShutdown());
}

// init rcpp module in base
void InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  function("mx.internal.set.seed", &SetSeed);
  function("mx.internal.notify.shudown", &NotifyShutdown);
}
}  // namespace R
}  // namespace mxnet


RCPP_MODULE(mxnet) {
  using namespace mxnet::R; // NOLINT(*)
  mxnet::R::InitRcppModule();
  Context::InitRcppModule();
  NDArray::InitRcppModule();
  NDArrayFunction::InitRcppModule();
  Symbol::InitRcppModule();
  SymbolFunction::InitRcppModule();
  Executor::InitRcppModule();
  DataIter::InitRcppModule();
  DataIterCreateFunction::InitRcppModule();
  KVStore::InitRcppModule();
  Exporter::InitRcppModule();
}
