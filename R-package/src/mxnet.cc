#include <Rcpp.h>
#include "./ndarray.h"

RCPP_MODULE(mxnet) {
  using namespace mxnet::R; // NOLINT(*)
  NDArray::InitRcppModule();
  NDArrayFunction::InitRcppModule();
}
