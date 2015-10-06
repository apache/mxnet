/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief Rcpp interface of MXNet
 */
#ifndef MXNET_RCPP_BASE_H_
#define MXNET_RCPP_BASE_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
// to be removed
#include <dmlc/logging.h>

namespace mxnet {
namespace R {

// change to Rcpp::cerr later, for compatiblity of older version for now
#define RLOG_FATAL LOG(FATAL)

/*!
 * \brief protected MXNet C API call, report R error if happens.
 * \param func Expression to call.
 */
#define MX_CALL(func)                                              \
  {                                                                \
    int e = (func);                                                \
    if (e != 0) {                                                  \
      RLOG_FATAL << MXGetLastError();                              \
    }                                                              \
  }
}  // namespace Rcpp
}  // namespace mxnet
#endif  // MXNET_RCPP_BASE_H_
