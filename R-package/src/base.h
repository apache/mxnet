/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief Rcpp interface of MXNet
 */
#ifndef MXNET_RCPP_BASE_H_
#define MXNET_RCPP_BASE_H_

#include <Rcpp.h>
#include <dmlc/base.h>
#include <mxnet/c_api.h>

namespace mxnet {
namespace R {

// change to Rcpp::cerr later, for compatiblity of older version for now
#define RLOG_FATAL ::Rcpp::Rcerr

// checking macro for R side
#define RCHECK(x)                                           \
  if (!(x))                                                 \
    RLOG_FATAL << "Check "                                  \
        "failed: " #x << ' '

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
