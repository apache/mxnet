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

#if DMLC_USE_CXX11 == 0
#define nullptr NULL
#endif

/*! \brief Context of device enviroment */
struct Context {
  /*! \brief The device ID of the context */
  int dev_type;
  /*! \brief The device ID of the context */
  int dev_id;
  /*! \brief The R object type of the context */
  typedef Rcpp::List RObjectType;
  /*! \brief default constructor  */
  Context() {}
  /*!
   * \brief Constructor
   * \param src source R representation.
   */
  explicit Context(const Rcpp::RObject& src) {
    Rcpp::List list(src);
    Context ctx;
    ctx.dev_type = list["device_typeid"];
    ctx.dev_id = list["device_id"];
  }
  /*! \return R object representation of the context */
  inline Rcpp::List RObject() const {
    const char *dev_name = "cpu";
    if (dev_type == kGPU) dev_name = "gpu";
    return Rcpp::List::create(
        Rcpp::Named("device") = dev_name,
        Rcpp::Named("device_id") = dev_id,
        Rcpp::Named("device_typeid") = dev_type);
  }
  static const int kGPU = 2;
  static const int kCPU = 1;
};
}  // namespace R
}  // namespace mxnet
#endif  // MXNET_RCPP_BASE_H_
