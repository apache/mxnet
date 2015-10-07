/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief Rcpp interface of MXNet
 *  All the interface is done through C API,
 *  to achieve maximum portability when we need different compiler for libmxnet.
 */
#ifndef MXNET_RCPP_BASE_H_
#define MXNET_RCPP_BASE_H_

#include <Rcpp.h>
#include <dmlc/base.h>
#include <mxnet/c_api.h>

/*! \brief namespace of mxnet */
namespace mxnet {
/*! \brief namespace of R package */
namespace R {

/*! \brief LOG FATAL to report error to R console */
#define RLOG_FATAL ::Rcpp::Rcerr

/*!
 * \brief Checking macro for Rcpp code, report error ro R console
 * \code
 *  RCHECK(data.size() == 1) << "Data size must be 1";
 * \endcode
 */
#define RCHECK(x)                                           \
  if (!(x)) RLOG_FATAL << "Check failed: " #x << ' ' /* NOLINT(*) */

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

/*! \brief macro to be compatible with non c++11 env */
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
    this->dev_id = list[1];
    this->dev_type = list[2];
  }
  /*! \return R object representation of the context */
  inline RObjectType RObject() const {
    const char *dev_name = "cpu";
    if (dev_type == kGPU) dev_name = "gpu";
    return Rcpp::List::create(
        Rcpp::Named("device") = dev_name,
        Rcpp::Named("device_id") = dev_id,
        Rcpp::Named("device_typeid") = dev_type);
  }
  /*!
   * Create a CPU context.
   * \param dev_id the device id.
   * \return CPU Context.
   */
  inline static RObjectType CPU(int dev_id = 0) {
    Context ctx;
    ctx.dev_type = kCPU;
    ctx.dev_id = dev_id;
    return ctx.RObject();
  }
  /*!
   * Create a GPU context.
   * \param dev_id the device id.
   * \return GPU Context.
   */
  inline static RObjectType GPU(int dev_id) {
    Context ctx;
    ctx.dev_type = kGPU;
    ctx.dev_id = dev_id;
    return ctx.RObject();
  }
  /*! \brief initialize all the Rcpp module functions */
  inline static void InitRcppModule() {
    using namespace Rcpp;  // NOLINT(*);
    function("mx.cpu", &CPU);
    function("mx.gpu", &GPU);
  }
  /*! \brief the device type id for CPU */
  static const int kCPU = 1;
  /*! \brief the device type id for GPU */
  static const int kGPU = 2;
};
}  // namespace R
}  // namespace mxnet
#endif  // MXNET_RCPP_BASE_H_
