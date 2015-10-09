/*!
 *  Copyright (c) 2015 by Contributors
 * \file executor.h
 * \brief Rcpp Symbolic execution interface of MXNet
 */
#ifndef MXNET_RCPP_EXECUTOR_H_
#define MXNET_RCPP_EXECUTOR_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
#include <string>
#include "./base.h"

namespace mxnet {
namespace R {
// forward declare symbol
class Symbol;

/*! \brief The Rcpp Symbol class of MXNet */
class Executor : public MXNetClassBase<Executor, ExecutorHandle, MXExecutorFree> {
 public:
  /*! \brief The type of Symbol in R's side */
  typedef Rcpp::RObject RObjectType;
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

 private:
  // friend with symbol
  friend class Symbol;
  friend class MXNetClassBase<Executor, ExecutorHandle, MXExecutorFree>;
  // internal constructor, enable trivial operator=
  Executor() {}
  explicit Executor(ExecutorHandle handle) {
    this->handle_ = handle;
  }
};
}  // namespace R
}  // namespace mxnet
#endif  // MXNET_RCPP_EXECUTOR_H_
