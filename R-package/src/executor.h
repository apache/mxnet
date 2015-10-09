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
#include "./symbol.h"

namespace mxnet {
namespace R {
// forward declare symbol
class Symbol;

/*! \brief The Rcpp Symbol class of MXNet */
class Executor : public MXNetMovable<Executor> {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXExecutor";
  }
  /*!
   * \return Get the arg arrays of executor.
   */
  Rcpp::List GetArgArrays() const {
    return CloneArray(*arg_arrays_);
  }
  /*!
   * \return Get the grad arrays of executor.
   */
  Rcpp::List GetGradArrays() const {
    RCHECK(grad_arrays_ != nullptr)
        << "This executor has not been binded with req.grad";
    return CloneArray(*grad_arrays_);
  }
  /*!
   * \return Get the auxiliary arrays of executor.
   */
  Rcpp::List GetAuxArrays() const {
    return CloneArray(*aux_arrays_);
  }
  /*!
   * \return Get the output arrays of executor.
   */
  Rcpp::List GetOuputArrays() const {
    return CloneArray(*out_arrays_);
  }
  /*!
   * \brief Set the arg_arrays of executor.
   * \param exec The executor R object, this object will be MOVED.
   * \return a result executor, moved from exec.
   */
  static RObjectType SetArgArray(const RObjectType& exec, const Rcpp::List& array);
  /*!
   * \brief Set the aux_arrays of executor.
   * \param exec The executor R object, this object will be MOVED.
   * \return a result executor, moved from exec.
   */
  static RObjectType SetAuxArray(const RObjectType& exec, const Rcpp::List& array);
  /*!
   * \brief Peform a forward operation on exec, this will set the out_arrays.
   * \param exec The executor R object, this object will be MOVED.
   * \param is_train whether it is training phase.
   * \param kwargs additional parameters.
   * \return a result executor, moved from exec.
   */
  static RObjectType Forward(const RObjectType& exec,
                             bool is_train,
                             const Rcpp::List& kwargs);
  /*!
   * \brief Peform a backward operation on exec, this will set the grad_arrays.
   * \param exec The executor R object, this object will be MOVED.
   * \param output_grads the gradient on outputs, to be propagated back.
   * \return a result executor, moved from exec.
   */
  static RObjectType Backward(const RObjectType& exec, const Rcpp::List& output_grads);
  /*!
   * \brief Create a new R Executor by bind on symbol
   * \param symbol The R symbol to bind.
   * \param context The device to bind.
   * \param arg_arrays The argument arrays giving the initial value of arguments.
   * \param aux_arrays The auxiliary state arrays giving the initial value of auxiliary states.
   * \param grad_reqs Array of booleans, giving the requirements of gradient.
   */
  static RObjectType Bind(const Symbol::RObjectType& symbol,
                          const Context::RObjectType& context,
                          const Rcpp::List& arg_arrays,
                          const Rcpp::List& aux_arrays,
                          const Rcpp::List& grad_reqs);
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();
  // destructor
  ~Executor() {
    // delete can handle nullptr safely
    delete out_arrays_;
    delete arg_arrays_;
    delete grad_arrays_;
    delete aux_arrays_;
  }

 private:
  // friend with symbol
  friend class Symbol;
  friend class MXNetMovable<Executor>;
  // internal constructor, enable trivial operator=
  Executor()
      : out_arrays_(nullptr),
        arg_arrays_(nullptr),
        grad_arrays_(nullptr),
        aux_arrays_(nullptr) {}
  /*! \return a new Object that is moved from current one */
  inline Executor* CreateMoveObject() {
    Executor *moved = new Executor();
    *moved = *this;
    out_arrays_ = nullptr;
    arg_arrays_ = nullptr;
    grad_arrays_ = nullptr;
    aux_arrays_ = nullptr;
    return moved;
  }
  // finalizer that invoked on non-movable object
  inline void DoFinalize() {
    MX_CALL(MXExecutorFree(handle_));
  }
  /*!
   * \brief Clone src into a new space.
   * \param src source list of arrays to clone.
   * \return A cloned list of arrays under same context.
   */
  static Rcpp::List CloneArray(const Rcpp::List& src);
  /*!
   * \brief Copy arrays from to to
   * \param from source list to copy from.
   * \param to target list to copy to.
   */
  static void CopyArray(const Rcpp::List& from, Rcpp::List *to);
  /*! \brief output arrays of Executor */
  Rcpp::List *out_arrays_;
  /*! \brief argument arrays of Executor */
  Rcpp::List *arg_arrays_;
  /*! \brief gradient arrays of Executor */
  Rcpp::List *grad_arrays_;
  /*! \brief auxiliary arrays of Executor */
  Rcpp::List *aux_arrays_;
  /*! \brief internal executor handle */
  ExecutorHandle handle_;
};
}  // namespace R
}  // namespace mxnet

RCPP_EXPOSED_CLASS_NODECL(::mxnet::R::Executor);

namespace Rcpp {
  template<>
  inline bool is<mxnet::R::Executor>(SEXP x) {
    return internal::is__module__object_fix<mxnet::R::Executor>(x);
  }
}

#endif  // MXNET_RCPP_EXECUTOR_H_
