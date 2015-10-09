/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.h
 * \brief Rcpp NDArray interface of MXNet
 */
#ifndef MXNET_RCPP_NDARRAY_H_
#define MXNET_RCPP_NDARRAY_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
#include <string>
#include <algorithm>

namespace mxnet {
namespace R {
// forward declare NDArrayFunction
class NDArrayFunction;

/*! \brief The Rcpp NDArray class of MXNet */
class NDArray : public MXNetClassBase<NDArray, NDArrayHandle, MXNDArrayFree> {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXNDArray";
  }
  /*! \return convert the NDArray to R's Array */
  Rcpp::NumericVector AsNumericVector() const;
  /*! \return The shape of the array */
  inline Rcpp::Dimension shape() const;
  /*! \return the context of the NDArray */
  inline const Context &ctx() const {
    return ctx_;
  }
  /*!
   * \brief Load a list of ndarray from the file.
   * \param filename the name of the file.
   * \return R List of NDArrays
   */
  static Rcpp::List Load(const std::string& filename);
  /*!
   * \brief Save a list of NDArray to file.
   * \param data R List of NDArrays
   * \param filename The name of the file to be saved.
   */
  static void Save(const Rcpp::RObject& data,
                   const std::string& filename);
  /*!
   * \brief function to create an empty array
   * \param shape The shape of the Array
   * \return a new created MX.NDArray
   */
  static RObjectType Empty(const Rcpp::Dimension& shape,
                           const Context::RObjectType& ctx);
  /*!
   * \brief Create a MX.NDArray by copy data from src R array.
   * \param src the source R array
   * \param ctx The context where
   */
  static RObjectType Array(const Rcpp::RObject& src,
                           const Context::RObjectType& ctx);
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

 private:
  friend MXNetClassBase<NDArray, NDArrayHandle, MXNDArrayFree>;
  // enable trivial operator= etc.
  NDArray() {}
  // constructor
  explicit NDArray(NDArrayHandle handle)
      : writable_(true) {
    this->handle_ = handle;
    MX_CALL(MXNDArrayGetContext(handle,
                                &ctx_.dev_type,
                                &ctx_.dev_id));
  }
  // Create a new Object that is moved from current one
  inline NDArray* CreateMoveObject() {
    NDArray* moved = new NDArray();
    *moved = *this;
    return moved;
  }
  // declare friend class
  friend class NDArrayFunction;
  /*! \brief The context of the NDArray */
  Context ctx_;
  /*! \brief Whether the NDArray is writable */
  bool writable_;
  /*! \brief Whether this object has been moved to another object */
  bool moved_;
};

/*! \brief The NDArray functions to be invoked */
class NDArrayFunction : public ::Rcpp::CppFunction {
 public:
  virtual SEXP operator() (SEXP * args);

  virtual int nargs() {
    return num_args_;
  }

  virtual bool is_void() {
    return false;
  }

  virtual void signature(std::string& s, const char* name) {  // NOLINT(*)
    ::Rcpp::signature< ::Rcpp::void_type >(s, name);
  }

  virtual const char* get_name() {
    return name_.c_str();
  }

  virtual SEXP get_formals() {
    return formals_;
  }

  virtual DL_FUNC get_function_ptr() {
    return (DL_FUNC)NULL; // NOLINT(*)
  }
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

 private:
  // make constructor private
  explicit NDArrayFunction(FunctionHandle handle);

  /*! \brief internal functioon handle. */
  FunctionHandle handle_;
  // name of the function
  std::string name_;
  // beginning position of use vars
  mx_uint begin_use_vars_;
  // number of use variable
  mx_uint num_use_vars_;
  // beginning position of scalars
  mx_uint begin_scalars_;
  // number of scalars
  mx_uint num_scalars_;
  // begining of mutate variables
  mx_uint begin_mutate_vars_;
  // number of mutate variables
  mx_uint num_mutate_vars_;
  // number of arguments
  mx_uint num_args_;
  // whether it accept empty output
  bool accept_empty_out_;
  // ther formals of arguments
  Rcpp::List formals_;
};
}  // namespace R
}  // namespace mxnet


RCPP_EXPOSED_CLASS_NODECL(::mxnet::R::NDArray)

namespace mxnet {
namespace R {
// implementatins of inline functions
inline Rcpp::Dimension NDArray::shape() const {
  mx_uint ndim;
  const mx_uint *pshape;
  MX_CALL(MXNDArrayGetShape(
      handle_, &ndim, &pshape));
  Rcpp::IntegerVector vec(ndim);
  std::copy(pshape, pshape + ndim, vec.begin());
  SEXP sexp = vec;
  return sexp;
}
}  // namespace R
}  // namespace mxnet

namespace Rcpp {
  template<>
  inline bool is<mxnet::R::NDArray>(SEXP x) {
    return internal::is__module__object_fix<mxnet::R::NDArray>(x);
  }
}
#endif  // MXNET_RCPP_NDARRAY_H_
