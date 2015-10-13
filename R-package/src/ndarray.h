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
#include <vector>
#include "./base.h"

namespace mxnet {
namespace R {
// forward declare NDArrayFunction
class NDArrayFunction;

/*! \brief The Rcpp NDArray class of MXNet */
class NDArray : public MXNetMovable<NDArray> {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXNDArray";
  }
  /*! \return convert the NDArray to R's Array */
  Rcpp::NumericVector AsNumericVector() const;
  /*!
   * \brief Return a clone of NDArray.
   *  Do not expose this to R side.
   * \return a new cloned NDArray.
   */
  RObjectType Clone() const;
  /*!
   * \brief Return a slice of NDArray.
   * \return a sliced NDArray.
   */
  RObjectType Slice(mx_uint begin, mx_uint end) const;
  /*! \return The shape of the array */
  inline Rcpp::Dimension shape() const;
  /*! \return The number of elements in the array */
  inline size_t size() const;
  /*! \return the context of the NDArray */
  inline const Context::RObjectType ctx() const {
    return ctx_.RObject();
  }
  /*!
   * \brief internal function to copy NDArray from to to
   *  Do not expose this to R side.
   * \param from The source NDArray.
   * \param to The target NDArray.
   */
  static void CopyFromTo(const NDArray& from, NDArray *to);
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
  /*!
   * \brief Extract NDArrayHandles from List.
   * \param array_list The NDArray list.
   * \param list_name The name of the list, used for error message.
   * \param allow_null If set to True, allow null in the list.
   */
  static std::vector<NDArrayHandle> GetHandles(const Rcpp::List& array_list,
                                               const std::string& list_name,
                                               bool allow_null = false);
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

  /*! \return the handle of NDArray */
  NDArrayHandle handle() const {
    return handle_;
  }
  /*!
   * \brief create a R object that correspond to the Class
   * \param handle the Handle needed for output.
   * \param writable Whether the array is writable.
   */
  inline static Rcpp::RObject RObject(NDArrayHandle handle,
                                      bool writable = true) {
    return Rcpp::internal::make_new_object(new NDArray(handle, writable));
  }

 private:
  // declare friend class
  friend class NDArrayFunction;
  friend class KVStore;
  friend class Executor;
  friend class MXNetMovable<NDArray>;
  // enable trivial operator= etc.
  NDArray() {}
  // constructor
  explicit NDArray(NDArrayHandle handle, bool writable)
      : handle_(handle), writable_(writable) {
    MX_CALL(MXNDArrayGetContext(handle,
                                &ctx_.dev_type,
                                &ctx_.dev_id));
  }
  // finalizer that invoked on non-movable object
  inline void DoFinalize() {
    MX_CALL(MXNDArrayFree(handle_));
  }
  // Create a new Object that is moved from current one
  inline NDArray* CreateMoveObject() {
    NDArray* moved = new NDArray();
    *moved = *this;
    return moved;
  }
  /*! \brief The context of the NDArray */
  Context ctx_;
  /*! \brief Whether the NDArray is writable */
  bool writable_;
  /*! \brief The internal handle to the object */
  NDArrayHandle handle_;
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

  // internal helper function to search function handle
  static FunctionHandle FindHandle(const std::string& hname);

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

// implementatins of inline functions
inline size_t NDArray::size() const {
  Rcpp::Dimension dim = this->shape();
  size_t sz = 1;
  for (size_t i =0; i < dim.size(); ++i) {
    sz *= dim[i];
  }
  return sz;
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
