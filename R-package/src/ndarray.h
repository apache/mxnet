/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.h
 * \brief Rcpp NDArray interface of MXNet
 */
#ifndef MXNET_RCPP_NDARRAY_H_
#define MXNET_RCPP_NDARRAY_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>

namespace mxnet {
namespace R {  // NOLINT(*)

// forward declare NDArrayFunction
class NDArrayFunction;

class NDArray {
 public:
  /*! \brief default constructor */
  NDArray() {}
  /*!
   * \brief create a R object that correspond to the NDArray
   * \param handle the NDArrayHandle needed for output.
   * \param writable Whether the NDArray is writable or not.
   */
  static SEXP RObject(NDArrayHandle handle, bool writable = true) {
    NDArray *nd = new NDArray();
    nd->handle_ = handle;
    nd->writable_ = writable;
    // will call destructor after finalize
    return Rcpp::XPtr<NDArray>(nd, true);
  }
  /*!
   * \brief Load a list of ndarray from the file.
   * \param filename the name of the file.
   * \return R List of NDArrays
   */
  static SEXP Load(const std::string& filename);
  /*!
   * \brief Save a list of NDArray to file.
   * \param data R List of NDArrays
   * \param filename The name of the file to be saved.
   */
  static void Save(SEXP data, const std::string& filename);
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

  /*! \brief destructor */
  ~NDArray() {
    // free the handle
    MX_CALL(MXNDArrayFree(handle_));
  }

 private:
  // declare friend class
  friend class NDArrayFunction;
  /*! \brief handle to the NDArray */
  NDArrayHandle handle_;
  /*! \brief Whether the NDArray is writable */
  bool writable_;
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

  virtual void signature(std::string& s, const char* name) {
    ::Rcpp::signature< ::Rcpp::void_type >(s, name);
  }

  virtual const char* get_name() {
    return name_.c_str();
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
  // number of mutate variables
  mx_uint num_mutate_vars_;
  // number of arguments
  mx_uint num_args_;
  // whether it accept empty output
  bool accept_empty_out_;
};
}  // namespace Rcpp
}  // namespace mxnet
#endif  // MXNET_RCPP_NDARRAY_H_
