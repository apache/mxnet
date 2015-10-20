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

/*! \brief Back-end chunk of NDArray */
struct NDBlob {
 public:
  /*!
   * \brief constructor
   * \param handle The handle
   */
  NDBlob(NDArrayHandle handle, bool writable)
      : handle(handle), writable(writable), moved(false) {
  }
  /*! \brief destructor */
  ~NDBlob() {
    if (!moved) {
      MX_CALL(MXNDArrayFree(handle));
    }
  }
  /*! \brief The internal handle of NDArray */
  NDArrayHandle handle;
  /*! \brief whether the Blob is writable */
  bool writable;
  /*! \brief whether if the */
  bool moved;
};

/*!
 * \brief Rcpp NDArray object of MXNet.
 *  We use lightweight Rcpp external ptr and S3 type object.
 *  For efficiently expose the object to R side.
 */
class NDArray  {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXNDArray";
  }
  /*! \brief The returning type of new NDArray */
  typedef Rcpp::XPtr<NDBlob> RObjectType;
  /*!
   * \brief copy constructor
   * \param other Another NDArray to be copied from.
   */
  NDArray(const NDArray& other)
      : ptr_(other.ptr_) {}
  /*!
   * \brief constructor from R SEXP
   * \param src The source SEXP
   */
  explicit NDArray(SEXP src)
      : ptr_(src) {}
  /*!
   * \brief Constructor
   * \param handle The handle
   */
  NDArray(NDArrayHandle handle, bool writable)
      : ptr_(new NDBlob(handle, writable)) {
    ptr_.attr("class") = "MXNDArray";
  }
  /*! \return RObject representation */
  inline RObjectType RObject()  const {
    return ptr_;
  }
  /*!
   * \brief Create a new moved NDArray
   */
  inline NDArray Move() const {
    RCHECK(ptr_->writable)
        << "Passing a read only NDArray to mutate function";
    ptr_->moved = true;
    return NDArray(ptr_->handle, ptr_->writable);
  }
  // operator overloading
  inline NDArray& operator=(const NDArray& other) {
    ptr_ = other.ptr_;
    return *this;
  }
  inline NDBlob* operator->() {
    return ptr_.get();
  }
  inline const NDBlob* operator->() const {
    return ptr_.get();
  }
  /*!
   * \param src The source array.
   * \return The dimension of the array
   */
  Rcpp::Dimension dim() const;
  /*!
   * \brief Return a clone of NDArray.
   *  Do not expose this to R side.
   * \return src The source NDArray.
   * \return a new cloned NDArray.
   */
  NDArray Clone() const;
  /*!
   * \return The context of NDArray.
   */
  Context ctx() const;
  /*!
   * \brief Return a slice of NDArray.
   * \param begin The begin of the slice.
   * \param end The end of the slice.
   * \return a sliced NDArray.
   */
  NDArray Slice(mx_uint begin, mx_uint end) const;
  /*!
   * \return The number of elements in the array
   */
  size_t Size() const;
  /*!
   * \return convert the NDArray to R's Array
   */
  Rcpp::NumericVector AsNumericVector() const;
  /*!
   * \brief Create NDArray from RObject
   * \param src Source object.
   * \return The created NDArray
   */
  inline static NDArray FromRObject(const Rcpp::RObject& src) {
    return NDArray(src);
  }
  /*!
   * \brief Create RObject NDArray.
   * \param handle The source handle.
   * \param writable Whether the NDArray is writable.
   * \return The created NDArray
   */
  inline static RObjectType RObject(NDArrayHandle handle, bool writable) {
    return NDArray(handle, writable).RObject();
  }
  /*!
   * \brief Move the NDArray.
   * \param src The source RObject.
   * \return The moved NDArray
   */
  inline static RObjectType Move(const Rcpp::RObject& src) {
    return NDArray(src).Move().RObject();
  }
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
  static void Save(const Rcpp::List& data,
                   const std::string& filename);
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

 private:
  /*! \brief internal pointer */
  Rcpp::XPtr<NDBlob> ptr_;
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

/*!
 * \brief An array packer that packs NDArray array together on
 *   slowest changing dimension.
 */
class NDArrayPacker {
 public:
  // constructor
  NDArrayPacker() {}
  /*!
   * \brief Push the array to the packer
   * \param nd The array to push the data into.
   */
  void Push(const NDArray::RObjectType& nd);
  /*!
   * \brief Get the R array out from packed data.
   * \return The packed data.
   */
  Rcpp::NumericVector Get() const;
  /*! \return constructor */
  static Rcpp::RObject CreateNDArrayPacker();

 private:
  /*! \brief The internal data */
  std::vector<mx_float> data_;
  /*! \brief The shape of data */
  std::vector<mx_uint> shape_;
};
}  // namespace R
}  // namespace mxnet

RCPP_EXPOSED_CLASS_NODECL(::mxnet::R::NDArrayPacker);

namespace Rcpp {
  template<>
  inline bool is<mxnet::R::NDArray>(SEXP x) {
    if (TYPEOF(x) != EXTPTRSXP) return false;
    Rcpp::XPtr<mxnet::R::NDBlob> ptr(x);
    SEXP attr = ptr.attr("class");
    return attr != R_NilValue &&
        Rcpp::as<std::string>(attr) == "MXNDArray";
    return true;
  }
}  // namespace Rcpp
#endif  // MXNET_RCPP_NDARRAY_H_
