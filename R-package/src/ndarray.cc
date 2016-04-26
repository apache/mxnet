/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.cc
 * \brief Rcpp NDArray of MXNet.
 */
#include <Rcpp.h>
#include "./base.h"
#include "./export.h"
#include "./ndarray.h"

namespace mxnet {
namespace R {
template<typename InputIter>
inline void ConvertLayout(InputIter it,
                          const mx_uint *ishape,
                          const size_t *ostride,
                          int dim,
                          size_t size,
                          mx_float *out_data) {
  for (size_t i = 0; i < size; ++i, ++it) {
    size_t offset = 0;
    size_t counter = i;
    for (int k = 0; k < dim; ++k) {
      size_t idx = counter % ishape[k];
      offset += idx * ostride[k];
      counter /= ishape[k];
    }
    out_data[offset] = *it;
  }
}

template<typename OutputIter>
inline void ConvertLayout(const mx_float *in_data,
                          const mx_uint *ishape,
                          const size_t *ostride,
                          int dim,
                          size_t size,
                          OutputIter it) {
  for (size_t i = 0; i < size; ++i, ++it) {
    size_t offset = 0;
    size_t counter = i;
    for (int k = 0; k < dim; ++k) {
      size_t idx = counter % ishape[k];
      offset += idx * ostride[k];
      counter /= ishape[k];
    }
    RCHECK(offset < size)
        << "offset=" << offset << ", size=" << size;
    *it = in_data[offset];
  }
}

inline std::vector<size_t> GetReverseStride(const std::vector<mx_uint>& ishape) {
  std::vector<size_t> stride(ishape.size());
  size_t prod = 1;
  int ndim = static_cast<int>(ishape.size());
  for (int k = ndim - 1; k >= 0 ; --k) {
    stride[k] = prod;
    prod *= ishape[k];
  }
  return stride;
}

template<typename InputIter>
inline void ColToRowMajor(InputIter begin,
                          const std::vector<mx_uint>& ishape,
                          size_t size,
                          mx_float *out_data) {
  int ndim = static_cast<int>(ishape.size());
  std::vector<size_t> out_stride = GetReverseStride(ishape);
  // manual unroll special constants
  const mx_uint *shape = dmlc::BeginPtr(ishape);
  const size_t *stride = dmlc::BeginPtr(out_stride);
  switch (ndim) {
    case 1: {
      ConvertLayout(begin, shape, stride, 1, size, out_data);
      break;
    }
    case 2: {
      ConvertLayout(begin, shape, stride, 2, size, out_data);
      break;
    }
    case 3: {
      ConvertLayout(begin, shape, stride, 3, size, out_data);
      break;
    }
    default: {
      ConvertLayout(begin, shape, stride, ndim, size, out_data);
      break;
    }
  }
}

template<typename OutputIter>
inline void RowToColMajor(const mx_float *in_data,
                          const std::vector<mx_uint>& ishape,
                          size_t size,
                          OutputIter begin) {
  int ndim = static_cast<int>(ishape.size());
  std::vector<size_t> out_stride = GetReverseStride(ishape);
  // manual unroll special constants
  const mx_uint *shape = dmlc::BeginPtr(ishape);
  const size_t *stride = dmlc::BeginPtr(out_stride);
  switch (ndim) {
    case 1: {
      ConvertLayout(in_data, shape, stride, 1, size, begin);
      break;
    }
    case 2: {
      ConvertLayout(in_data, shape, stride, 2, size, begin);
      break;
    }
    case 3: {
      ConvertLayout(in_data, shape, stride, 3, size, begin);
      break;
    }
    default: {
      ConvertLayout(in_data, shape, stride, ndim, size, begin);
      break;
    }
  }
}

void NDArrayPacker::Push(const NDArray::RObjectType& nd) {
  NDArray arr(nd);
  Rcpp::Dimension rshape = arr.dim();
  if (shape_.size() == 0) {
    shape_.resize(rshape.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape_[i] = rshape[i];
    }
  } else {
    RCHECK(shape_.size() == rshape.size())
        << "The number of dimension need to be matched";
    for (size_t i = 0; i < shape_.size() - 1; ++i) {
      RCHECK(shape_[i] == rshape[i])
          << "The dimension besides last need to be consistent for arrays pushed";
    }
    shape_.back() += rshape[shape_.size() - 1];
  }
  size_t begin = data_.size();
  size_t size = rshape.prod();
  data_.resize(begin + size);
  MX_CALL(MXNDArraySyncCopyToCPU(
      arr->handle, dmlc::BeginPtr(data_) + begin, size));
}

Rcpp::NumericVector NDArrayPacker::Get() const {
  Rcpp::IntegerVector sp(shape_.begin(), shape_.end());
  Rcpp::RObject sexp = sp;
  Rcpp::Dimension dim(sexp);
  Rcpp::NumericVector ret(dim);
  RCHECK(ret.size() == data_.size());
  std::copy(data_.begin(), data_.end(), ret.begin());
  return ret;
}

Rcpp::RObject NDArrayPacker::CreateNDArrayPacker() {
  return Rcpp::internal::make_new_object(new NDArrayPacker());
}

Rcpp::Dimension NDArray::dim() const {
  mx_uint ndim;
  const mx_uint *pshape;
  MX_CALL(MXNDArrayGetShape(
      ptr_->handle, &ndim, &pshape));
  Rcpp::IntegerVector dat(pshape, pshape + ndim);
  std::reverse(dat.begin(), dat.end());
  Rcpp::RObject ret = dat;
  return Rcpp::Dimension(ret);
}

NDArray NDArray::Clone() const {
  std::vector<mx_uint> shape = Dim2InternalShape(this->dim());
  Context ctx = this->ctx();
  NDArrayHandle handle;
  MX_CALL(MXNDArrayCreate(dmlc::BeginPtr(shape),
                          static_cast<mx_uint>(shape.size()),
                          ctx.dev_type, ctx.dev_id, true, &handle));
  NDArray ret(handle, true);
  CopyFromTo(*this, &ret);
  return ret;
}

Context NDArray::ctx() const {
  Context ctx;
  MX_CALL(MXNDArrayGetContext(ptr_->handle, &ctx.dev_type, &ctx.dev_id));
  return ctx;
}

size_t NDArray::Size() const {
  Rcpp::Dimension dim = this->dim();
  size_t sz = 1;
  for (size_t i = 0; i < dim.size(); ++i) {
    sz *= dim[i];
  }
  return sz;
}

NDArray NDArray::Slice(mx_uint begin, mx_uint end) const {
  NDArrayHandle out;
  MX_CALL(MXNDArraySlice(ptr_->handle, begin, end, &out));
  return NDArray(out, ptr_->writable);
}

Rcpp::NumericVector NDArray::AsNumericVector() const {
  Rcpp::Dimension rshape = this->dim();
  std::vector<mx_float> temp(rshape.prod());
  MX_CALL(MXNDArraySyncCopyToCPU(
      ptr_->handle, dmlc::BeginPtr(temp), temp.size()));
  Rcpp::NumericVector ret(rshape);
  std::copy(temp.begin(), temp.end(), ret.begin());
  return ret;
}

void NDArray::Save(const Rcpp::List& data_lst,
                   const std::string& filename) {
  std::vector<std::string> lst_names;
  if (HasName(data_lst)) {
    lst_names = data_lst.names();
  }
  size_t num_args = data_lst.size();
  std::vector<NDArrayHandle> handles(num_args);

  for (int i = 0 ; i < data_lst.size(); ++i) {
    Rcpp::RObject obj = data_lst[i];
    handles[i] = NDArray(obj)->handle;
  }
  std::vector<const char*> keys = CKeys(lst_names);
  MX_CALL(MXNDArraySave(filename.c_str(), num_args,
                        dmlc::BeginPtr(handles),
                        dmlc::BeginPtr(keys)));
}

Rcpp::List NDArray::Load(const std::string& filename) {
  mx_uint out_size;
  NDArrayHandle* out_arr;
  mx_uint out_name_size;
  const char** out_names;
  MX_CALL(MXNDArrayLoad(filename.c_str(),
                        &out_size, &out_arr,
                        &out_name_size, &out_names));
  Rcpp::List out(out_size);
  for (mx_uint i = 0; i < out_size; ++i) {
    out[i] = NDArray::RObject(out_arr[i], true);
  }
  if (out_name_size != 0) {
    std::vector<std::string> lst_names(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      lst_names[i] = out_names[i];
    }
    out.names() = lst_names;
  }
  return out;
}

NDArray::RObjectType NDArray::Empty(
    const Rcpp::Dimension& rshape,
    const Context::RObjectType& rctx) {
  std::vector<mx_uint> shape = Dim2InternalShape(rshape);
  Context ctx(rctx);
  NDArrayHandle handle;
  MX_CALL(MXNDArrayCreate(dmlc::BeginPtr(shape),
                          static_cast<mx_uint>(shape.size()),
                          ctx.dev_type, ctx.dev_id, false, &handle));
  return NDArray::RObject(handle, true);
}

std::vector<NDArrayHandle> NDArray::GetHandles(const Rcpp::List& array_list,
                                               const std::string& list_name,
                                               bool allow_null) {
  std::vector<NDArrayHandle> ret(array_list.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    if (array_list[i] == R_NilValue) {
      RCHECK(allow_null)
          << "Expect " << list_name << " to be list of non-NULL " << NDArray::TypeName();
      ret[i] = nullptr;
    } else {
      RCHECK(TYPEOF(array_list[i]) == EXTPTRSXP)
          << "Expect " << list_name << " to  be list of " << NDArray::TypeName();
      Rcpp::RObject obj = array_list[i];
      Rcpp::XPtr<NDBlob> ptr(obj);
      Rcpp::RObject attr = ptr.attr("class");
      RCHECK(attr != R_NilValue && Rcpp::as<std::string>(attr) == "MXNDArray")
          << "Expect " << list_name << " to  be list of " << NDArray::TypeName();
      ret[i] = ptr->handle;
    }
  }
  return ret;
}

void NDArray::CopyFromTo(const NDArray& from, NDArray* to) {
  static FunctionHandle copy_handle = NDArrayFunction::FindHandle("_copyto");
  NDArrayHandle from_handle = from->handle;
  NDArrayHandle to_handle = (*to)->handle;
  RCHECK(from_handle != to_handle)
      << "Attempt to copy NDArray to itself";
  MX_CALL(MXFuncInvoke(copy_handle, &from_handle, nullptr, &to_handle));
}

NDArray::RObjectType NDArray::Array(
    const Rcpp::RObject& src,
    const Context::RObjectType& ctx) {
  Rcpp::NumericVector rdata(src);
  Rcpp::RObject dim = rdata.attr("dim");
  Rcpp::Dimension rshape(dim);
  RObjectType ret = NDArray::Empty(rshape, ctx);
  std::vector<mx_float> temp(rdata.size());
  std::copy(rdata.begin(), rdata.end(), temp.begin());
  MX_CALL(MXNDArraySyncCopyFromCPU(
      NDArray(ret)->handle,
      dmlc::BeginPtr(temp), rdata.size()));
  return ret;
}

NDArrayFunction::NDArrayFunction(FunctionHandle handle)
    : handle_(handle) {
  // initialize the docstring
  {
    const char *name;
    const char *description;
    mx_uint num_args;
    const char **arg_names;
    const char **arg_type_infos;
    const char **arg_descriptions;
    const char *ret_type;
    MX_CALL(MXFuncGetInfo(handle, &name, &description, &num_args,
                          &arg_names, &arg_type_infos, &arg_descriptions,
                          &ret_type));
    if (name[0] == '_') {
      name_ = std::string("mx.nd.internal.") + (name + 1);
    } else {
      name_ = std::string("mx.nd.") + name;
    }
    for (size_t i = 0; i < name_.length(); ++i) {
      if (name_[i] == '_') name_[i] = '.';
    }
    // dostring: generate python style for now, change to R style later
    std::ostringstream os;
    os << description << "\n\n"
       << MakeDocString(num_args, arg_names, arg_type_infos, arg_descriptions)
       << "@return out The result mx.ndarray\n\n"
       << "@export\n";
    this->docstring = os.str();
  }
  // initialize the function information
  {
    const int kNDArrayArgBeforeScalar = 1;
    const int kAcceptEmptyMutateTarget = 1 << 2;
    int type_mask;

    MX_CALL(MXFuncDescribe(
        handle, &num_use_vars_, &num_scalars_,
        &num_mutate_vars_, &type_mask));
    if ((type_mask & kNDArrayArgBeforeScalar) != 0) {
      begin_use_vars_ = 0;
      begin_scalars_ = num_use_vars_;
    } else {
      begin_use_vars_ = num_scalars_;
      begin_scalars_ = 0;
    }
    begin_mutate_vars_ = num_use_vars_ + num_scalars_;
    num_args_ = num_use_vars_ + num_scalars_ + num_mutate_vars_;
    accept_empty_out_ = ((type_mask & kAcceptEmptyMutateTarget) != 0) && num_mutate_vars_ == 1;
  }

  // construct formals
  {
    Rcpp::List arg_values(num_args_);
    std::vector<std::string> arg_names(num_args_);
    for (mx_uint i = 0; i < num_use_vars_; ++i) {
      std::ostringstream os;
      os << "X" << (i + 1);
      arg_names[begin_use_vars_ + i] = os.str();
    }
    for (mx_uint i = 0; i < num_scalars_; ++i) {
      std::ostringstream os;
      os << "s" << (i + 1);
      arg_names[begin_scalars_ + i] = os.str();
    }
    if (accept_empty_out_) {
      arg_names[begin_mutate_vars_] = "out";
      // this is really optional
      arg_values[begin_mutate_vars_] = R_NilValue;
    } else {
      for (mx_uint i = 0; i < num_mutate_vars_; ++i) {
        std::ostringstream os;
        os << "out" << (i + 1);
        arg_names[begin_mutate_vars_ + i] = os.str();
      }
    }
    formals_ = arg_values;
    formals_.attr("names") = arg_names;
  }
}

SEXP NDArrayFunction::operator() (SEXP* args) {
  BEGIN_RCPP;

  std::vector<mx_float> scalars(num_scalars_);
  std::vector<NDArrayHandle> use_vars(num_use_vars_);

  for (mx_uint i = 0; i < num_scalars_; ++i) {
    scalars[i] = Rcpp::as<mx_float>(args[begin_scalars_ + i]);
  }
  for (mx_uint i = 0; i < num_use_vars_; ++i) {
    use_vars[i] = NDArray(args[begin_use_vars_ + i])->handle;
  }

  std::vector<NDArrayHandle> mutate_vars(num_mutate_vars_);
  std::vector<NDArray> out;
  out.reserve(num_mutate_vars_);

  for (mx_uint i = 0; i < num_mutate_vars_; ++i) {
    if (args[begin_mutate_vars_ + i] == R_NilValue) {
      if (accept_empty_out_) {
        NDArrayHandle ohandle;
        MX_CALL(MXNDArrayCreateNone(&ohandle));
        mutate_vars[i] = ohandle;
        out.push_back(NDArray(ohandle, true));
      } else {
        RLOG_FATAL << "Parameter out need to be specified";
      }
    } else {
      // move the old parameters, these are no longer valid
      NDArray nd(args[begin_mutate_vars_ + i]);
      mutate_vars[i] = nd->handle;
      out.push_back(nd.Move());
    }
  }

  MX_CALL(MXFuncInvoke(handle_,
                       dmlc::BeginPtr(use_vars),
                       dmlc::BeginPtr(scalars),
                       dmlc::BeginPtr(mutate_vars)));
  if (num_mutate_vars_ == 1) {
    return out[0].RObject();
  } else {
    Rcpp::List olist(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      olist[i] = out[i].RObject();
    }
    return olist;
  }
  END_RCPP;
}

FunctionHandle NDArrayFunction::FindHandle(const std::string& hname) {
  mx_uint out_size;
  FunctionHandle *arr;
  MX_CALL(MXListFunctions(&out_size, &arr));
  for (int i = 0; i < out_size; ++i) {
    FunctionHandle handle = arr[i];
    const char *name;
    const char *description;
    mx_uint num_args;
    const char **arg_names;
    const char **arg_type_infos;
    const char **arg_descriptions;
    const char *ret_type;
    MX_CALL(MXFuncGetInfo(handle, &name, &description, &num_args,
                          &arg_names, &arg_type_infos, &arg_descriptions,
                          &ret_type));
    if (name == hname) return handle;
  }
  RLOG_FATAL << "FindHandle: cannot find function " << hname;
  return nullptr;
}


// internal namespace of functions inside
namespace ndarray {
/*!
 * \brief internal function to parse NDArray arguments
 * \param sexp The soure value
 * \param handle the output handle, if it is NDArray type.
 * \param value the output value, if it is numeric type.
 * \return whether it is NDArray type
*/
inline bool ParseNDArrayArg(SEXP sexp, NDArrayHandle *handle, float *value) {
  switch (TYPEOF(sexp)) {
    case REALSXP: {
      *value = static_cast<float>(Rcpp::as<double>(sexp));
      return false;
    }
    case INTSXP: {
      *value = static_cast<float>(Rcpp::as<int>(sexp));
      return false;
    }
    case EXTPTRSXP: {
      Rcpp::XPtr<NDBlob> ptr(sexp);
      Rcpp::RObject attr = ptr.attr("class");
      RCHECK(attr != R_NilValue && Rcpp::as<std::string>(attr) == "MXNDArray")
          << "MXNDArray binary operations only support NDArray and numeric values";
      RCHECK(!ptr->moved)
          << "Passing in an NDArray that has been moved";
      *handle = ptr->handle;
      return true;
    }
    default: {
      RLOG_FATAL << "MXNDArray binary operations only support "
                 << "NDArray and numeric values as operands";
    }
  }
  return true;
}

// dispatch the binary ops of MXNDArray
NDArray::RObjectType DispatchOps(SEXP op, SEXP lhs, SEXP rhs) {
  RLOG_INFO << "Why";
  // function handles
  static FunctionHandle plus = NDArrayFunction::FindHandle("_plus");
  static FunctionHandle plus_scalar = NDArrayFunction::FindHandle("_plus_scalar");
  static FunctionHandle minus = NDArrayFunction::FindHandle("_minus");
  static FunctionHandle minus_scalar = NDArrayFunction::FindHandle("_minus_scalar");
  static FunctionHandle rminus_scalar = NDArrayFunction::FindHandle("_rminus_scalar");
  static FunctionHandle mul = NDArrayFunction::FindHandle("_mul");
  static FunctionHandle mul_scalar = NDArrayFunction::FindHandle("_mul_scalar");
  static FunctionHandle div = NDArrayFunction::FindHandle("_div");
  static FunctionHandle div_scalar = NDArrayFunction::FindHandle("_div_scalar");
  static FunctionHandle rdiv_scalar = NDArrayFunction::FindHandle("_rdiv_scalar");
  // parse the arguments
  float values[2];
  NDArrayHandle handles[2], out;
  bool lhs_nd = ParseNDArrayArg(lhs, &handles[0], &values[0]);
  bool rhs_nd = ParseNDArrayArg(rhs, &handles[1], &values[1]);
  RCHECK(lhs_nd || rhs_nd);
  // create output and dispatch.
  MX_CALL(MXNDArrayCreateNone(&out));
  std::string sop = Rcpp::as<std::string>(op);
  switch (sop[0]) {
    case '+': {
      if (lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(plus, handles, nullptr, &out));
      } else if (lhs_nd && !rhs_nd) {
        MX_CALL(MXFuncInvoke(plus_scalar, &handles[0], &values[1], &out));
      } else if (!lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(plus_scalar, &handles[1], &values[0], &out));
      }
      break;
    }
    case '-': {
      if (lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(minus, handles, nullptr, &out));
      } else if (lhs_nd && !rhs_nd) {
        MX_CALL(MXFuncInvoke(minus_scalar, &handles[0], &values[1], &out));
      } else if (!lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(rminus_scalar, &handles[1], &values[0], &out));
      }
      break;
    }
    case '*': {
      if (lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(mul, handles, nullptr, &out));
      } else if (lhs_nd && !rhs_nd) {
        MX_CALL(MXFuncInvoke(mul_scalar, &handles[0], &values[1], &out));
      } else if (!lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(mul_scalar, &handles[1], &values[0], &out));
      }
      break;
    }
    case '/': {
      if (lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(div, handles, nullptr, &out));
      } else if (lhs_nd && !rhs_nd) {
        MX_CALL(MXFuncInvoke(div_scalar, &handles[0], &values[1], &out));
      } else if (!lhs_nd && rhs_nd) {
        MX_CALL(MXFuncInvoke(rdiv_scalar, &handles[1], &values[0], &out));
      }
      break;
    }
    default: {
      RLOG_FATAL << "Operator " << sop << "not supported for MXNDArray";
    }
  }
  return NDArray::RObject(out, true);
}

Rcpp::Dimension dim(const NDArray::RObjectType& src) {
  return NDArray(src).dim();
}

Context::RObjectType ctx(const NDArray::RObjectType& src) {
  return NDArray(src).ctx().RObject();
}

unsigned long Size(const NDArray::RObjectType& src) {  // NOLINT(*)
  return NDArray(src).Size();
}

Rcpp::NumericVector AsNumericVector(const NDArray::RObjectType& src) {
  return NDArray(src).AsNumericVector();
}

NDArray::RObjectType Slice(const NDArray::RObjectType& src,
                           mx_uint begin, mx_uint end) {
  NDArray nd(src);
  Rcpp::Dimension dim = nd.dim();
  size_t ndim = dim.size();
  RCHECK(dim[ndim - 1] >= end)
      << "end=" << end << ", max-dim=" << dim[ndim - 1];
  return nd.Slice(begin, end).RObject();
}
}  // namespace ndarray

// initialize the Rcpp module functions.
void NDArray::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  function("mx.nd.slice", &ndarray::Slice);
  function("mx.nd.internal.load", &NDArray::Load);
  function("mx.nd.internal.save", &NDArray::Save);
  function("mx.nd.internal.array", &NDArray::Array);
  function("mx.nd.internal.empty.array", &NDArray::Empty);
  function("mx.nd.internal.dispatch.Ops", &ndarray::DispatchOps);
  // exposing members
  function("mx.nd.internal.dim", &ndarray::dim);
  function("mx.nd.internal.ctx", &ndarray::ctx);
  function("mx.nd.internal.length", &ndarray::Size);
  function("mx.nd.internal.as.array", &ndarray::AsNumericVector);

  class_<NDArrayPacker>("NDArrayPacker")
      .method("push", &NDArrayPacker::Push)
      .method("get", &NDArrayPacker::Get);
  function("mx.nd.arraypacker", &NDArrayPacker::CreateNDArrayPacker);
}

void NDArrayFunction::InitRcppModule() {
  Rcpp::Module* scope = ::getCurrentScope();
  RCHECK(scope != nullptr)
      << "Init Module need to be called inside scope";
  mx_uint out_size;
  FunctionHandle *arr;
  MX_CALL(MXListFunctions(&out_size, &arr));
  for (int i = 0; i < out_size; ++i) {
    NDArrayFunction *f = new NDArrayFunction(arr[i]);
    scope->Add(f->get_name(), f);
  }
}
}  // namespace R
}  // namespace mxnet
