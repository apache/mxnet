/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.cc
 * \brief Rcpp NDArray of MXNet.
 */
#include <Rcpp.h>
#include "./base.h"
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
    for (int k = dim - 1; k >= 0; --k) {
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
    for (int k = dim - 1; k >= 0; --k) {
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
  for (int k = 0; k < ndim; ++k) {
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

// implementation of NDArray functions
Rcpp::NumericVector NDArray::AsNumericVector() const {
  Rcpp::Dimension rshape = this->shape();
  std::vector<mx_float> temp(rshape.prod());
  MX_CALL(MXNDArraySyncCopyToCPU(
      handle_, dmlc::BeginPtr(temp), temp.size()));
  Rcpp::NumericVector ret(rshape);
  RowToColMajor(dmlc::BeginPtr(temp), Dim2Vec(rshape), temp.size(), ret.begin());
  return ret;
}

void NDArray::Save(const Rcpp::RObject &sxptr,
                   const std::string& filename) {
  if (TYPEOF(sxptr) == VECSXP) {
    Rcpp::List data_lst(sxptr);
    std::vector<std::string> lst_names = data_lst.names();
    size_t num_args = data_lst.size();
    std::vector<NDArrayHandle> handles(num_args);
    std::vector<const char*> keys(num_args);

    for (int i = 0 ; i < data_lst.size(); ++i) {
      keys[i] = lst_names[i].c_str();
      handles[i] = NDArray::XPtr(data_lst[i])->handle_;
    }
    MX_CALL(MXNDArraySave(filename.c_str(), num_args,
                          dmlc::BeginPtr(handles),
                          dmlc::BeginPtr(keys)));
  } else if (Rcpp::is<NDArray>(sxptr)) {
    MX_CALL(MXNDArraySave(filename.c_str(), 1,
                          &(NDArray::XPtr(sxptr)->handle_), nullptr));
  } else {
    RLOG_FATAL << "Only accept NDArray or list of NDArray";
  }
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
    out[i] = NDArray::RObject(out_arr[i]);
  }
  if (out_name_size != 0) {
    std::vector<std::string> lst_names(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      lst_names[i] = out_names[i];
    }
    out.attr("names") = lst_names;
  }
  return out;
}

NDArray::RObjectType NDArray::Empty(
    const Rcpp::Dimension& rshape,
    const Context::RObjectType& rctx) {
  std::vector<mx_uint> shape = Dim2Vec(rshape);
  Context ctx(rctx);
  NDArrayHandle handle;
  MX_CALL(MXNDArrayCreate(dmlc::BeginPtr(shape),
                          static_cast<mx_uint>(shape.size()),
                          ctx.dev_type, ctx.dev_id, false, &handle));
  return NDArray::RObject(handle);
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
      RCHECK(Rcpp::is<NDArray>(array_list[i]))
          << "Expect " << list_name << " to  be list of " << NDArray::TypeName();
      ret[i] = NDArray::XPtr(array_list[i])->handle_;
    }
  }
  return ret;
}

NDArray::RObjectType NDArray::Clone() const {
  std::vector<mx_uint> shape = Dim2Vec(this->shape());
  NDArrayHandle handle;
  MX_CALL(MXNDArrayCreate(dmlc::BeginPtr(shape),
                          static_cast<mx_uint>(shape.size()),
                          ctx_.dev_type, ctx_.dev_id, true, &handle));
  RObjectType ret = NDArray::RObject(handle);
  CopyFromTo(*this, NDArray::XPtr(ret));
  return ret;
}

void NDArray::CopyFromTo(const NDArray& from, NDArray* to) {
  static FunctionHandle copy_handle = NDArrayFunction::FindHandle("_copyto");
  NDArrayHandle from_handle = from.handle_;
  NDArrayHandle to_handle = to->handle_;
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
  ColToRowMajor(rdata.begin(), Dim2Vec(rshape),
                temp.size(), dmlc::BeginPtr(temp));
  MX_CALL(MXNDArraySyncCopyFromCPU(
      NDArray::XPtr(ret)->handle_,
      dmlc::BeginPtr(temp), temp.size()));
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
    MX_CALL(MXFuncGetInfo(handle, &name, &description, &num_args,
                          &arg_names, &arg_type_infos, &arg_descriptions));
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
       << "Parameters\n"
       << "----------\n"
       << MakeDocString(num_args, arg_names, arg_type_infos, arg_descriptions)
       << "Returns\n"
       << "-------\n"
       << "out : NDArray\n"
       << "    The output result of the function";
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
      begin_scalars_ = num_scalars_;
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
    use_vars[i] = NDArray::XPtr(args[begin_use_vars_ + i])->handle_;
  }

  std::vector<NDArrayHandle> mutate_vars(num_mutate_vars_);
  Rcpp::List out(num_mutate_vars_);
  for (mx_uint i = 0; i < num_mutate_vars_; ++i) {
    if (args[begin_mutate_vars_ + i] == R_NilValue) {
      if (accept_empty_out_) {
        NDArrayHandle ohandle;
        MX_CALL(MXNDArrayCreateNone(&ohandle));
        out[i] = NDArray::RObject(ohandle);
      } else {
        RLOG_FATAL << "Parameter out need to be specified";
      }
    } else {
      // move the old parameters, these are no longer valid
      RCHECK(NDArray::XPtr(args[begin_mutate_vars_ + i])->writable_)
          << "Passing a read only NDArray to mutate function";
      out[i] = NDArray::Move(args[begin_mutate_vars_ + i]);
    }
    mutate_vars[i] = NDArray::XPtr(out[i])->handle_;
  }

  MX_CALL(MXFuncInvoke(handle_,
                       dmlc::BeginPtr(use_vars),
                       dmlc::BeginPtr(scalars),
                       dmlc::BeginPtr(mutate_vars)));
  if (num_mutate_vars_ == 1) {
    return out[0];
  } else {
    return out;
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
    MX_CALL(MXFuncGetInfo(handle, &name, &description, &num_args,
                          &arg_names, &arg_type_infos, &arg_descriptions));
    if (name == hname) return handle;
  }
  RLOG_FATAL << "FindHandle: cannot find function " << hname;
  return nullptr;
}

// initialize the Rcpp module functions.
void NDArray::InitRcppModule() {
  using namespace Rcpp;  // NOLINT(*)
  class_<NDArray>("MXNDArray")
      .finalizer(&NDArray::Finalizer)
      .method("as.array", &NDArray::AsNumericVector)
      .const_method("dim", &NDArray::shape);
  function("mx.nd.internal.load", &NDArray::Load);
  function("mx.nd.internal.save", &NDArray::Save);
  function("mx.nd.internal.empty", &NDArray::Empty);
  function("mx.nd.internal.array", &NDArray::Array);
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
