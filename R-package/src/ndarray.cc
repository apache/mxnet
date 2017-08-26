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
    lst_names = Rcpp::as<std::vector<std::string> >(data_lst.names());
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
                                               bool allow_null,
                                               bool move_old_array) {
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
      if (move_old_array) {
        RCHECK(ptr->writable)
            << "Passing a read only NDArray to mutate function";
        ptr->moved = true;
      }
      ret[i] = ptr->handle;
    }
  }
  return ret;
}

void NDArray::CopyFromTo(const NDArray& from, NDArray* to) {
  static OpHandle copy_handle = NDArrayFunction::FindHandle("_copyto");
  NDArrayHandle from_handle = from->handle;
  NDArrayHandle to_handle = (*to)->handle;
  RCHECK(from_handle != to_handle)
      << "Attempt to copy NDArray to itself";
  NDArrayHandle* p_output_vars = &to_handle;
  int num_output = 1;
  MX_CALL(MXImperativeInvoke(copy_handle, 1, &from_handle,
                             &num_output, &p_output_vars,
                             0, nullptr, nullptr));
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

NDArrayFunction::NDArrayFunction(OpHandle handle, std::string name)
    : handle_(handle) {
  // initialize the docstring
  const char* real_name;
  const char* description;
  mx_uint num_args;
  const char **arg_names;
  const char **arg_type_infos;
  const char **arg_descriptions;
  const char *key_var_num_args;
  const char *ret_type;

  MX_CALL(MXSymbolGetAtomicSymbolInfo(
      handle_, &real_name, &description, &num_args,
      &arg_names, &arg_type_infos, &arg_descriptions,
      &key_var_num_args, &ret_type));
  if (key_var_num_args != nullptr) {
    key_var_num_args_ = key_var_num_args;
  }

  if (name[0] == '_') {
    name_ = std::string("mx.nd.internal.") + (name.c_str() + 1);
  } else {
    name_ = std::string("mx.nd.") + name;
  }
  for (size_t i = 0; i < name_.length(); ++i) {
    if (name_[i] == '_') name_[i] = '.';
  }

  // dostring: generate python style for now, change to R style later
  std::ostringstream os;
  std::string descp = description;
  if (descp.length() == 0) {
    os << name;
  } else {
    os << description;
  }
  os << "\n\n"
     << MakeDocString(num_args, arg_names, arg_type_infos, arg_descriptions)
     << "@return out The result mx.ndarray\n\n"
     << "@export\n";
  this->docstring = os.str();

  Rcpp::List arg_values(num_args + 1);
  arg_names_.resize(num_args + 1);
  arg_nd_array_.resize(num_args + 1, false);

  for (mx_uint i = 0; i < num_args; ++i) {
    arg_names_[i] = arg_names[i];
    std::string dtype = arg_type_infos[i];
    // check data type.
    if (dtype.substr(0, 7) == "NDArray" ||
        dtype.substr(0, 6) == "Symbol") {
      arg_nd_array_[i] = true;
    } else {
      // all kwargs are optional in front-end
      arg_values[i] = R_NilValue;
    }
  }
  arg_names_[num_args + 0] = "out";
  // out is are optional in front-end
  arg_values[num_args + 0] = R_NilValue;
  formals_ = arg_values;
  formals_.attr("names") = arg_names_;
}

SEXP NDArrayFunction::operator() (SEXP* args) {
  BEGIN_RCPP;

  std::vector<NDArrayHandle> nd_args;
  std::vector<std::string> sparam_vals;
  std::vector<const char*> param_keys;
  std::vector<const char*> param_vals;
  std::vector<NDArrayHandle> out_args;

  for (mx_uint i = 0; i < arg_names_.size() - 1; ++i) {
    if (arg_nd_array_[i]) {
      if (TYPEOF(args[i]) == 22) {
        nd_args.push_back(NDArray(args[i])->handle);
      } else if (TYPEOF(args[i]) == 19) {
        Rcpp::List data_lst = Rcpp::as<Rcpp::List>(args[i]);
        for (size_t k = 0; k < data_lst.size(); k++) {
          nd_args.push_back(NDArray((SEXP)data_lst[k])->handle);
        }
      }
    } else {
      if (args[i] != R_NilValue) {
        param_keys.push_back(arg_names_[i].c_str());
        sparam_vals.push_back(toPyString(arg_names_[i], args[i]));
      }
    }
  }
  param_vals.resize(sparam_vals.size());
  for (size_t i = 0; i < sparam_vals.size(); ++i) {
    param_vals[i] = sparam_vals[i].c_str();
  }
  // contain out
  if (args[arg_names_.size()-1] != R_NilValue) {
    SEXP old_output = args[arg_names_.size() - 1];
    if (TYPEOF(old_output) == VECSXP) {
      out_args = NDArray::GetHandles(old_output, "out", false, true);
    } else {
      out_args.push_back(NDArray(old_output)->handle);
    }
  }

  int num_output = static_cast<int>(out_args.size());
  NDArrayHandle* p_output_vars = nullptr;

  if (num_output != 0) {
    p_output_vars = &out_args[0];
  }

  MXImperativeInvoke(
      handle_,
      static_cast<int>(nd_args.size()),
      dmlc::BeginPtr(nd_args),
      &num_output,
      &p_output_vars,
      static_cast<int>(param_keys.size()),
      dmlc::BeginPtr(param_keys),
      dmlc::BeginPtr(param_vals));

  if (num_output == 1) {
    if (out_args.size() != 0) {
      return NDArray(args[arg_names_.size() - 1]).Move().RObject();
    } else {
      return NDArray(p_output_vars[0], true).RObject();
    }
  } else {
    Rcpp::List olist(num_output);
    for (int i = 0; i < num_output; ++i) {
      olist[i] = NDArray(p_output_vars[i], true).RObject();
    }
    return olist;
  }

  END_RCPP;
}

OpHandle NDArrayFunction::FindHandle(const std::string& hname) {
  OpHandle h;
  if (NNGetOpHandle(hname.c_str(), &h) == 0 && h != nullptr)  {
    return h;
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
inline bool ParseNDArrayArg(SEXP sexp, NDArrayHandle *handle, std::string *value) {
  switch (TYPEOF(sexp)) {
    case REALSXP: {
      *value = toString<double>(sexp);
      return false;
    }
    case INTSXP: {
      *value = toString<int>(sexp);
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

inline NDArrayHandle BinaryOp(OpHandle op, NDArrayHandle* handles) {
  int num_output = 0;
  NDArrayHandle* p_output_vars = nullptr;
  MX_CALL(MXImperativeInvoke(op, 2, handles,
                             &num_output, &p_output_vars,
                             0, nullptr, nullptr));
  RCHECK(num_output == 1);
  return p_output_vars[0];
}

inline NDArrayHandle BinaryScalarOp(
    OpHandle op, NDArrayHandle handle, const std::string &scalar) {
  int num_output = 0;
  NDArrayHandle* p_output_vars = nullptr;
  const char* skey = "scalar";
  const char* svalue = scalar.c_str();

  MX_CALL(MXImperativeInvoke(op, 1, &handle,
                             &num_output, &p_output_vars,
                             1, &skey, &svalue));
  RCHECK(num_output == 1);
  return p_output_vars[0];
}

// dispatch the binary ops of MXNDArray
NDArray::RObjectType DispatchOps(SEXP op, SEXP lhs, SEXP rhs) {
  // function handles
  static OpHandle plus = NDArrayFunction::FindHandle("_plus");
  static OpHandle plus_scalar = NDArrayFunction::FindHandle("_plus_scalar");
  static OpHandle minus = NDArrayFunction::FindHandle("_minus");
  static OpHandle minus_scalar = NDArrayFunction::FindHandle("_minus_scalar");
  static OpHandle rminus_scalar = NDArrayFunction::FindHandle("_rminus_scalar");
  static OpHandle mul = NDArrayFunction::FindHandle("_mul");
  static OpHandle mul_scalar = NDArrayFunction::FindHandle("_mul_scalar");
  static OpHandle div = NDArrayFunction::FindHandle("_div");
  static OpHandle div_scalar = NDArrayFunction::FindHandle("_div_scalar");
  static OpHandle rdiv_scalar = NDArrayFunction::FindHandle("_rdiv_scalar");
  static OpHandle mod = NDArrayFunction::FindHandle("_mod");
  static OpHandle mod_scalar = NDArrayFunction::FindHandle("_mod_scalar");
  static OpHandle rmod_scalar = NDArrayFunction::FindHandle("_rmod_scalar");
  static OpHandle equal = NDArrayFunction::FindHandle("_equal");
  static OpHandle equal_scalar = NDArrayFunction::FindHandle("_equal_scalar");
  static OpHandle not_equal = NDArrayFunction::FindHandle("_not_equal");
  static OpHandle not_equal_scalar = NDArrayFunction::FindHandle("_not_equal_scalar");
  static OpHandle greater = NDArrayFunction::FindHandle("_greater");
  static OpHandle greater_scalar = NDArrayFunction::FindHandle("_greater_scalar");
  static OpHandle greater_equal = NDArrayFunction::FindHandle("_greater_equal");
  static OpHandle greater_equal_scalar = NDArrayFunction::FindHandle("_greater_equal_scalar");
  static OpHandle lesser = NDArrayFunction::FindHandle("_lesser");
  static OpHandle lesser_scalar = NDArrayFunction::FindHandle("_lesser_scalar");
  static OpHandle lesser_equal = NDArrayFunction::FindHandle("_lesser_equal");
  static OpHandle lesser_equal_scalar = NDArrayFunction::FindHandle("_lesser_equal_scalar");
  // parse the arguments
  std::string values[2];
  NDArrayHandle handles[2];
  NDArrayHandle out = nullptr;
  bool lhs_nd = ParseNDArrayArg(lhs, &handles[0], &values[0]);
  bool rhs_nd = ParseNDArrayArg(rhs, &handles[1], &values[1]);
  RCHECK(lhs_nd || rhs_nd);
  // create output and dispatch.
  std::string sop = Rcpp::as<std::string>(op);
  switch (sop[0]) {
    case '+': {
      if (lhs_nd && rhs_nd) {
        out = BinaryOp(plus, handles);
      } else if (lhs_nd && !rhs_nd) {
        out = BinaryScalarOp(plus_scalar, handles[0], values[1]);
      } else {
        out = BinaryScalarOp(plus_scalar, handles[1], values[0]);
      }
      break;
    }
    case '-': {
      if (lhs_nd && rhs_nd) {
        out = BinaryOp(minus, handles);
      } else if (lhs_nd && !rhs_nd) {
        out = BinaryScalarOp(minus_scalar, handles[0], values[1]);
      } else {
        out = BinaryScalarOp(rminus_scalar, handles[1], values[0]);
      }
      break;
    }
    case '*': {
      if (lhs_nd && rhs_nd) {
        out = BinaryOp(mul, handles);
      } else if (lhs_nd && !rhs_nd) {
        out = BinaryScalarOp(mul_scalar, handles[0], values[1]);
      } else {
        out = BinaryScalarOp(mul_scalar, handles[1], values[0]);
      }
      break;
    }
    case '/': {
      if (lhs_nd && rhs_nd) {
        out = BinaryOp(div, handles);
      } else if (lhs_nd && !rhs_nd) {
        out = BinaryScalarOp(div_scalar, handles[0], values[1]);
      } else {
        out = BinaryScalarOp(rdiv_scalar, handles[1], values[0]);
      }
      break;
    }
    case '%': {
      if (lhs_nd && rhs_nd) {
        out = BinaryOp(mod, handles);
      } else if (lhs_nd && !rhs_nd) {
        out = BinaryScalarOp(mod_scalar, handles[0], values[1]);
      } else {
        out = BinaryScalarOp(rmod_scalar, handles[1], values[0]);
      }
      break;
    }
    case '=': {
      if (lhs_nd && rhs_nd) {
        out = BinaryOp(equal, handles);
      } else if (lhs_nd && !rhs_nd) {
        out = BinaryScalarOp(equal_scalar, handles[0], values[1]);
      } else {
        out = BinaryScalarOp(equal_scalar, handles[1], values[0]);
      }
      break;
    }
    case '!': {
      if (lhs_nd && rhs_nd) {
        out = BinaryOp(not_equal, handles);
      } else if (lhs_nd && !rhs_nd) {
        out = BinaryScalarOp(not_equal_scalar, handles[0], values[1]);
      } else {
        out = BinaryScalarOp(not_equal_scalar, handles[1], values[0]);
      }
      break;
    }
    case '>': {
      if (sop == ">=") {
        if (lhs_nd && rhs_nd) {
          out = BinaryOp(greater_equal, handles);
        } else if (lhs_nd && !rhs_nd) {
          out = BinaryScalarOp(greater_equal_scalar, handles[0], values[1]);
        } else {
          out = BinaryScalarOp(lesser_equal_scalar, handles[1], values[0]);
        }
      } else {
        if (lhs_nd && rhs_nd) {
          out = BinaryOp(greater, handles);
        } else if (lhs_nd && !rhs_nd) {
          out = BinaryScalarOp(greater_scalar, handles[0], values[1]);
        } else {
          out = BinaryScalarOp(lesser_scalar, handles[1], values[0]);
        }
      }
      break;
    }
    case '<': {
      if (sop == "<=") {
        if (lhs_nd && rhs_nd) {
          out = BinaryOp(lesser_equal, handles);
        } else if (lhs_nd && !rhs_nd) {
          out = BinaryScalarOp(lesser_equal_scalar, handles[0], values[1]);
        } else {
          out = BinaryScalarOp(greater_equal_scalar, handles[1], values[0]);
        }
      } else {
        if (lhs_nd && rhs_nd) {
          out = BinaryOp(lesser, handles);
        } else if (lhs_nd && !rhs_nd) {
          out = BinaryScalarOp(lesser_scalar, handles[0], values[1]);
        } else {
          out = BinaryScalarOp(greater_scalar, handles[1], values[0]);
        }
      }
      break;
    }
    default: {
      RLOG_FATAL << "Operator " << sop << " not supported for MXNDArray";
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
  const char** op_name_ptrs;
  std::vector<std::string> op_names;
  MX_CALL(MXListAllOpNames(&out_size, &op_name_ptrs));
  for (size_t i = 0; i < out_size; ++i) {
    op_names.push_back(std::string(op_name_ptrs[i]));
  }

  for (int i = 0; i < out_size; ++i) {
    OpHandle handle;
    MX_CALL(NNGetOpHandle(op_names[i].c_str(), &handle));
    NDArrayFunction *f = new NDArrayFunction(handle, op_names[i]);
    scope->Add(f->get_name(), f);
  }
}
}  // namespace R
}  // namespace mxnet
